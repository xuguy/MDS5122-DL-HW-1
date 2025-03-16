from dezero.core import Parameter
import weakref, os
import numpy as np
import dezero.functions as F
from dezero import cuda
from dezero.utils import pair


# base class
class Layer:
    # Layer这个基类定义了一些所有Layer都会有的attribute和method，例如为了方便管理参数而设定的params()方法，cleargrads()方法
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        # __setattr__是一个在实例化Layer时自动调用的方法，我们这里重新定义了__setattr__的逻辑，当实例化Layer后，该实例也会自动调用这个方法，虽然我们重写了这个方法的逻辑，但后面又继承了base类的__setattr__方法，因此__setattr__方法依旧会起效，只是我们规定，在他起效前增加一个判断语句
        # 另外，一个class的所有实例变量都会被__setattr__自动以字典形式存到实例变量__dict__中，其中实例变量的名字为name，变量的值为value，以{name:value}形式保存
        # 只有当value是Parameter实例时才向self._params增加name
        if isinstance(value, (Parameter, Layer)):
            # Parameter和Layer类自身都可以作为参数
            self._params.add(name)
        #只有Parameter类实例变量的name会被添加到self._params
        #但所有实例变量都会被添加到__dict__中，到时候按需要取出即可
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        # 这个call方法作用可以参考Function：call定义层类的计算行为，并规范化输入和输出。

        # forward方法将会在继承Layer类的子类中实现
        # 这里可以与Function类的实现做一个对比，Function类中，如果不需要反向传播（例如推理模式），实例不会保留inputs和outputs，仅仅做计算并return outputs，然后所有局部变量（非self.var)就会被删除
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        #  弱引用：不增加被引用对象的引用计数，因此被引用对象（局部变量inputs/outputs）在用完后就会被删除(回收)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        # 
        return outputs if len(outputs)>1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError()
    
    def params(self):
        # 按顺序逐个取出Layer实例所有的Parameter实例
        # 取出Layer实例_params中所有的参数，_params中的参数原来只有Parameter实例，现在又扩充了Layer实例
        for name in self._params:
            
            #取出参数obj
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                #如果参数obj是Layer实例，那么就从Layer实例obj中递归地取出所有参数
                # 我们通过 yield from 来使用一个生成器创建另一个新的生成器
                # print(name)
                yield from obj.params()
            else:
                #否则，也就是obj是Parameter实例，那么就从obj = self.__dict__[name]中取出参数
                yield obj

    def cleargrads(self):
        # reset all Parameters' grad
        for param in self.params():
            param.cleargrad()

    # cupy adaptation
    # 作用对象是Layer中的params()
    def to_cpu(self):
        for param in self.params():
            # param是继承自Variable类的Parameter类，把层中的param拿出来修改后，会直接改动层中的对应的param；且Variable类同Parameter类同样具备to_cpu方法
            param.to_cpu()
    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, params_dict, parent_key=''):
        for name in self._params:

            obj = self.__dict__[name]

            # 命名规则：如果有子layer，那么用'/'分割parent和child，如果没有自layer（直接获取到的是Parameter），那么就直接用该parameter的名字
            key = parent_key + '/' + name if parent_key else name
            #如果取出的是Layer，则递归调用
            # 如果取出的是Parameter，那么就把该Parameter加到param_dict中
            if isinstance(obj, Layer):
                # 递归遍历
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):

        # make sure Parameters are all in cpu RAM, so that we could use np.savez
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)

        # 取出数据（np.ndarray），这样才能调用savez保存
        # 注意，param.data里面是一个np.array数组
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}

        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            # 检查path是否存在，如存在则删除
            if os.path.exists(path):
                os.remove(path)
            raise
    
    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        # _flatten_params是为了获取原Layer的结构，这要求我们要有和参数对应的一模一样的model
        # 用load来的数据逐个替换对应key名下的数据
        for key, param in params_dict.items():
            param.data = npz[key]

#作为Layer的Linear类，而不是作为函数的Linear类
class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__() #激活Layer的init
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name = 'W')
        # 上面先设定为None（self.W.data=None），在forward方法中再创建权重（延迟创建权重W的时间），这样就能自动确定Linear类的输入大小（in_size)而无需用户指定
        # 如果in_size是None，那么下面这个if判断判断为假，不处理_init_W()，也就是不初始化权重，留到forward里再初始化
        # 然后再给Parameter的name属性(attribute)标记上'W'
        # Parameter 的name属性继承自Variable，这样self.W这个Parameter类的实例就有一个name属性，我们后续就可以通过self.Parameter.__dict__筛选不同名字的Parameter实例
        if self.in_size is not None:
            self._init_W()
        
        if nobias:
            self.b = None

        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name = 'b')
            # print(self.b)
            # self.b.data
    
    # 初始化W的方法（初始化，即往原来是W.data==None的self.W中传入具体的非None数据）
    def _init_W(self, xp=np):
        # out_size已知，而in_size可能未知
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype)*np.sqrt(1/I)
        self.W.data = W_data

    def forward(self, x):
        # forward将会根据输入x的shape创建权重数据
        # x是被传入的数据，x.shape[1]就是Linear层参数的in_size
        # 我们只需要按照layer=Linear(100)的方式指定输出大小即可
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)
        # y = xW+b
        y = F.linear(x, self.W, self.b)
        return y


    
# ============ conv2d ===============
# 此Conv2d是Layer，是专门用来管理卷积层的参数的，注意与functions_conv中的Conv2d区分：该Conv2d定义了卷积核的计算
class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride = 1, pad = 0, nobias = False, dtype=np.float32, in_channels=None):
        super().__init__()

        # 输入数据的channels数：int/None，如果是None，那么in_channels的值将在下面的forward(x)中的x的形状中获得
        self.in_channels = in_channels

        # 输出数据的channels数: int
        self.out_channels = out_channels

        # int / (int, int)
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

        # numpy.dtype
        self.dtype = dtype

        self.W = Parameter(None, name = 'W')
        
        # 按照惯例判断是否指定了in_channels(与之对应的是自动分配)，如果未指定了in_channels，那么调用特制的_init_W方法初始化W
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None

        else:
            self.b = Parameter(np.zeros(out_channels, dtype = dtype), name = 'b')

    # conv2d里面的_init_W允许我们在runtime中自动初始化参数矩阵W，这样我们初始化Conv2d的时候就不需要非得初始化W，节省内存
    def _init_W(self, xp = np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)

        # kind of dropout 
        scale = np.sqrt(1/(C*KH*KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype)*scale

        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            #先获取in_channels的值
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            # 再进行初始化
            self._init_W(xp)
        # bugged here when updating params
        # check if F.conv2dv works as expected
        # y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        y = F.conv2dv(x, self.W, self.b, self.stride, self.pad)

        return y
    

class Conv2dV(Layer):
    def __init__(self, out_channels, kernel_size, stride = 1, pad = 0, nobias = False, dtype=np.float32, in_channels=None):
        super().__init__()

        # 输入数据的channels数：int/None，如果是None，那么in_channels的值将在下面的forward(x)中的x的形状中获得
        self.in_channels = in_channels

        # 输出数据的channels数: int
        self.out_channels = out_channels

        # int / (int, int)
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

        # numpy.dtype
        self.dtype = dtype

        self.W = Parameter(None, name = 'W')
        
        # 按照惯例判断是否指定了in_channels(与之对应的是自动分配)，如果未指定了in_channels，那么调用特制的_init_W方法初始化W
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None

        else:
            self.b = Parameter(np.zeros(out_channels, dtype = dtype), name = 'b')

    def _init_W(self, xp = np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)

        # kind of dropout 
        scale = np.sqrt(1/(C*KH*KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype)*scale

        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            #先获取in_channels的值
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            # 再进行初始化
            self._init_W(xp)

        y = F.conv2dv(x, self.W, self.b, self.stride, self.pad)

        return y


# RNN
# 注意，所谓的Layer，只是一个保存Variable的地方，他承担的主要角色是 1) 保存Parameter 2）定义数据的传输方式，也即forward，目的是为了形成计算图，因此不需要backward方法，因为gradient可以通过计算图直接反向传播得到，虽然这样的效率不如直接写一个backward方法，但是具备通用性。
class RNN(Layer):
    def __init__(self, hidden_size, in_size = None):
        super().__init__()
        self.x2h = Linear(hidden_size, in_size = in_size)
        self.h2h = Linear(hidden_size, in_size = in_size, nobias = True)
        # 保存隐藏状态，通过隐藏状态与之前的计算图建立连接
        '''
        每个时间步的梯度不仅来自当前步的输出损失，还来自下一个时间步的隐藏状态的梯度。因为h_t依赖于h_{t-1}，所以在计算h_{t-1}的梯度时，必须考虑到h_t的梯度会传递到h_{t-1}。也就是说，梯度会沿着时间步反向传播，每个隐藏状态的梯度由两部分组成：当前时间步的输出损失带来的梯度，以及下一个时间步隐藏状态传递回来的梯度。

        隐藏状态在每个时间步被更新，但梯度是通过反向传播按时间步依次计算的。每个时间步的隐藏状态的梯度不仅来自当前步的损失，还来自后续时间步的梯度，这样就能将梯度传递到旧的隐藏状态，从而更新它们的参数。所以，虽然隐藏状态在每个时间步被覆盖，但反向传播过程中，梯度会沿着时间链式传播，从而影响到之前的参数。

        举个例子，假设我们有三个时间步t=1,2,3。计算h3的梯度时，它会影响第三个时间步的损失，同时h3又依赖于h2，所以在反向传播时，h2的梯度会包括来自h3的梯度部分。同理，h1的梯度来自h2的梯度，依此类推。这样，即使每个时间步的隐藏状态被更新了，梯度仍然可以通过链式法则追溯到之前的隐藏状态。

        通过BPTT，每个参数的梯度是所有时间步贡献的累加。的最终梯度是各个时间步通过链式法则传递的梯度之和。

        链式法则的计算见：
        https://d2l.ai/chapter_recurrent-neural-networks/bptt.html
        '''
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        # self.h is None 说明是0时刻，此时还没有hidden state，只有第一个输入数据x，需要初始化一个，初始化的方法就是xW_x
        if self.h is None:
             h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))
        # 更新hidden state
        self.h = h_new
        return h_new
    

# ===== batchnorm, direct migrate unverified =========
class BatchNorm2d(Layer):
    def __init__(self):
        super().__init__()
        # `.avg_mean` and `.avg_var` are `Parameter` objects, so they will be
        # saved to a file (using `save_weights()`).
        # But they don't need grads, so they're just used as `ndarray`.
        self.avg_mean = Parameter(None, name='avg_mean')
        self.avg_var = Parameter(None, name='avg_var')
        self.gamma = Parameter(None, name='gamma')
        self.beta = Parameter(None, name='beta')

    def _init_params(self, x):
        xp = cuda.get_array_module(x)
        D = x.shape[1]
        if self.avg_mean.data is None:
            self.avg_mean.data = xp.zeros(D, dtype=x.dtype)
        if self.avg_var.data is None:
            self.avg_var.data = xp.ones(D, dtype=x.dtype)
        if self.gamma.data is None:
            self.gamma.data = xp.ones(D, dtype=x.dtype)
        if self.beta.data is None:
            self.beta.data = xp.zeros(D, dtype=x.dtype)

    def __call__(self, x):
        if self.avg_mean.data is None:
            self._init_params(x)
        return F.batch_nrom(x, self.gamma, self.beta, self.avg_mean.data, self.avg_var.data)
