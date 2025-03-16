import numpy as np
import weakref
import contextlib
import dezero

try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError as e:
    print(e)
    array_types = (np.ndarray)

class Config:
    enable_backprop = True



class Function:
    '''
    DeZero函数的输入是Variable实例或ndarray实例，输出是Variable实例。如果函数继承自Function类（如Reshape)，ndarray实例会在该函数类的__call__方法中自动转换为Variable实例。
    '''
    def __call__(self, *inputs):
        #step21 运算符重载
        inputs = [as_variable(x) for x in inputs]

        # forward pass
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        # 是否启动反向传播
        if Config.enable_backprop:
        # ======== BP code ==========
            # func的generation就是inputs的gen中最大的那个
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self) # 设置前后连接，self就是函数本身
            self.inputs = inputs 
            # 注意观察上面的outputs是如何生成的，就可以理解下面这里的列表推导式
            # 函数输出变量这一环节使用弱引用，打破循环应用
            self.outputs = [weakref.ref(output) for output in outputs]# original: self.outputs = outputs
            # self.outputs保存了反向传播需要的东西
        # ========= BP code end ==========
        return outputs if len(outputs) > 1 else outputs[0]
    def forward(self,x):
        raise NotImplementedError('Function.forward not implemented')
    
    def backward(self, gy):
        raise NotImplementedError('Function.backward not implemented')

class Variable(object):

    # step21：运算符重载，调高实例调用运算符优先级，高者优先调用
    __array_priority__ = 10086
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0
    
    def set_creator(self, func):
        self.creator = func
        # 在正向传播(__call__)的过程中对变量对output设定generation，为什么只对output？：因为只有output会被调用set_creator
        self.generation = func.generation + 1

    # create_graph =False，意思是不对第一次反向传播的计算创建计算图
    # 但是第一次反向传播的计算图还是会继续创建的
    def backward(self, retain_grad = False, create_graph=False):
        if self.grad is None:
            # Modified
            # self.grad = np.ones_like(self.data)
            self.grad = Variable(np.ones_like(self.data))
        funcs = []
        # 用于防止同一个函数被多次添加到funcs中，从而防止一个函数的backward方法被错误地多次调用: 图论有关的算法常用技巧，用来防止cycle
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)
        
        while funcs:
            f = funcs.pop()
            # gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs]
            # 下面这行with语句决定是否关闭Function中的计算的反向传播：关闭反向传播保存中间变量+关闭创建计算图，仅保留第一次反向传播的结果，因为gxs=f.backward(*gys)一定会计算导数
            # 第一次反向传播一定会进行，但是第一次反向传播的进行（也即第一次反向传播内部的计算）涉及到的计算在关闭反向传播的状态下进行。
            with using_config('enable_backprop', create_graph):

                # ==== 反向传播在此启动 ====
                gxs = f.backward(*gys)
                # 注意，这里传入的gys就是后面所有Functions的backward方法里面的gy
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx
                    if x.creator is not None:
                        add_func(x.creator)
            # retain grad adaptation：在backwards计算完grad以后，删除输出的grad（y.grad）,保留x.grad继续反向传播
            if not retain_grad:
                for y in f.outputs:
                    # f.outputs里包括了除了最开始输入的x以外的所有中间变量Variable
                    # 注意，f.outputs 是弱引用，因为f.outputs就是Function类里面的self.outputs，which已经被改造成弱引用
                    y().grad = None

    def cleargrad(self):
        self.grad = None

    # 新增shape\ndim\size等方法，让Variable实例和ndarray实例一样好用
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)# 返回data dim=0 的元素数量

    # 重写__repr__方法，自定义print(Variable)输出的内容
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        
        p = str(self.data).replace('\n', '\n' + ' '*9)
        return 'variable(' + p +')'
    
    # 重载 * 运算符
    def __mul__(self, other):
        # 在使用 * 计算Variable类实例时，调用的就是__mul__方法，此时，运算符 * 左侧的a作为self参数，右侧的b作为other参数传给了__mul__方法，b在 * 的右侧，调用的特殊方法是__rmul__
        return mul(self, other)
    # 运算符左右对称性改造
    def __rmul__(self, other):
        return mul(self, other)
    # 重载 + 运算符
    def __add__(self, other):
        return add(self, other)
    
    def __radd__(self, other):
        return add(self, other)
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    # less general transpose
    # def transpose(self):
    #     return dezero.funcstions.transpose(self)

    # more general transpose
    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        
        return dezero.functions.transpose(self, axes)
    # 上面这个transpose的构造有点tricky，我们来看数据x是如何流动的
    # 首先 从变量实例call：Variable.transpose((1,0,2))
    # Variable.transpose对传进来的axes进行处理（兼容各种不同的写法），把处理过后的axes以及数据本身(self)传入dezero.functions.transpose
    # 接着由functions.transpose内部实例化一个Transpose类
    # 因为Transpose类是Function类的子类，因此实际执行的是dezero.core.Function类里面的__call__方法
    # Variable进入__call__方法后首先会初始化self.axes=axes，因为Variable里面的tensor是以[ndarray]的列表的形式存在的，当__call__运行到ys.forward(*xs)后，list[ndarray]又会被解包，变成一个一个的ndarray，forward里面又是y = x.transpose(self.axes)，也就是用的ndarray的transpose方法，这才把Variable成功transpose
    # # all make sense. remember, batched data always in the 0th dim of Variable: xs = [Variable(np.random.randn(2,2,3)).data], xs can be viewed as a batchsize=2, 2-d matrix
    
    # same for 2 axes or multi axes
    @property
    def T(self):
        # so that we can transpose with Variable.T
        return dezero.functions.transpose(self)
     #使sum函数也可以作为Variable的方法使用
    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)


# no need to modified add method
class Add(Function):
    def forward(self, x0, x1):
        # 把原输入的shape保存下来，反向的时候会用，因为加法会损失信息，所以要保存
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y # 返回一个元素而不需要返回元组
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        # 如果两个值不等，那就需要broadcast，broadcast_to的反向是sum_to因此需要用sum_to见page 273图40-1
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Square(Function):
    def forward(self, x):
        y = x**2
        return y
    
    def backward(self, gy):
        # modified
        # x = self.inputs[0].data
        x, = self.inputs
        gx = 2*x*gy
        return gx

# modified Variable
class Mul(Function):
    def forward(self, x0, x1):
        # 这里不需要像Add那样保存x0, x1的shape是因为这样内存消耗少，反正backward的计算都需要直接用到x0和x1本身而不仅仅是他俩的shape，那干脆backward再引进x0,x1
        y = x0*x1
        return y
    def backward(self, gy):
        # 因为Mul继承了Function类，因此也继承了Function类的inputs/data
        # modified
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        x0, x1 = self.inputs
        gx0 = gy*x1
        gx1 = gy*x0
        if x0.shape != x1.shape:
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        x0, x1 = self.inputs
        return gx0, gx1

# no need to modify
class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
# no need to modify
class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0-x1
        return y
    
    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)

        return gx0, gx1
    
class Div(Function):
    def forward(self, x0, x1):
        y = x0/x1
        return y
    def backward(self, gy):
        # self.inputs来源: Div()(x0, x1)
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        # modified
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


class Pow(Function):
    def __init__(self, c):
        self.c = c
    def forward(self, x):
        y = x**self.c
        return y
    def backward(self, gy):
        # modified
        # x = self.inputs[0].data
        x, = self.inputs
        c = self.c
        gx = c*x**(c-1)*gy
        return gx


@contextlib.contextmanager
def using_config(name, value):

    # get the value of Config.name
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

# make it easier to use
def no_grad():
    return using_config('enable_backprop', False)

'''
with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)
'''


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
# step 21, 运算符重载：为了让Variable能供兼容ndarray的计算
def as_variable(obj):
    '''
    如果 obj 是Variable 实例， 则不做任何修改 直拨返。否则，将只转换为Variable实例
    '''
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)
def neg(x):
    return Neg()(x)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0) # 交换x1和x0
def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

def add(x0, x1):
    # step21 运算符重载：与float和int一起使用
    x1 = as_array(x1)

    return Add()(x0, x1)


def square(x):
    f = Square()
    return f(x)
    

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


def pow(x, c):
    return Pow(c)(x)

'''
with using_config('enable_backprop', False):
    x = Variable(np.array(2.0))
    y = square(x)
'''

def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = dezero.functions.get_item

    Variable.matmul = dezero.functions.matmul
    Variable.dot = dezero.functions.matmul
    Variable.max = dezero.functions.max
    Variable.min = dezero.functions.min


class Parameter(Variable):
    pass

