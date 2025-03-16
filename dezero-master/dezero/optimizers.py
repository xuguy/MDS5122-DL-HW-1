import math
from dezero import cuda, Parameter


# base class
class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    # target 指的是model
    def setup(self, target):
        self.target = target
        # return self这种写法把class自己返回，这样我们就可以写连续调用语法：optimizer.SGD(...).setup(...)
        return self
    
    def update(self):
        # collect non-empty grad parameters
        params = [p for p in self.target.params() if p.grad is not None]

        # 参数预处理：hooks are pre-defined preprocess function (weight decay for example), its optional
        for f in self.hooks:
            f(params)
        # update params
        for param in params:
            # call update_one() method, which will be implemented in child class
            self.update_one(param)

    def update_one(self, param):
        raise NotImplementedError()
    
    def add_hook(self, f):
        self.hooks.append(f)


# SGD Optimizer
# load from external by from dezero.optimizers import SGD
class SGD(Optimizer):

    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr*param.grad.data


# SGD with Momentum
import numpy as np

class MomentumSGD(Optimizer):
    # momentum aka alpha
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum

        #这里引入了速度变量但我们为什么需要一个字典来保存这些速度呢？直接当场计算当场更新不就完事了吗？
        # 参数独立性：神经网络中每个参数（如各层的权重、偏置）需要独立的速度变量。不同参数的速度不能共享。回顾我们之前发现的iterator与Set()结合后带来的随机性，我们无法保证每一次通过yield from遍历所有层后取出的参数的顺序都和上一次遍历取出的参数的顺序一样; 然而我们更新速度需要用到上一次的速度的信息，既然每次遍历的顺序都可能不一样，你只有通过id才能取出和参数对应的速度

        # 还有另外几种好处，例如1) 惰性初始化：仅在首次遇到参数时初始化速度，避免预先为所有参数分配内存（节省资源，尤其对大型模型) 2) 防止重复创建：避免多次初始化导致速度历史信息丢失。
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            # self.vs这个字典的key为了满足唯一性，选择用Layer的param的内存地址id
            # 第一次调用update_one时，self.vs（保存速度的字典）是空的，开始调用后，下面这行代码会自动创建和param.data也就是W形状相同的ndarray，用于保存并更新W
            # updata_one()被用在update()里面的for循环里，for的对象是model.params()里的所有Layer的params,相当于在每次反向传播后（Variable.grad被赋予新的值），对所有Layer的参数进行逐个更新
            # 组要注意的是，这里面所有参与计算和更新的变量的数据类型都是ndarray
            xp = cuda.get_array_module(param.data)

            self.vs[v_key] = xp.zeros_like(param.data)
            # 动态管理：仅存储实际存在的参数的速度，无需预知参数数量或形状。
            # 内存效率：参数可能动态增减（如某些模型结构变化），字典自动适应。

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v

# Adam check dezero/optimizers.py
# adaptive gradient with momentum
class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, lmbda = 1e-6):
        super().__init__()
        self.t = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lmbda = lmbda
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def adjust(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        '''
        这里把self.lr放到修正项内部计算有几个好处：
        更清晰的接口设计，符合优化器惯例。
        简化动态学习率调整（如调度器仅需修改 self.alpha）。
        集中管理学习率计算逻辑，减少错误风险。
        '''
        return  self.lr*math.sqrt(fix2) / fix1

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data
        grad += self.lmbda * param.data # weight decay

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)
        param.data -= self.adjust * m / (xp.sqrt(v) + eps)
        # print('v7')