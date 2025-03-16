import numpy as np
gpu_enable = True
try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False

from dezero import Variable

def get_array_module(x):

    # 如果传入一个Variable对象，那么把Variable对象用于计算的data拿出来，把数据类型改掉
    if isinstance(x, Variable):
        x = x.data
    
    # if gpu_enable = False, then not gpu_enable is True, we get the module of data: numpy
    if not gpu_enable:
        return np
    
    xp = cp.get_array_module(x)
    return xp

def as_numpy(x):

    if isinstance(x, Variable):
        x = x.data
    
    if np.isscalar(x):
        return np.array(x)
    
    elif isinstance(x, np.ndarray):
        return x
    
    return cp.asnumpy(x)

def as_cupy(x):

    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception('Cupy load fail.')
    # 把numpy数据移动到Gpu的竟然是cp.asarray()，佛了
    return cp.asarray(x)