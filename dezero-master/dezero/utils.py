import os
import subprocess
import numpy as np
from dezero import as_variable
from dezero import Variable, cuda


def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name+=': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)

def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt +=dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt +=dot_edge.format(id(f), id(y()))

    return txt

def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
    
    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g {\n' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file = 'graph.png'):
    
    dot_graph = get_dot_graph(output, verbose)

    # 获取dot数据并保存至文件
    # os.path.expanduser('~\\123') 返回'C:\\Users\\xuguy\\123'
    # 其作用就是把 ~ 替换成user路径
    # tmp_dir可以设置在任何地方，书作者用的mac，直接apt安装的graphviz，因此把tmp_dir设定在user目录，这是不必要的
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    # Graphviz并没有安装在c盘，因此需要改造一下路径
    cd = r'cd /d D:\graphviz\bin'
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    full_cmd = cd+r'&'+cmd
    print(f'cmd: {full_cmd}')
    subprocess.run(full_cmd, shell=True)

# utility functions for numpy
def sum_to(x, shape):
    """
    sum_to(x，shape)函数用于求x的元素之和并将结果的形状转变为shape
    的形状。不过NumPy中没有这样的函数。因此，DeZero在dezero/utils.py（也就是本.py文件）中提供了一个NumPy版本的sum_to函数。使用该函数可以进行以下计算。

    值得注意的是，sum_to()函数需要实现的功能和np.sum()函数一样，但是接收的参数不同，为了本框架的一致性，我们需要对np.sum()函数进行以下的改造

    Args:
        x (ndarray): Input array.
        shape: the shape we want to achieve by summing

    Returns:
        ndarray: Output array of the shape.
    """

    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    # 这里用的是ndarray的sum方法
    y = x.sum(lead_axis + axis, keepdims=True)

    # 因为反向传播需要对gy的形状稍加调整(因为使用axis和keepdims 求和时会出现改变梯度形状的情况)，所以才需要下面这些调整shape的操作，而不是仅仅只是用个np.sum()就完事
    if lead > 0:
        y = y.squeeze(lead_axis)
    
    return y
'''
# testing code and thoughts:
import numpy as np
x = np.random.rand(2,2,3) # batch_size = 2, two 2x3matrix
shape = (2,1) #target shape: we want to sum it along the 0-axis, to get one 2x3 matrix
# so by np.sum(), we just need to assign axis = (0,)，meaning sum along the 0-axis (equivalent: axis=(0,)/(0)/0)
# how can we convert our target shape to np.sum(axis)'s axis parameter:
# x.shape (2,2,3), target shape (2,3) ->np.sum() axis (0,), check the following algo:

ndim = len(shape)
print(ndim)
# original ndim - after-sum ndim: 3-dim minus 2-dim
lead = x.ndim - ndim
print(lead) # out: 1

# if lead = 1, then lead_axis = (0,) <- the 0-th axis
# if lead = 0, then lead_axis = () <- None of axes being cancel
lead_axis = tuple(range(lead)) # out: lead_axis = (0,) <- 0th-dim

# iter through each axis of target shape, sx==1 means sx's index i is the axis being sum and its dimension reduced to 1, for example: if we want to go from (2,2,3) to (2,1,3), the 1-th axis (the 1-dim) is 1, hence we sum in the 1-th dimension
# 我们惊喜的发现，axis的shape和lead axis的shape是一样的，有多少个dim被求和（即lead的数量），意味着有多少个sx==1
# 如果说，我们要把一个(2,2,3,5)sum成一个(2,1,5)，我们知道，(3,)是一个长度为3的行向量，而(1,3)是长度为3的列向量，从(2,2,3,5)变成(2,1,5)只可能是第0个维度被求和掉，因为除了第0各维度被求和，其他维度被求和只会被减到1，而不会消失，因此，只要lead>0，必然说明有维度消失，也就是原来的第0维消失，如果lead=2，那么原来的第0维先消失，第1维度变成了第0维，接着消失，这样才可能使lead>0。所以用range(lead)就可以生成消失的维度的编号（这个编号只能从0开始，按顺序增加），这就是lead_axis名字的来源：前导轴
# 从前导轴开始数，数到维度等于1的那个轴，也即sx==1的那个轴，就是除了前导轴以外被求和的轴，这就是什么要i+lead，就是为了找除了前导轴以外的其他轴
# 为什么前导轴特殊呢？因为为了反向传播我们可以把输出broadcast成和输入一样，我们必须妥善管理shape，前面说了，求和这个操作如果发生在前导轴上，前导轴会变成1，比如np.array([[1,2,3],[4,5,6]]) （shape=(2,3))对第0轴求和，那么会得到np.array([[5,7,9]])，shape=(1,3)，这不利于我们broadcast，因此我们需要把所有被sum的前导轴squeeze掉;并且，如果不用keepsdim=True，那么所有被sum的轴都会消失，我们的目的是去除先导轴，但是保留其他非先导轴但是被求和的轴，这正是为了broadcast方便，所以原装的np.sum不能满足我们的要求（原装的要么都保留为1，要么都删除），只能改造。
axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1]) # axis = ()

lead_axis + axis # = (0,)
y=x.sum(lead_axis + axis, keepdims = True)

# ndarray.sum(axis=None), here axis = (0,), None or int or tuple of ints, meaning sum along the first dim
x.sum((0,), keepdims=True) # shape: (1,2,3)

np.array([[1,2,3],[4,5,6]]).sum(axis = 1)
'''

def reshape_sum_backward(gy, x_shape, axis, keepdims):

    ndim = len(x_shape)

    # 这里的axis就是传入sum函数的期望target axis：你希望在哪些axis上sum，在这些axis上sum，就会有把这些axis弄没的风险，下面我们想办法把弄没了的axis还原
    tupled_axis = axis
    # 如果sum(axis=None),表示所有元素相加，输出scalar
    if axis is None:
        tupled_axis = None
    #如果sum(axis=0)，即等于一个整数，把axis改造成tuple
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    # 3个条件，比较复杂的是当x不是标量的时候（第一个条件不成立），倘若x的tupled_axis 非None（第二个条件不成立，也即是说我们至少指定了一个我们要sum的axis），在这个情况下，如果keepdims=Flase（第三个条件不成立）,那么括号里面全部不成立，需要进入判断执行语句。keepdims=False意味着所有被sun的axis都会没掉，因此需要把sum的axis，也即tupled_axis对应的sum axis全部恢复，如果keepdims=True，那么不进入判断里面的语句，但是keepdims会保留sum的axis至少为1，这样我们就不需要恢复gy的shape；如果第二个条件成立，也就是说我们没有指定任何要sum的axis，那么最后也会保留所有axis，尽管这样sum出来的应该是标量
    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a>=0 else a+ndim for a in tupled_axis]

        # 获取前向传播的output的grad的shape，gy就是gys
        # gys = [output().grad for output in f.outputs]
        # gy的shape就是我们需要调整的shape
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1) # 把sum掉的轴补回去
        
    else:
        shape = gy.shape
    
    gy = gy.reshape(shape)
    return gy

# ====================================================
# test code for examining utils.reshape_sum_backward

'''
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
from dezero import as_variable
import dezero.functions as F

x_shape = x.shape
ndim = len(x_shape) # ndim=2
axis = (0,)
tupled_axis = (0,)
'''

# about utils.reshape_sum_backward:

# we need to derive the "actual axis" of y
# this judgement: if not (ndim == 0 or tupled_axis is None or keepdims):
# tells us 1) if x_shape is None: x is a scalar; 2) tupled_axis is None -> axis is None: if we sum everything out，if we sum everything out, there will be no need to recover those disappeared axis; 3)keepdims = False (default): there might be axis disapper, like leading axis or non-leading

# if any 1 of those 3 condition does not met: if x is not a scalar, or sum(axis != None), or keepdims=False, we need to reshape gy to adapt to the backward prop

# how? after sum, we go from (2,3) -> (3,), and we know the axis we sum along is 0 <=> (0,)，so the actual_axis (where we sum along) is 0; if axis has negative value like (1,-1), it means counting from the end to start, simply use a+ndim to recover

# also remember how gy is initialized and input into Variable.backward: core.Variable.backward: gys = [output().grad for output in f.outputs], where f is self.creator, which is set in the process of core.Function.__call__ forward pass, and f.outputs here is ys = self.forward(*xs) -> result after calling y = x.sum(axis=self.axis, keepdins=self.keepdims) is the forward of functions.Sum

# here use x = Variable(np.array([[1,2,3], [4,5,6]])) a (2,3) Variable as example, if we y = F.sum(x, axis=0,keepdims=False), we end up with variable([5 7 9]): shape (3,),
# the 3 conditions does not hold: ndim=2 !=0, tupled_axis = (0,) which is not None, and keepdims = False, so let's proceed to see hot the shape should be reshape:

# actual_axis = [0] since we do not use negative axis
# shape = list(gy.shape), gy(gys of core.Function.backward) is f.outputs <- output of forward pass, which is F.sum(x, axis=0,keepdims=False), so the shape is shape = list((3,)) = [3,0]

# for a in sorted(actual_axis):
#   shape.insert(a,1) -> .insert(index, x): 在指定位置index插入元素x, 也就是在index=a这个地方插入1(把消失的axis都补回去），在list((3,)) = [3]的第a=0这个地方插入1，得到[1, 3]
'''
shape = [1, 3]
gy = F.sum(x, axis=0,keepdims=False)

# gy: variable([5 7 9])
shape = (3,)

gy.reshape(shape)

# gy.reshape([1,3])-> shape (1,3)，变成了一个行向量！

'''


def max_backward_shape(x, axis):
    if axis is None:
        axis = range(x.ndim)

    elif isinstance(axis, int):
        axis = (axis,)

    else:
        axis = axis

    shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
    return shape

# 计算 log(exp(x1) + exp(x2) +...+ exp(xn))
# = x* + log(exp(x1-x*) + exp(x2-x*) +...+ exp(xn-x*))
def logsumexp(x, axis = 1):
    '''
    check:
    https://en.wikipedia.org/wiki/LogSumExp
    '''
    # 防止溢出
    xp = cuda.get_array_module(x)
    m = x.max(axis = axis, keepdims = True)
    y = x - m
    xp.exp(y, out = y)
    s = y.sum(axis = axis, keepdims = True)
    xp.log(s, out = s)
    m += s
    return m


#  ========== CNN ===========
# 获取卷积输出shape
def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad*2 - kernel_size) // stride + 1

# 将shape相关的输入规范化之函数
def pair(x):
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError


'''
get_deconv_outsize
This function takes the size of input feature map, kernel, stride, and pooling of one particular dimension, then calculates the output feature map size of that dimension.
check:
https://docs.chainer.org/en/latest/reference/generated/chainer.utils.get_deconv_outsize.html
'''
def get_deconv_outsize(size, k, s, p):
    return s*(size - 1) + k - 2*p



# ========= download function from book, unverified

import os
import subprocess
import urllib.request
def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0: p = 100.0
    if i >= 30: i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')


cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')

def get_file(url, file_name=None):
    """Download a file from the `url` if it is not in the cache.

    The file at the `url` is downloaded to the `~/.dezero`.

    Args:
        url (str): URL of the file.
        file_name (str): Name of the file. It `None` is specified the original
            file name is used.

    Returns:
        str: Absolute path to the saved file.
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path




