import numpy as np
from dezero import cuda
from dezero.core import Function, as_variable
from dezero.utils import pair, get_conv_outsize, get_deconv_outsize

from dezero.functions import linear, broadcast_to


def conv2d_simple(x, W, b = None, stride = 1, pad = 0):
    x, W = as_variable(x), as_variable(W)
    Weight = W # Weight 卷积核
    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col = im2col(x, (KH, KW), stride, pad, to_matrix=True)
    Weight = Weight.reshape(OC, -1).transpose()
    t = linear(col, Weight, b)
    # reshape to (N, OC, OH, OW)
    y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
    return y

# =========== vanilla way to do Conv2d ============
import time
class Conv2dV(Function):
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)

    def forward(self, x, W, b):
        # x: 输入: (N, C, W, H)
        # W: 卷积核的参数 (FN, C, KH, KW)
        xp = cuda.get_array_module(x)
        # kernel height/width
        # KH, KW = W.shape[2:]
        Weight = W
        # OC, C, KH, KW = W.shape
        SH, SW = self.stride
        PH, PW = self.pad
        OC, C, KH, KW = Weight.shape
        N, C, H, W = x.shape
        out_h = get_conv_outsize(H, KH, SH, PH)
        out_w = get_conv_outsize(H,KW,SW,PW)
        # we output the flatten input x here: to_matrix = True, insteand of False
        # x:(N, C_in, KH, KW, OH, OW) -> col: (N*out_h*out_w, C*filter_h*filter_w)
        # col_W: (C*KH*KW, OC)
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=True) # here return a numpy.array
        
        # col = im2col(x, (KH, KW), self.stride, self.pad, to_matrix=True) # here return a Variable
        # print(col)
        # print(col.shape)
        # print(type(col))
        
        # col: (N, C_in, KH, KW, OH, OW)
        # W: (C_out, C_in, KH, KW)
        # 依照书第一册：CNN里面的，把W.reshape(FN,-1),这里的FN就是C_out = OC
        # (C*KH*KW, OC)
        col_W = Weight.reshape(OC, -1).T
        # print(col_W.shape)
        # y = xp.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        # out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # y = col.dot(col_W) 
        y = xp.dot(col, col_W)

        if b is not None:
            y += b
        y = y.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # y = np.transpose(y, (0, 3, 1, 2))
        self.x = x
        self.col = col
        self.col_W = col_W
        return y

    def backward(self, gy):
        xp = cuda.get_array_module(gy)
        x, W, b = self.inputs # they are all Variable
        # print(W)
        FN, C, FH, FW = W.shape
        # 这里如果不用gy.data会有问题：观察其他的实现方式，发现他们的forward/backward都有一个方法会把x/gy转换成numpy.ndarray后再进行计算，这是因为他们继承了Function类，可是我们这里的backward方法中，所有参与计算的变量都没有通过继承了Function类的函数，因此所有变量的计算都用的是Variable，这并不好
        gy = gy.transpose(0, 2, 3, 1).reshape(-1, FN)

        # gb = gy.sum(axis = 0)
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=0)
        
        gW  = xp.dot(self.col.T, gy.data) # costly
        # gw  = self.col.T.dot(gy.data)
        # print(type(self.col))
        # print((self.col.T).dot(gy))
        # print(f'gy.shape: {gy}, self.col.T:{self.col.T}')
        # gw = self.col.T.dot(gy)
        gW = gW.transpose(1, 0).reshape(FN, C, FH, FW)
        # dcol = np.dot(gy, self.col_W.T)
        # col_WT = self.col_W.T
        dcol = xp.dot(gy.data, self.col_W.T) # costly2
        # dcol = gy.data.dot(self.col_W.T)
        # print(dcol)
        # print(dcol2)
        # y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad, to_matrix=False)
        gx = col2im_array(dcol, self.x.shape,(FH, FW), self.stride, self.pad)
        # gx = col2im(dcol, self.x.shape,(FH, FW), self.stride, self.pad)
        # gx = col2im(dcol, x.shape,(FH, FW), self.stride, self.pad)
        # e3-e2 : 0.56, e5-e4: 0.48
        gx, gW, gb = as_variable(gx), as_variable(gW), as_variable(gb)
        return gx, gW, gb


def conv2dv(x, W, b=None, stride=1, pad=0):
    return Conv2dV(stride, pad)(x, W, b)

# =========================================
# better way to do Conv2d （尚未考察，仅移植）

class Conv2d(Function):
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)

    def forward(self, x, W, b):
        # x: 输入: (N, C, W, H)
        # W: 卷积核的参数 (FN, C, KH, KW)
        xp = cuda.get_array_module(x)
        # kernel height/width
        KH, KW = W.shape[2:]

        # 展平, 中间的参数(KH, KW)为卷积核的尺寸，这里to_matrix = False是出于性能考虑，因为不需要真的吧x展开就能做到，况且展开以后还要reshape，reshape还不能用专门优化过的np.tensordot/np.rollaxis处理，性能较差
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False) # it returns col as numpy.ndarray
        # print(type(col)) # numpy.ndarray
        

        '''
        col的形状：(N, C_in, KH, KW, OH, OW)，不是说展平吗？为什么多出这么多维度？im2col的目的是把原本的x按照卷积核的移动特征扩张成一个一个的向量，使得这些向量和扩张后的卷积核做向量内积和不扩张+逐个stride移动做内积等效，因此我们需要把卷积核作用过的每一个区域拿出来（看成一个整体：就好像1d向量的一个元素一样）排成一个向量

        W的形状：(C_out, C_in, KH, KW)，C_out即为经过卷积后输出的特征图的通道数，也即为卷积核的数量，C_in保持与输入图像一致，每个卷积核的通道数必须要和输入图像一致

        收缩轴：col的(1,2,3)轴（对应C_in, KH, KW）与W的(1,2,3)轴相乘。

        结果形状：(N, OH, OW, C_out)，随后通过rollaxis调整为(N, C_out, OH, OW): 相当于把第3轴放到第1轴前
        '''
        # col: (N, C_in, KH, KW, OH, OW)
        # W: (C_out, C_in, KH, KW)
        y = xp.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        # print(type(y))
        if b is not None:
            y += b
        y = xp.rollaxis(y, 3, 1)
        # y = np.transpose(y, (0, 3, 1, 2))
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        
        # xp = cuda.get_array_module(gy)
        # gcol = xp.tensordot(W, gy, (0, 1))
        # print(gcol)
        # ==== gx ====
        #  CNN: dcol = np.dot(dout, self.col_W.T)
        gx = deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad, outsize=(x.shape[2], x.shape[3]))
        # print(type(gy)) # Variable
        # print(type(gx)) # variable
        # x的前2个维度是N和C，第2/3维度才是H与W，这里通过输入数据的shape以及pad\stride等信息用gy倒推gx
        # ==== gW ====
        gW = Conv2DGradW(self)(x, gy)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        # print(gW)
        return gx, gW, gb


def conv2d(x, W, b=None, stride=1, pad=0):
    return Conv2d(stride, pad)(x, W, b)


class Deconv2d(Function):
    def __init__(self, stride=1, pad=0, outsize=None):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.outsize = outsize

    def forward(self, x, W, b):
        xp = cuda.get_array_module(x)

        Weight = W
        SH, SW = self.stride
        PH, PW = self.pad
        C, OC, KH, KW = Weight.shape
        N, C, H, W = x.shape
        if self.outsize is None:
            out_h = get_deconv_outsize(H, KH, SH, PH)
            out_w = get_deconv_outsize(W, KW, SW, PW)
        else:
            out_h, out_w = pair(self.outsize)
        img_shape = (N, OC, out_h, out_w)

        # gcol: 展平后的矩阵的grad
        gcol = xp.tensordot(Weight, x, (0, 1))
        # print(type(x)) # numpy.ndarray
        # 轴调整
        gcol = xp.rollaxis(gcol, 3)
        # 与conv2d互为逆操作：conv2d用im2con_array把原始输入展开做卷积
        # deconv2d用col2im_array把展开后的矩阵还原
        y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad, to_matrix=False)
        # b, k, h, w
        if b is not None:
            self.no_bias = True
            y += b.reshape((1, b.size, 1, 1))
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        # print(type(gy))
        # ==== gx ====
        gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
        # ==== gW ====
        f = Conv2DGradW(self)
        gW = f(gy, x)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
    return Deconv2d(stride, pad, outsize)(x, W, b)


class Conv2DGradW(Function):
    # 接受一个Conv2d实例作为参数
    def __init__(self, conv2d):
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
        xp = cuda.get_array_module(x)

        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        # Linear层的反向传播: gW = np.dot(self.col.T, dout), dout <=> gy
        gW = xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
        return gW

    def backward(self, gys):
        x, gy = self.inputs
        gW, = self.outputs

        xh, xw = x.shape[2:]
        gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad,
                      outsize=(xh, xw))
        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy
    
'''
Conv2d和Deconv2d互为逆运算关系：Conv2d接受原始输入图像x以及与卷积核有关参数W/b(卷积核的参数 W:(FN, C, KH, KW))、与卷积核行为有关的参数pad/stride， 输出经过卷积处理的特征图(FN, C_out, OH, OW)
Deconv2d接受特征图x、与卷积核有关的参数W/b、与卷积核行为有关的参数pad/stride，倒推出原始输入图形的数据y，即把特征图还原为输入图像：是一种逆向的卷积操作，目的是通过参数化的方式恢复输入的空间分辨率
'''


# =========== numpy im2col ============
def im2col_array(img, kernel_size, stride, pad, to_matrix = True):
    '''
    img: input data, typically a np.array
    '''

    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    xp = cuda.get_array_module(img)
    if xp != np:
        col = _im2col_gpu(img, kernel_size, stride, pad)
    else:
        img = np.pad(img, ((0, 0), (0, 0), (PH, PH + SH -1), (PW, PW + SW - 1)), mode = 'constant', constant_values = (0,))

        col = np.ndarray((N, C, KH, KW, OH, OW), dtype = img.dtype)

        for j in range(KH):
            j_lim = j + SH*OH
            for i in range(KW):
                i_lim = i + SW*OW
                col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]
    '''
    输入与输出形状分析
    输入图像：img的原始形状为(N, C, H, W)（批大小、通道数、高度、宽度）。

    填充后的图像：通过np.pad对img进行填充，使其形状变为(N, C, H+2PH+SH-1, W+2PW+SW-1)，确保卷积窗口能覆盖所有可能的输入区域。

    输出col数组：形状为(N, C, KH, KW, OH, OW)，其中KH和KW是卷积核的高和宽，OH和OW是输出特征图的高和宽。
    ===========================================
    用子数组的思路考察ndarray切片与赋值
    - 右侧切片：img[:, :, j:j_lim:SH, i:i_lim:SW],该切片从填充后的输入图像中提取数据：

    - 高度方向：起始索引为j，步长SH，结束索引j_lim = j + SH*OH。由于步长为SH且需覆盖OH个输出位置，实际切片长度为OH。

    - 宽度方向：同理，切片长度为OW。

    结果形状：(N, C, OH, OW)，对应批大小、通道数、输出特征图的高和宽。

    左侧赋值：col[:, :, j, i, :, :]
    目标位置是col数组的(N, C, j, i, OH, OW)部分。由于j和i是标量索引，col的这部分 子数组 形状为(N, C, OH, OW)，与右侧切片形状完全一致
    '''

    if to_matrix:
    # 为什么要这样transpose：因为我们的最终目的是为了输出一个2dim矩阵以供计算，所以要调整原来的5维array，让特定的维度被压缩成1维；这里transpose以后进行了reshape，我们观察reshape里面的参数：N*out_h*out_w正好把col的第0、4、5维压缩成1维，第二个参数-1表示把 余下的维度压缩成1维
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N*OH*OW, -1))

    return col

# =========== im2col / col2im ============
class Im2col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return y
    
    def backward(self, gy):
        gx = col2im(gy, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)

        return gx
    
def im2col(x, kernel_size, stride = 1, pad = 0, to_matrix = True):
    y = Im2col(kernel_size, stride, pad, to_matrix)(x)
    return y

class Col2im(Function):
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        y = col2im_array(x, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)

        return y

    def backward(self, gy):
        gx = im2col(gy, self.kernel_size, self.stride, self.pad, self.to_matrix)

        return gx

def col2im(x, input_shape, kernel_size, stride = 1 ,pad = 0, to_matrix = True):
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)

def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix = True):
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    xp = cuda.get_array_module(col)
    if xp != np:
        img = _col2im_gpu(col, SH, SW, PH, PW, H, W)
        return img
    else:
        img = np.zeros((N, C, H + 2*PH + SH -1, W + 2*PW + SW -1), dtype = col.dtype)

        for j in range(KH):
            j_lim = j + SH*OH
            for i in range(KW):
                i_lim = i + SW*OW
                # codes given are wrong: not += but =
                img[:, :, j:j_lim:SH, i:i_lim:SW] = col[:, :, j, i, :, :]
        return img[:, :, PH : H+PH, PW : W + PW]
    

# code from chainer
def _im2col_gpu(img, kernel_size, stride, pad):
    """im2col function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, h, w = img.shape
    kh, kw = pair(kernel_size)
    sy, sx = pair(stride)
    ph, pw = pair(pad)
    out_h = get_conv_outsize(h, kh, sy, ph)
    out_w = get_conv_outsize(w, kw, sx, pw)
    dy, dx = 1, 1
    col = cuda.cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    cuda.cupy.ElementwiseKernel(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)

    return col


def _col2im_gpu(col, sy, sx, ph, pw, h, w):
    """col2im function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, kh, kw, out_h, out_w = col.shape
    dx, dy = 1, 1
    img = cuda.cupy.empty((n, c, h, w), dtype=col.dtype)

    cuda.cupy.ElementwiseKernel(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img

# ========= max pooling ==========
class Pooling(Function):
    def __init__(self, kernel_size, stride = 1, pad = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):

        # 先展平图像(N, C, H, W)，输出 （N, C, KH, KW, OH, OW）,注意，to_matrix = False
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix = False)

        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH*KW, OH, OW)

        # 通过 argmax(axis=2) 获取每个池化窗口内最大值的索引，并保存在 self.indexes 中供反向传播使用
        self.indexes = col.argmax(axis = 2)
        # 通过 max(axis=2) 提取每个窗口的最大值，得到池化后的输出一维向量 y
        y = col.max(axis = 2)
        return y
    
    def backward(self, gy):
        return Pooling2DGrad(self)(gy)
    
class Pooling2DGrad(Function):
    def __init__(self, mpool2d):
        # 传入的mpool2d 是Pooling实例，用来获取上面的Pooling的前向传播过程中产生的数据
        # 初始化参数：从前向传播的 Pooling 实例中获取池化参数（kernel_size、stride、pad）和输入形状
        self.mpool2d = mpool2d
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        # Pooling2DGrad的正向传播就是Pooling的反向传播
        # indexes的保存和恢复是关键，因为在max pooling中，只有最大值的位置才有梯度传播回去，所以反向传播时需要知道前向时哪个位置被选中
        self.indexes = mpool2d.indexes

    def forward(self, gy):
        xp = cuda.get_array_module(gy)

        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)

        # 创建全零数组：并用前向传播中的最大值index回填，因为只有在前向传播中选中的最大值的位置才有梯度
        # 创建一个全零的展开列 gcol，形状为 (N*C*OH*OW*KH*KW)
        gcol = xp.zeros((N*C*OH*OW*KH*KW), dtype = self.dtype)
        
        indexes = (self.indexes.ravel() + xp.arange(0, self.indexes.size*KH*KW, KH*KW))
        # 通过前向传播保存的 indexes，将上游梯度 gy 的值精确填充到 gcol 中对应最大值的位置
        gcol[indexes] = gy.ravel()

        # 将(N*C*OH*OW*KH*KW) reshape 为 (N, C, OH, OW, KH, KW)
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = xp.swapaxes(gcol, 2, 4)
        gcol = xp.swapaxes(gcol, 3, 5) # 经过2此swap axes， gcol变为 (N, C, KH, KW, OH, OW)
        # 通过 col2im_array 将填充后的梯度列 gcol 转换回原始输入形状，得到输入梯度 gx
        # col2im_array 接受参数：col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix = True)
        gx = col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride, self.pad, to_matrix = False)

        return gx
    
    def backward(self, ggx):
        # 这个backwards方法不是很明，pass掉也不会有啥问题
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(ggx)
        # pass

# 计算更高阶的grad？可能在需要二次反向传播的时候才会用到，比如在计算Hessian矩阵或者某些需要二阶导数的优化场景中
class Pooling2DWithIndexes(Function):
    def __init__(self, mpool2d):

        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shpae = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape

        col = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)


        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(N, C, OH, OW)

def pooling(x, kernel_size, stride=1, pad=0):
    return Pooling(kernel_size, stride, pad)(x)


# ======= direct migrate unverified ========
class AveragePooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        y = col.mean(axis=(2, 3))
        return y

    def backward(self, gy):
        # TODO(Koki): This is simple implementation
        N, C, OH, OW = gy.shape
        KW, KH = pair(self.kernel_size)
        gy /= (KW*KH)
        gcol = broadcast_to(gy.reshape(-1), (KH, KW, N*C*OH*OW))
        gcol = gcol.reshape(KH, KW, N, C, OH, OW).transpose(2, 3, 0, 1, 4, 5)
        gx = col2im(gcol, self.input_shape, self.kernel_size, self.stride,
                    self.pad, to_matrix=False)
        return gx


def average_pooling(x, kernel_size, stride=1, pad=0):
    return AveragePooling(kernel_size, stride, pad)(x)
