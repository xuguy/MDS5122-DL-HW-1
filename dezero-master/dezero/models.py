from dezero import Layer, utils
import dezero.functions as F
import dezero.layers as L
import numpy as np


# base class for Model
class Model(Layer):

    # 给models加入plot方法，直接画出计算图
    def plot(self, *inputs, to_file = 'model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose = True, to_file=to_file)
    

# MLP model: MLP又是全连接层(fc)神经网络的别名

class MLP(Model):
    def __init__(self, fc_output_sizes, activation = F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            # 把层设置为模型的实例变量来对层的参数进行管理
            # 给自己添加以 l1/l2/l3...命名的L.Linear的实例layer作为自己的属性（属性就是类似self.var这样的类变量，类变量会被保存到MLP.__dicit__中作为MLP的参数
            setattr(self, 'l'+str(i), layer)
            self.layers.append(layer)
    
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        #最后一层的输出就是前向传播的结果
        return self.layers[-1](x)
    
# C5L3
class C5L4(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = L.Conv2d(128,kernel_size=3, stride=1, pad=1)
        self.bn1 = L.BatchNorm2d()

        self.conv2 = L.Conv2d(256,kernel_size=3, stride=1, pad=1)
        self.bn2 = L.BatchNorm2d()

        self.conv3 = L.Conv2d(512,kernel_size=3, stride=1, pad=1)
        self.bn3 = L.BatchNorm2d()

        self.conv4 = L.Conv2d(512,kernel_size=3, stride=1, pad=1)
        self.bn4 = L.BatchNorm2d()

        self.conv5 = L.Conv2d(256,kernel_size=3, stride=1, pad=1)
        self.bn5 = L.BatchNorm2d()



        self.fc1 = L.Linear(512)

        self.fc2 = L.Linear(256)
        self.fc3 = L.Linear(128)
        self.fc4 = L.Linear(10)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.pooling(x, 2, 2)

        # flatten, x.shape[0] is N(batchsize)
        x = F.reshape(x, (x.shape[0], -1))
        # print(x.shape)
        x = F.dropout(F.relu(self.fc1(x)), dropout_ratio=0.3)
        x = F.dropout(F.relu(self.fc2(x)), dropout_ratio=0.3)
        x = F.dropout(F.relu(self.fc3(x)), dropout_ratio=0.3)
        x = self.fc4(x)
        return x



# ========== vgg code from books, unverified ===========
class VGG16(Model):

    def __init__(self, pretrained=False):
        super().__init__()
        self.conv1_1 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv1_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv2_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv2_2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv3_1 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_2 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_3 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv4_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(1000)


    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.pooling(x, 2, 2)
        x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x

    

# ===== resnet, unverified =====
class ResNet(Model):

    def __init__(self, n_layers=152, pretrained=False):
        super().__init__()

        if n_layers == 18:
            block = [2, 2, 2, 2]    
        elif n_layers == 34:
            block = [3, 4, 6, 3] 

        elif n_layers == 50:
            block = [3, 4, 6, 3]
        elif n_layers == 101:
            block = [3, 4, 23, 3]
        elif n_layers == 152:
            block = [3, 8, 36, 3]
        else:
            raise ValueError('The n_layers argument should be either 50, 101,'
                             ' or 152, but {} was given.'.format(n_layers))
        # original 1st layer:
        # 注意，这里第一层，也就是先行层并没有指定输入数据的in_channel，这是因为我们实现的L.Conv2d可以自适应调整in_channels
        # 如果我们使用torch.nn.conv2d，那么我们不得不指定in_channels
        self.conv1 = L.Conv2d(64, 7, 2, 3)
        # adapt to 1 channel image:
        # self.conv1 = L.Conv2d(64, 7, 2, 3, in_channels=1)

        self.bn1 = L.BatchNorm2d()
        self.res2 = BuildingBlock(block[0], 64, 64, 256, 1)
        self.res3 = BuildingBlock(block[1], 256, 128, 512, 2)
        self.res4 = BuildingBlock(block[2], 512, 256, 1024, 2)
        self.res5 = BuildingBlock(block[3], 1024, 512, 2048, 2)
        self.fc6 = L.Linear(1000)
        # self.fc6 = L.Linear(10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.pooling(x, kernel_size=3, stride=2)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = _global_average_pooling_2d(x)
        x = self.fc6(x)
        return x



class ResNet34(ResNet):
    def __init__(self, pretrained=False):
        super().__init__(34, pretrained)

class ResNet18(ResNet):
    def __init__(self, pretrained=False):
        super().__init__(18, pretrained)


def _global_average_pooling_2d(x):
    N, C, H, W = x.shape
    h = F.average_pooling(x, (H, W), stride=1)
    h = F.reshape(h, (N, C))
    return h


class ResNet152(ResNet):
    def __init__(self, pretrained=False):
        super().__init__(152, pretrained)


class ResNet101(ResNet):
    def __init__(self, pretrained=False):
        super().__init__(101, pretrained)


class ResNet50(ResNet):
    def __init__(self, pretrained=False):
        super().__init__(50, pretrained)



class BuildingBlock(Layer):
    def __init__(self, n_layers=None, in_channels=None, mid_channels=None,
                 out_channels=None, stride=None, downsample_fb=None):
        super().__init__()

        self.a = BottleneckA(in_channels, mid_channels, out_channels, stride,
                             downsample_fb)
        self._forward = ['a']
        for i in range(n_layers - 1):
            name = 'b{}'.format(i+1)
            bottleneck = BottleneckB(out_channels, mid_channels)
            setattr(self, name, bottleneck)
            self._forward.append(name)

    def forward(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
        return x
    

class BottleneckA(Layer):
    """A bottleneck layer that reduces the resolution of the feature map.
    Args:
        in_channels (int): Number of channels of input arrays.
        mid_channels (int): Number of channels of intermediate arrays.
        out_channels (int): Number of channels of output arrays.
        stride (int or tuple of ints): Stride of filter application.
        downsample_fb (bool): If this argument is specified as ``False``,
            it performs downsampling by placing stride 2
            on the 1x1 convolutional layers (the original MSRA ResNet).
            If this argument is specified as ``True``, it performs downsampling
            by placing stride 2 on the 3x3 convolutional layers
            (Facebook ResNet).
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 stride=2, downsample_fb=False):
        super().__init__()
        # In the original MSRA ResNet, stride=2 is on 1x1 convolution.
        # In Facebook ResNet, stride=2 is on 3x3 convolution.
        stride_1x1, stride_3x3 = (1, stride) if downsample_fb else (stride, 1)
        #.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0
        self.conv1 = L.Conv2d(mid_channels, 1, stride_1x1, 0,
                              nobias=True)
        self.bn1 = L.BatchNorm2d()

        self.conv2 = L.Conv2d(mid_channels, 3, stride_3x3, 1,
                              nobias=True)
        self.bn2 = L.BatchNorm2d()

        self.conv3 = L.Conv2d(out_channels, 1, 1, 0, nobias=True)
        self.bn3 = L.BatchNorm2d()

        self.conv4 = L.Conv2d(out_channels, 1, stride, 0,
                              nobias=True)
        self.bn4 = L.BatchNorm2d()

    def forward(self, x):
        h1 = F.relu(self.bn1(self.conv1(x))) # 1x1
        h1 = F.relu(self.bn2(self.conv2(h1))) # 3x3
        h1 = self.bn3(self.conv3(h1)) # 1x1
        h2 = self.bn4(self.conv4(x))
        return F.relu(h1 + h2)
    


class BottleneckB(Layer):
    """A bottleneck layer that maintains the resolution of the feature map.
    Args:
        in_channels (int): Number of channels of input and output arrays.
        mid_channels (int): Number of channels of intermediate arrays.
    """

    def __init__(self, in_channels, mid_channels):
        super().__init__()
        
        self.conv1 = L.Conv2d(mid_channels, 1, 1, 0, nobias=True)
        self.bn1 = L.BatchNorm2d()

        self.conv2 = L.Conv2d(mid_channels, 3, 1, 1, nobias=True)
        self.bn2 = L.BatchNorm2d()
        
        self.conv3 = L.Conv2d(in_channels, 1, 1, 0, nobias=True)
        self.bn3 = L.BatchNorm2d()

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return F.relu(h + x)



