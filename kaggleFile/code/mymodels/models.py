import torch.nn as nn
import torch


C5L3_MNIST = nn.Sequential(
    nn.Conv2d(1, 128, 3, padding=1),
    nn.BatchNorm2d(128),  # Batch Normalization after the first convolution
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Conv2d(128, 256, 3, padding=1),
    nn.BatchNorm2d(256),  # Batch Normalization after the second convolution
    nn.ReLU(inplace=True),
    # nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Conv2d(256, 512, 3, padding=1),
    nn.BatchNorm2d(512),  # Batch Normalization after the third convolution
    nn.ReLU(inplace=True),
    
    nn.Conv2d(512, 512, 3, padding=1),
    nn.BatchNorm2d(512),  # Batch Normalization after the fourth convolution
    nn.ReLU(inplace=True),
    
    nn.Conv2d(512, 256, 3, padding=1),
    nn.BatchNorm2d(256),  # Batch Normalization after the fifth convolution
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Flatten(),
    nn.Linear(256 * 7 * 7, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(256, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(128, 10),
)

C5L3_cifar10 = nn.Sequential(
    nn.Conv2d(3, 128, 3, padding=1),
    nn.BatchNorm2d(128),  # Batch Normalization after the first convolution
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Conv2d(128, 256, 3, padding=1),
    nn.BatchNorm2d(256),  # Batch Normalization after the second convolution
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Conv2d(256, 512, 3, padding=1),
    nn.BatchNorm2d(512),  # Batch Normalization after the third convolution
    nn.ReLU(inplace=True),
    
    nn.Conv2d(512, 512, 3, padding=1),
    nn.BatchNorm2d(512),  # Batch Normalization after the fourth convolution
    nn.ReLU(inplace=True),
    
    nn.Conv2d(512, 256, 3, padding=1),
    nn.BatchNorm2d(256),  # Batch Normalization after the fifth convolution
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Flatten(),
    nn.Linear(256 * 4 * 4, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(256, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(128, 10),
)

C3L2_MNIST = nn.Sequential(
    nn.Conv2d(1, 32, 3,padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace = True),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, 3,padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace = True),
    # nn.MaxPool2d(2),

    nn.Conv2d(64,128,3,padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace = True),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(128*7*7, 128),
    nn.ReLU(inplace = True),
    nn.Dropout(p=0.2),

    nn.Linear(128, 10)
    
)

C3L2_cifar10 = nn.Sequential(
    nn.Conv2d(3, 32, 3,padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace = True),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, 3,padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace = True),
    nn.MaxPool2d(2),

    nn.Conv2d(64,128,3,padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace = True),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(128*4*4, 128),
    nn.ReLU(inplace = True),
    nn.Dropout(p=0.2),

    nn.Linear(128, 10)
    
)


# ============= resnet ==============
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ============ 在这里用 nn.LazyConv2d 改造 =============
'''
那么，解决方案是让用户在使用模型前，先进行一次前向传播，用虚拟输入初始化参数。或者，用户可以将LazyConv2d替换为普通的Conv2d，并正确指定输入通道数，这样就不需要延迟初始化。

例如，用户原来的注释中，初始卷积层被注释掉了，用的是LazyConv2d。而原来的代码可能使用的是普通的Conv2d，输入通道为img_channels（比如3）。用户可能想尝试使用LazyConv2d来避免手动指定输入通道数。但此时，当模型初始化时，LazyConv2d的输入通道数尚未确定，直到第一次前向传播才会确定。而后续的BatchNorm层中的参数已经设置为64，这没问题，因为LazyConv2d的输出通道是64，所以BatchNorm的通道数匹配。

但是，在构造其他层的时候，比如在_make_layer中，第一个ResidualBlock可能引用了self.in_channels，此时self.in_channels被初始化为64。而如果LazyConv2d的输出通道是64，那这里应该没问题。因为假设LazyConv2d的输出是64，所以BN层是64没问题，之后layer1的输入通道也是64。那在第一次前向传播时，输入经过conv1之后得到64通道的特征图，后续层都能正确构建。

问题可能出在，用户可能在创建模型之后，立即调用了某些需要参数初始化的函数，比如打印模型参数、统计参数量，或者优化器在初始化时访问了这些参数。例如，在定义模型后，用户可能执行了类似print(model.parameters())或者将其传递给优化器，此时LazyConv2d的参数尚未初始化，导致错误
'''
class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10, img_channels = 3):
        super(ResNet, self).__init__()
        self.img_channels = img_channels
        self.in_channels = 64
        
        # 初始卷积层 (CIFAR-10 适配)
        # self.conv1 = nn.Conv2d(self.img_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.LazyConv2d(64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 构建残差层
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        
        # 分类器
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 创建 ResNet-20 实例 (适用于 CIFAR-10)
ResNet20_MNIST =  ResNet(num_blocks=[3, 3, 3], img_channels=1)
ResNet20_cifar10 =  ResNet(num_blocks=[3, 3, 3], img_channels=3)
ResNet20_omni =  ResNet(num_blocks=[3, 3, 3], img_channels=3)



# ============ resnet 18 ==============
class BasicBlock(nn.Module):
    expansion = 1  # 扩展系数（用于调整通道数）

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 下采样函数（用于调整残差连接维度）

    def forward(self, x):
        identity = x

        # 这里尝试一下用 in_channels = x.shape[1] ，自适应调整根据输入x的通道数调整模型的in_channels
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:  # 调整残差连接维度
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = self.relu(out)

        return out

# 定义完整的 ResNet-18
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64  # 初始通道数
        
        # 初始卷积层（适配小尺寸输入）
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 构建四个残差阶段
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

ResNet18_MNIST = ResNet(BasicBlock, [2, 2, 2, 2], 10, 1)
ResNet18_cifar10 = ResNet(BasicBlock, [2, 2, 2, 2], 10, 3)


# ============== resnet 50 ==================
class Bottleneck(nn.Module):
    expansion = 4  # 最终通道扩展系数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # 1x1 卷积降维
        # 注意，第一层的stride采用默认值，也就是stride = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 卷积升维
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 定义 ResNet 主结构
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_channels=3):
        super().__init__()
        self.in_channels = 64
        
        # 初始卷积层（适配小尺寸输入）
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 残差阶段
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, base_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != base_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, base_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(base_channels * block.expansion),
            )

        layers = []
        # first block needs to be processed specifically
        '''
        第一个 Block 的特殊性

        负责处理维度变化（通道数或空间尺寸），通过 downsample 调整残差分支。

        示例：layer2 的第一个 Bottleneck 中：

        主分支：conv2 的 stride=2，空间尺寸减半。

        残差分支：downsample 的 stride=2，同时调整通道数。
        '''
        layers.append(block(self.in_channels, base_channels, stride, downsample))
        self.in_channels = base_channels * block.expansion
        for _ in range(1, blocks):
            # 后续的block中没有stride参数
            layers.append(block(self.in_channels, base_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


ResNet50_MNIST = ResNet(Bottleneck, [3, 4, 6, 3], 10, 1)
ResNet50_cifar10 = ResNet(Bottleneck, [3, 4, 6, 3], 10, 3)


# ============ vgg 16 ================
class VGG16(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(VGG16, self).__init__()
        
        # 特征提取层配置（卷积+池化）
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2x2
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1x1
        )
        
        # 分类器（适配小尺寸输入）
        self.classifier = nn.Sequential(
            nn.Linear(512*1*1, 4096),  # 输入尺寸根据最后特征图调整
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        
        # 参数初始化
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平特征图
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


VGG16_MNIST = VGG16(in_channels=1)
VGG16_cifar10 = VGG16(in_channels=3)



# ============ baseline model ===============
C5L4_base_cifar10 = nn.Sequential(
    nn.Conv2d(3, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout(0.3),
    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout(0.3),
    nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
    nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout(0.3),
    nn.Flatten(),
    nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
    nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
    nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
    nn.Linear(128, 10),
)


# baseline model: BEST: -dropout(conv)+batchnorm
C5L4_BEST_cifar10 = nn.Sequential(
    nn.Conv2d(3, 128, 3, padding=1),
    nn.BatchNorm2d(128),  # Batch Normalization after the first convolution
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Conv2d(128, 256, 3, padding=1),
    nn.BatchNorm2d(256),  # Batch Normalization after the second convolution
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Conv2d(256, 512, 3, padding=1),
    nn.BatchNorm2d(512),  # Batch Normalization after the third convolution
    nn.ReLU(inplace=True),
    
    nn.Conv2d(512, 512, 3, padding=1),
    nn.BatchNorm2d(512),  # Batch Normalization after the fourth convolution
    nn.ReLU(inplace=True),
    
    nn.Conv2d(512, 256, 3, padding=1),
    nn.BatchNorm2d(256),  # Batch Normalization after the fifth convolution
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Flatten(),
    nn.Linear(256 * 4 * 4, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(256, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(128, 10),
)

# best, removing all dropout layer
C5L4_BEST_cifar10_nodrp = nn.Sequential(
    nn.Conv2d(3, 128, 3, padding=1),
    nn.BatchNorm2d(128),  # Batch Normalization after the first convolution
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Conv2d(128, 256, 3, padding=1),
    nn.BatchNorm2d(256),  # Batch Normalization after the second convolution
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Conv2d(256, 512, 3, padding=1),
    nn.BatchNorm2d(512),  # Batch Normalization after the third convolution
    nn.ReLU(inplace=True),
    
    nn.Conv2d(512, 512, 3, padding=1),
    nn.BatchNorm2d(512),  # Batch Normalization after the fourth convolution
    nn.ReLU(inplace=True),
    
    nn.Conv2d(512, 256, 3, padding=1),
    nn.BatchNorm2d(256),  # Batch Normalization after the fifth convolution
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Flatten(),
    nn.Linear(256 * 4 * 4, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(256, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(128, 10),
)