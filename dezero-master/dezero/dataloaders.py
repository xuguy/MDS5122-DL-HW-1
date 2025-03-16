import math
import random
import numpy as np
# cuda.py中已经定义了一套方法，可以安全地import cupy
from dezero import cuda

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle = True, gpu = False):

        # dataset mush be a well-defined sub-class of Dataset class
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)

        # 向上取整，保证可以遍历完整个数据集
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.gpu = gpu

        self.reset()

    # 重新打乱顺序（重置iter），意思是本epoch结束
    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))

        else:
            self.index = np.arange(len(self.dataset))

    # 返回iterator对象，也就是class 本身
    def __iter__(self):
        return self
    
    # 定义返回下一个元素的方法
    def __next__(self):
        '''
        dataloader创建batch数据都是在__next__中创建，这里根据实例变量gpu的值在cupy 和 numpy 之间切换，init 相应数据类型的变量
        '''
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration
        
        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i*batch_size : (i+1)*batch_size]

        # 从原始数据self.dataset中取出数据，此时还没有转换成xp
        # 因为Dataset不支持切片操作，因此只能利用iterator逐个单独取出，放到列表里
        # 调用Dataset对象的__getitem__方法将得到一个2元tuple
        # 第一个元时data，第二个元是label
        batch = [self.dataset[i] for i in batch_index]

        # 根据是否开启DataLoader中的gpu来选择xp
        # xp = cuda.cupy if self.gpu else np
        xp = cuda.cp if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])
        self.iteration += 1

        return x, t

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True
    
    def next(self):
        return self.__next__()