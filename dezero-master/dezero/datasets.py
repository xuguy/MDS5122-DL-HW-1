import numpy as np

class Dataset:
    def __init__(self, train=True, transform = None, target_transform = None):
        self.train = train
        self.transform = transform

        # target_transform 对标签进行变换
        self.target_transofrm = target_transform

        if self.transform is None:
            # 原封不动地返回数据，下同
            self.transform = lambda x: x
        if self.target_transofrm is None:
            self.target_transofrm = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    # 不支持切片操作，只支持int类型索引
    def __getitem__(self, index):

        assert np.isscalar(index)
        # label is None == 无监督学习
        if self.label is None:
            return self.transform(self.data[index]), None
        
        else:
            # transform 对单个数据/标签进行变换处理
            return self.transform(self.data[index]), self.target_transofrm(self.label[index])
        
    def __len__(self):
        return len(self.data)
    
    def prepare(self):
        pass

'''
transform usage:
def f(x):
    y = x / 2.0
    return t

train_set = dezero.datasets.Spiral(transform = f)

'''

class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)

# ==== get_spiral datasets generator =====
def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int32)

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix] = j
    # Shuffle
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t

# ============ MNIST ================
import gzip
import matplotlib.pyplot as plt
class MNIST(Dataset):

    # def __init__(self, train=True,
    #              transform=Compose([Flatten(), ToFloat(),
    #                                  Normalize(0., 255.)]),
    #              target_transform=None):
    #     super().__init__(train, transform, target_transform)
    def __init__(self, train=True,
                 transform=None,
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        # this url has been changed from Yan Lecun from local
        # url = './dezero/MNISTdataset/'
        url = './dezero/MNISTdataset/'
        # url = 'http://yann.lecun.com/exdb/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}
        
        files = train_files if self.train else test_files
        data_path = url + files['target']
        label_path = url + files['label']
        print(data_path)
        print(label_path)
        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[
                    np.random.randint(0, len(self.data) - 1)].reshape(H, W)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels():
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
    

# ========== sequential data ==============
class SinCurve(Dataset):

    def prepare(self):
        num_data = 1000
        dtype = np.float64

        x = np.linspace(0, 2 * np.pi, num_data)
        noise_range = (-0.05, 0.05)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)
        if self.train:
            y = np.sin(x) + noise
        else:
            y = np.cos(x)
        y = y.astype(dtype)
        self.data = y[:-1][:, np.newaxis]
        self.label = y[1:][:, np.newaxis]


