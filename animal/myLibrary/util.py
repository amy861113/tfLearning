import numpy as np
import cv2
import math
from tensorflow.keras.utils import Sequence

class ImageDataset:
    def __init__(self, X, Y, transformer = lambda x:x):
        self.X = X
        self.Y = Y
        self.transformer = transformer
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, idx):
        if isinstance(idx, int) or isinstance(idx, slice):
            imgs = self.X[idx]
            labels = self.Y[idx]
        else:
            imgs = [self.X[i] for i in idx]
            labels = [self.Y[i] for i in idx]

        if isinstance(imgs, list):
            imgs = [cv2.imread(img) for img in imgs]
            imgs = [self.transformer(img) for img in imgs]
            imgs = [np.expand_dims(img, axis=0) for img in imgs]
            imgs = np.concatenate(imgs, axis=0)
        else:
            imgs = cv2.imread(imgs)
            imgs = self.transformer(imgs)
        return np.array(imgs), np.array(labels)

class MyGenerator:
    def __init__(self, datasets, batch_size, shuffle=True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __len__(self):
        return int(math.ceil(len(self.datasets)/self.batch_size))
    def __iter__(self):
        indices = np.arange(0, len(self.datasets))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(self.datasets), self.batch_size):
            yield self.datasets[i:i+self.batch_size]

class Transformer:
    def __init__(self):
        self.layers = []
    def add(self, func):
        self.layers.append(func)
    def __call__(self, img):
        for layer in self.layers:
            img = layer(img)
        return img

class Batcher(Sequence):
    def __init__(self, datasets, batch_size, shuffle=False):
        self.datasets = datasets
        self.batch_size = batch_size
        self.indices = np.arange(0, len(self.datasets))
        if shuffle:
            np.random.shuffle(self.indices)
    def __len__(self):
        return math.ceil(len(self.datasets) / self.batch_size)
    def __getitem__(self, idx):
        indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        return self.datasets[indices]

class OldDataset:
    def __init__(self, height, width, focus1, focus2, focus3, focus4, filename):
        self.height = height
        self.width = width
        self.focus1 = focus1
        self.focus2 = focus2
        self.focus3 = focus3
        self.focus4 = focus4
        self.filename = filename

    def do_nothing(self):
        pass


def one_hot(data, size):
    shape = (len(data), size)
    value = np.zeros(shape=shape)
    for i in range(len(data)):
        value[i][data[i]] = 1
    return value
