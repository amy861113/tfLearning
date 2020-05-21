import matplotlib.pyplot as plt
import os
from myLibrary.util import ImageDataset, MyGenerator, Transformer, Batcher, one_hot
import cv2
from myLibrary.model import vgg16
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model

img_list = []
def read_data(db):
    X = []
    Y = []

    for files in os.listdir(db):
        if os.path.isdir(db+files+'/'):
            for img in os.listdir(db+files+'/'):
                X.append(db+files+'/'+img)
                Y.append(int(files))
    return X, Y

trans = Transformer()
trans.add(lambda img: cv2.resize(img, (224, 224)))
trans.add(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

DB = 'animal_train_data/'
imgs, labels = read_data(DB)
labels = one_hot(labels, 4)
dataset = ImageDataset(imgs, labels, transformer=trans)

from tensorflow.keras import metrics, optimizers

net = VGG16(weights='imagenet', include_top=True)

'''
net = models.Sequential()
net.add(layers.Flatten())

net.add(layers.Dense(4096, activation='relu'))
net.add(layers.Dense(4096, activation='relu'))
net.add(layers.Dense(3, activation='softmax'))


net = Model(input = pretrained.input, output = net)'''
net.summary()

'''net = vgg16(3, (224, 224, 3))
net.summary()

batcher = Batcher(dataset, 10, shuffle=True)
#opt = optimizers.Adam(lr=1e-3, decay=1e-6)
net.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
net.fit(x=batcher, epochs=100)'''

net.save('animal.h5')






