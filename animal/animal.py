import matplotlib.pyplot as plt
import os
from util import ImageDataset, MyGenerator, Transformer, Batcher, one_hot
import cv2
from model import vgg16, myModel
import numpy as np



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

from tensorflow.keras import metrics

net = vgg16(output=4, input_shape=(224, 224, 3))
net.summary()

batcher = Batcher(dataset, 20, shuffle=True)
net.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy])
net.fit(x=batcher, epochs=100)

net.save('animal.h5')






