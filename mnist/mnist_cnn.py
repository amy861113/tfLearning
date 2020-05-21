from tensorflow.keras.datasets import mnist
from myLibrary.util import one_hot
import  numpy as np
np.set_printoptions(edgeitems=256)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(-1, 28, 28, 1)
train_images = train_images/255
train_labels = one_hot(train_labels, 10)

from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x=train_images, y=train_labels, batch_size=1000, epochs=30, validation_split=0.2)

model.save('mnist_cnn.h5')
