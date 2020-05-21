from tensorflow.keras.datasets import cifar10
import numpy as np
from myLibrary.util import one_hot

np.set_printoptions(edgeitems=256)

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

'''import matplotlib.pyplot as plt

img = train_images[0]
plt.imshow(img)
plt.show()
'''

tags = ['飛機', '汽車', '鳥', '貓', '鹿', '狗', '青蛙', '馬', '船', '卡車']

print(train_images.shape)

train_images = train_images/255
test_images = test_images/255
print(train_images.shape)

train_labels = one_hot(train_labels, 10)
test_labels = one_hot(test_labels, 10)

print(type(test_images))

from myLibrary.model import to_cifar10

net = to_cifar10(10)
net.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
net.fit(x=train_images, y=train_labels, batch_size=1000, epochs=20, validation_split=0.2)

predict = net.predict(test_images)
count = (predict.argmax(axis=1) == test_labels.argmax(axis=1)).sum()
print(count/len(test_images))