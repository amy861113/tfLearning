from tensorflow.keras.datasets import mnist
import  numpy as np
from myLibrary.util import one_hot
import matplotlib.pyplot as plt

np.set_printoptions(edgeitems=256)


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

'''print(train_images)

img = train_images[10]
plt.imshow(img, cmap='gray')
plt.show()
'''

train_images = train_images/255
test_images = test_images/255

train_images = train_images.reshape((-1, 28*28))
test_images = test_images.reshape((-1, 28*28))
print(train_images.shape)

train_labels = one_hot(train_labels, 10)
#print(train_labels)

test_labels = one_hot(test_labels, 10)
#print(test_labels)

from tensorflow.keras import models, layers

model = models.Sequential()
model.add(layers.Dense(256, input_dim=784, activation='tanh'))
model.add(layers.Dense(128, activation='tanh'))
model.add(layers.Dense(128, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x=train_images, y=train_labels, batch_size=1000, epochs=100, validation_split=0.2)

predict = model.predict(test_images)
print(predict.argmax(axis=1))
print(test_labels.argmax(axis=1))
count = (predict.argmax(axis=1) == test_labels.argmax(axis=1)).sum()
print(count/len(test_images))

plt.plot(history.history['acc'], "r-")
plt.title('Training/validating accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training accuracy', 'validating accuracy'], loc="best")
plt.show()