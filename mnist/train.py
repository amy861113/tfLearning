from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from myLibrary.util import one_hot

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

test_images = test_images.reshape(-1, 28, 28, 1)
test_images = test_images/255
test_labels = one_hot(test_labels, 10)

net = load_model('mnist_cnn.h5')
predict = net.predict(test_images)
print(predict.argmax(axis=1))
print(test_labels.argmax(axis=1))
count = (predict.argmax(axis=1) == test_labels.argmax(axis=1)).sum()
print(count/len(test_images))