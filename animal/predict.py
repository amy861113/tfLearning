import cv2
import numpy as np
from tensorflow.keras.models import load_model

tmp = []
testDB = 'animal_test_data/'
test_imgs = testDB+'haung.jpg'

#test_labels = one_hot(test_labels, 4)
test_img = cv2.imread(test_imgs)
test_img = cv2.resize(test_img, (224, 224))
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
tmp.append(test_img)
test = np.array(tmp)

net = load_model('animal.h5')
predict = net.predict(test)
print(predict.argmax(axis=1))
#print(test_labels[0])
'''count = (predict.argmax(axis=1) == test_labels[0].argmax(axis=1)).sum()
print(count/len(test))'''