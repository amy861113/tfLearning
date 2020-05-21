
from tensorflow.keras.models import load_model

net = load_model('animal.h5')
predict = net.predict(test_images)
print(predict.argmax(axis=1))
print(test_labels.argmax(axis=1))
count = (predict.argmax(axis=1) == test_labels.argmax(axis=1)).sum()
print(count/len(test_images))

plt.plot(history.history['acc'], "r-")
plt.plot(history.history['val_acc'], "b--")
plt.title('Training/validating accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training accuracy', 'validating accuracy'], loc="best")
plt.show()