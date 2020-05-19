from tensorflow.keras import layers, models


def lenet(output, input_shape=(32, 32, 3)):
    net = models.Sequential()
    net.add(layers.Conv2D(32, kernel_size=5, padding='same', activation='relu', input_shape=input_shape))
    net.add(layers.MaxPool2D(pool_size=2))
    net.add(layers.Conv2D(64, kernel_size=5, padding='same', activation='relu'))
    net.add(layers.MaxPool2D(pool_size=2))
    net.add(layers.Flatten())
    net.add(layers.Dense(1024, activation='relu'))
    net.add(layers.Dense(output, activation='softmax'))
    return net


def vgg16(output, input_shape=(224, 224, 3)):
    net = models.Sequential()
    net.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    net.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    net.add(layers.MaxPool2D(pool_size=2))

    net.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    net.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    net.add(layers.MaxPool2D(pool_size=2))

    net.add(layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    net.add(layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    net.add(layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    net.add(layers.MaxPool2D(pool_size=2))

    net.add(layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'))
    net.add(layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'))
    net.add(layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'))
    net.add(layers.MaxPool2D(pool_size=2))

    net.add(layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'))
    net.add(layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'))
    net.add(layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'))
    net.add(layers.MaxPool2D(pool_size=2))

    net.add(layers.Flatten())

    net.add(layers.Dense(4096, activation='relu'))
    net.add(layers.Dense(4096, activation='relu'))
    net.add(layers.Dense(output, activation='softmax'))

    return net

def myModel(output, input_shape=(128, 128, 3)):

    net = models.Sequential()

    net.add(layers.Conv2D(10, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    net.add(layers.MaxPooling2D(pool_size=2))
    net.add(layers.Conv2D(20, kernel_size=(3, 3), padding='same', activation='relu'))
    net.add(layers.MaxPooling2D(pool_size=2))
    net.add(layers.Conv2D(40, kernel_size=(3, 3), padding='same', activation='relu'))
    net.add(layers.MaxPooling2D(pool_size=2))
    net.add(layers.Flatten())
    net.add(layers.Dense(output, activation='softmax'))

    return net