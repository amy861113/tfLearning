import pandas as pd
import numpy as np
from myLibrary.util import randSplit

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

''''
randSplit運作原理:先創建inds大小陣列,然後random Inds, 切割成兩個陣列, 回傳資料(陣列inds)
inds = [0, 1, 2, 3, 4] => inds = [1, 3, 4, 0, 2] => testind = inds[:2], traininds[2:] => 
return data[1, 3, 4], data[0, 2]
'''

#讀入資料, 並切割出測試集
train = pd.read_csv('iris.csv')
train, test = randSplit(train, 0.2)

#檢查資料是否有null值
print('train.info():')
print(train.info())
print('test.info():')
print(test.info())

feature = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
trainX = train[feature]
testX = test[feature]

label = ['variety']
trainY = train[label]
testY = test[label]

#label為英文字串, 所以做one_hot
trainY = pd.get_dummies(trainY)
testY = pd.get_dummies(testY)

print('trainY:')
print(trainY)
print('testY:')
print(testY)

print('trainX.describe(歸一化前):')
print(trainX.describe())
trainX = (trainX - trainX.min()) / (trainX.max() - trainX.min())
print('trainX.describe(歸一化後):')
print(trainX.describe())

print('testX.describe(歸一化前):')
print(testX.describe())
testX = (testX - testX.min()) / (testX.max() - testX.min())
print('test.describe(歸一化後):')
print(testX.describe())

from tensorflow.keras import models, layers

model = models.Sequential()

model.add(layers.Dense(8, input_dim=4, activation='tanh'))
model.add(layers.Dense(8, activation='tanh'))
model.add(layers.Dense(8, activation='tanh'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x=trainX.to_numpy(), y=trainY.to_numpy(), batch_size=1000, epochs=500, validation_split=0.2)

predict = model.predict(testX[0:].to_numpy())

print(predict.argmax(axis=1))
print(testY[0:].to_numpy().argmax(axis=1))

count = predict.argmax(axis=1) == testY[0:].to_numpy().argmax(axis=1)
print(count/len(testX))