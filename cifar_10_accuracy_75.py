# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:20:58 2020

@author: Ratul
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,MaxPooling2D,Flatten,Dropout
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.datasets import cifar10
(trainX,trainY),(testX,testY)=cifar10.load_data()


print(testX.shape)
from keras.utils import to_categorical
#trainY = to_categorical(trainY)
#testY = to_categorical(testY)
trainX=trainX.astype('float32')
testX=testX.astype('float32')
trainX /=255
testX /=255

model= Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',padding='same', input_shape=[32,32,3]))
model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2,padding='valid'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2,padding='valid'))
model.add(Dropout(0.4))


model.add(Flatten())
model.add(Dense(units=512,activation='relu'))
model.add(Dense(units=10,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(trainX,trainY,batch_size=10,epochs=20,verbose=1,validation_data=(testX,testY))





