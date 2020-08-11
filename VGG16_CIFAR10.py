# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:48:54 2020

@author: Ratul
"""

import tensorflow as tf
import keras
from keras.datasets import cifar10
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train.shape
x_train.dtype
x_test.shape

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical

# We normalize the input according to the methods used in the paper
X_train = preprocess_input(x_train)
Y_train = to_categorical(y_train)

# We one-hot-encode the labels for training
X_test = preprocess_input(x_test)
Y_test = to_categorical(y_test)

from tensorflow.keras.applications.vgg16 import VGG16

X_train.shape
x_train.dtype
x_test.shape

model = VGG16(
    weights=None, 
    include_top=True, 
    classes=10,
    input_shape=(32,32,3)
)

# Expand this cell for the model summary
model.summary()
from tensorflow.keras import optimizers

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'model.h5', 
    monitor='val_acc', 
    verbose=0, 
    save_best_only=True, 
    save_weights_only=False,
    mode='auto'
)

# Train the model
history = model.fit(
    x=X_train,
    y=Y_train,
    validation_data=(X_test,Y_test),
    batch_size=256,
    epochs=50,
    callbacks=[checkpoint],
    verbose=1
)

with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()