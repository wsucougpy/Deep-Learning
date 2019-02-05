# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 16:56:06 2019

@author: Jikhan Jeong
"""

## 2019 Spring, MNIST with CNN
## reference : https://github.com/gilbutITbook/006958

from keras.datasets  import mnist
from keras.utils     import np_utils
from keras.models    import Sequential
from keras.layers    import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

## Setting for handling randomness to generate the same results in each trials
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed) 

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')/255  
# 60000 column, and 28x 28 feature space / 255 for normalization to making it between 0 and 1

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')/255  

Y_train = np_utils.to_categorical(Y_train,10) # making dummies, class, randge of dummies
Y_test  = np_utils.to_categorical(Y_test,10)

model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), input_shape = (28, 28, 1), activation = 'relu')) # 32 mask input 28 * 28 2d matrix 1 means balck 
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25)) # %25 node out to escape overfitting
model.add(Flatten()) # 2D -> 1D
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) # %50 nodes out to escape overfitting
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer ='adam',
              metrics=['accuracy'])

MODEL_DIR ='C:/python/a_python/2019_Spring_Deep_learning/model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
modelpath ='C:/python/a_python/2019_Spring_Deep_learning/model/{epoch:02d} - {val_loss:.4f}.hdf5'
    
checkpointer = ModelCheckpoint(filepath = modelpath, monitor ='val_loss', verbose =1, save_best_only = True)
# val_acc = test set accuracy, val_loss = test set error, loss = train set error
# saving position = modelpath, verbose =1 printed, otherwise not    
early_stopping_callback = EarlyStopping(monitor='val_loss',patience=1)


history = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 10, 
                    batch_size = 200, verbose = 0, callbacks =[early_stopping_callback,checkpointer])

print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1])) # .4f means 4 decimal percision

y_test_loss  = history.history['val_loss'] # test  set error
y_train_loss = history.history['loss']     # train set error

x_len = np.arange(len(y_train_loss)) # y_train_loss = 19, x_len = arrange(19)

plt.legend(loc='upper right')
plt.axis([0, 20, 0, 0.35])
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x_len, y_test_loss,  marker='.', c="red" , label ='Test_error')
plt.plot(x_len, y_train_loss, marker='.', c="blue", label ='Train_error')
plt.show()