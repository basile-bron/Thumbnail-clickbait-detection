import numpy as np
import os
import sys
#keras
from keras.models import Sequential, Model
from keras.layers import Convolution1D, Convolution2D, MaxPooling1D,MaxPooling2D, Flatten,Dropout, Dense, Embedding, Activation, BatchNormalization, GlobalAveragePooling1D, Input, merge, ZeroPadding1D, ZeroPadding2D
from keras.preprocessing import sequence
from keras.optimizers import RMSprop, Adam, SGD
from keras.regularizers import l2

from keras import backend as K
K.set_image_dim_ordering('th')
#shuffle
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

#custom
print('import custom')
from import_data import X ,Y
X,Y =unison_shuffled_copies(X ,Y)
#assigning train data
print('assigning train data')

print('x')
X_train = X[301:]
print('y')
Y_train = Y[301:]

print('x_test')
X_test = X[:300]
print('y_test')
Y_test = Y[:300]
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

#seting the model
print('setting the model')

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(3,128,128)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

#compile
print('compile')
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=4, nb_epoch=50, shuffle=True, verbose=2)


#debug
print('model layer##################')
print(model.layers)
print('model sumary##################')
print(model.summary())




#model.save_weights("models/detector.finetuned.h5")
