import numpy as np
import os
import sys
#keras
from keras.models import Sequential, Model
from keras.layers import Convolution1D, Conv2D, MaxPooling1D,MaxPooling2D, Flatten,Dropout, Dense, Embedding, Activation, BatchNormalization, GlobalAveragePooling1D, Input, merge, ZeroPadding1D, ZeroPadding2D
from keras.preprocessing import sequence
from keras.optimizers import RMSprop, Adam, SGD
from keras.regularizers import l2

#from keras import backend as K
#K.set_image_dim_ordering('th')

#for gpu support
#K.tensorflow_backend._get_available_gpus()
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

#shuffle
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

#custom
print('import custom')
from import_data import *
#importing X and Y
X,Y = loadDataset('data_manualy_tag/')
# cheking integrity
print("#integrity check#")
print(Y[0:10])







X,Y = unison_shuffled_copies(X ,Y)
X_train, X_test= train_test_split(X,0.8)
Y_train, Y_test= train_test_split(Y,0.8)

#assigning train data
print('assigning train data')

print('x')
#X_train = X[301:1301]
print('y')
#Y_train = Y[301:1301]

print('x_test')
#X_test = X[:300]
print('y_test')
#Y_test = Y[:300]
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)




#seting the model
print('setting the model')
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(size_image,size_image,3)))
model.add(Conv2D(64,( 3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64,( 3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128,( 3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128,( 3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256,( 3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256,( 3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256,( 3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256,( 3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,( 3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,( 3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,( 3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,( 3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,( 3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,( 3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,( 3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,( 3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))






#compile
print('compile')
#sgd = SGD(lr=0.6, decay=1e-6, momentum=1.9)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=2, epochs=10, shuffle=True, verbose=1)

try:
    model.save_weights("/content/drive/My Drive/data/detector.finetuned.h5")
except Exception as e:
    print(e)





# calculate predictions
predictions = model.predict(predict)
print(predictions)
for i in range(len(predict)):
	print("Predicted= ", predictions[i])
#debug

#debug
print('model layer##################')
print(model.layers)
print('model sumary##################')
print(model.summary())




#model.save_weights("models/detector.finetuned.h5")
