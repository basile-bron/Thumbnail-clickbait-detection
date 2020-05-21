import numpy as np
import os
import sys
import re
import cv2
import PIL.Image as im
import logging
#########################LOGGING######################################
#Creation and configuration of the logger
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename = "log.txt",
                    level = logging.DEBUG, #you can change DEBUG to : INFO,WARNING,ERROR,CRITICAL
                    format = LOG_FORMAT,
                    filemode = 'w')
logger = logging.getLogger()
""" here is the diferent level you can use :
logger.debug("")
logger.info("")
logger.warning("")
logger.error("")
logger.critical("")
"""

###########################IMPORT#####################################
size_image = 128

def loadImg(dataset,filename):
	img = []

	try:
		#assign fullpath to variable "name"
		name = dataset + filename
		img = im.open(name)
		#print(img.size)
		if img.size != (size_image,size_image):
			print("resizing img :", img)
			#####RESIZING######
			basewidth = size_image
			wpercent = (basewidth/float(img.size[0]))
			img = img.resize((basewidth,basewidth), im.ANTIALIAS)
			img.save(dataset+filename)
			name = (dataset+filename)
			#####RESIZING######
		#assign the data of the image to variable img
		img = im.open(name)
		#print the image
		#img.show()
		#print()
		#print(img.shape)
	except Exception as e:
		print('Could not load the image: '+ str(e))
	return img

#Import dataset
def loadDataset(dataset):
	print("############## Loading dataset ###############")

	#listing the images of the directory
	imgs = os.listdir(dataset)
	#print('number of images', len(imgs))
	#initialise the var
	X = np.zeros((len(imgs)-1,size_image,size_image,3))
	#print('shape of X',X.shape)

	Y = np.zeros((len(imgs)-1))
	#for each images
	for i in range(0,len(imgs)-1):

		#print('current image name', imgs[i])
		#assign the image to x
		X[i] = np.asarray(loadImg(dataset,imgs[i]))
		Y[i] = imgs[i][:3]
		#print(imgs[i],Y[i])
		#X = X.reshape(-1,len(imgs)-1)

	Y = Y.astype(int)
	print("Dataset Loaded", X.shape)
	print("############## Dataset Loaded ###############")

	return X,Y


#importing X
#X,Y = loadDataset('data/')

#import prediction test
#predict, trash = loadDataset('predict/')

from random import randrange

#shuffle function
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# Split a dataset into a train and test set
def train_test_split(dataset, split):
  train = dataset[:500]
  test = dataset[500:len(dataset)]
  return train, test
