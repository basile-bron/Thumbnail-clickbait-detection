import numpy as np
import os
import sys
import re
import cv2
import PIL.Image as im
################################################################################
print("##############START IMPORT###############")
def loadImg(dataset,filename):
	img = []

	try:
		#assign fullpath to variable "name"
		name = dataset + filename
		img = im.open(name)
		#print(img.size)
		if img.size != (128,128):
			#print("resizing img :", img)
			#####RESIZING######
			basewidth = 128
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
	#listing the images of the directory
	imgs = os.listdir(dataset)
	#print('number of images', len(imgs))
	#initialise the var
	X = np.zeros((len(imgs)-1,128,128,3))
	#print('shape of X',X.shape)

	Y = np.zeros((len(imgs)-1))
	#for each images
	for i in range(0,len(imgs)-1):
		#print('current image name', imgs[i])
		#assign the image to x
		#print(X.shape)
		X[i] = np.asarray(loadImg(dataset,imgs[i]))
		#if it is a cat (if the image as true in its name)
		Y[i] = imgs[i][:3]
		#print(imgs[i],Y[i])
		#X = X.reshape(-1,len(imgs)-1)

	print("Dataset Loaded", X.shape)
	return X,Y

#importing X
#X,Y = loadDataset('data/')

#import prediction test
predict, trash = loadDataset('predict/')


print("##############END OF IMPORT###############")
