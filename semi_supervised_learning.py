"""
this is a sketch that had the main purpose of expanding the dataset using
semi supervised learning, but LogisticRegression is not fited for thmbnail images

switching to convolutional neural net would defenetly solve the issue
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import numpy as np

#shuffle function
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

#importing manualy tagged dataset
from import_data import loadDataset
X,Y = loadDataset('data_manualy_tag/')

#shuffle manualy tagged dataset
X,Y =unison_shuffled_copies(X ,Y)

#split training and test set of the manualy tagged dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y)
# reshape training set input into a 2D array
nsamples, nx, ny, nz= X_train.shape
X_train = X_train.reshape((nsamples,nx*(ny*nz)))

# reshape training set input into a 2D array
nsamples, nx, ny, nz= X_test.shape
X_test = X_test.reshape((nsamples,nx*(ny*nz)))

#you need to categorise before converting to one hot otherwise it will get the last value of the dataset as the number of classes
"""Categorise data"""
y_train = pd.cut(y_train, bins=[-1,24,49,74,99,np.inf], labels=[0,1,2,3,4])
y_test = pd.cut(y_test, bins=[-1,25,50,75,100,np.inf], labels=[0,1,2,3,4])

###################################################
#Linear regression test
###################################################
X_train = np.c_[np.ones((100, 1)), X_train] #add x0 = 1 to each instances because we don't want to multiply by less than 1

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_reg.intercept_, lin_reg.coef_
y_predict = lin_reg.predict(X_test)
lin_reg.score(X_test, y_test)


theta_best_svd, residuals, rank , s = np.linalg.lstsq(X_train, y_train, rcond=None)
print("theta best :", theta_best_svd)

f = np.linalg.pinv(X_train).dot(y_train)
print("y_predict best :", y_predict)

#PLOT
plt.plot(X_test, y_test, "r-")
#plt.plot(X_test, f, "b-")
plt.plot(X_train, y_train, "b.")
#plt.axis([0, 2, 0, 15])
plt.show()


###################################################
#Logistic regression test
###################################################

# CLUSTERING FOR SEMI SUPERVISED LEARNING
n_labeled = 50
log_reg = LogisticRegression(solver='lbfgs', multi_class='auto',max_iter=10000)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])


#performance of the model on the test set ?
log_reg.score(X_test, y_test)

############################################################################################
#### Tag some of the images
k = 50
kmeans = KMeans(n_clusters=k)
X_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]

#display images
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """

    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
"""
########## manualy tag data#######
#unecessary for this Dataset

X_representative_digits.shape
img = X_representative_digits.reshape([50, 128,128,3])
img.shape
show_images(img[10:20]/255, cols = 1, titles = None)
plt.imshow(X_representative_digits[3]/255)
y_representative_digits = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,])
"""

############################################################################################
X_representative_digits,y_representative_digits = loadDataset('data_manualy_tag/')
# reshape training set input into a 2D array
nsamples, nx, ny, nz= X_representative_digits.shape
X_representative_digits = X_representative_digits.reshape((nsamples,nx*(ny*nz)))
X_representative_digits.shape


log_reg = LogisticRegression(solver='lbfgs', multi_class='auto',max_iter=1000)
log_reg.fit(X_representative_digits, y_representative_digits)
log_reg.score(X_test, y_test)

"""
#LABEL PROPAGATION
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]
"""

y_train_propagated = y_train

#training the model and see performance result
log_reg = LogisticRegression(solver='lbfgs', multi_class='auto',max_iter=10000)
log_reg.fit(X_train, y_train_propagated)
log_reg.score(X_test, y_test)

############################################################################################
#LABEL PARTIAL PROPAGATION
#try to only propagate to the 20% of instances closest to the Centroids
percentile_closest = 20

X_cluster_dist = X_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1

partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

#let's train the model on this partially propageted dataset
log_reg = LogisticRegression(solver='lbfgs', multi_class='auto',max_iter=10000)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
log_reg.score(X_test, y_test)

#################################################################################################
#display graph
#plots
print(y_test.shape)
print(X_test.shape)

colors = ['olive', 'maroon','royalblue',  'forestgreen', 'mediumorchid', 'tan', 'deeppink',  'goldenrod', 'lightcyan', 'navy']
#colors = { 0:'green', 1:'yellow'}
#vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
#plt.scatter(X_test[:, 0], y_test[:, 0], c=vectorizer(y_train_partially_propagated), s=50, cmap='viridis')
plt.plot(X_train)
plt.show()

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)
