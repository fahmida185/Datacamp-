#
# Compute Ontario Summer School
# Machine Learning in Python
# 13 June 2018
# Erik Spence
#
# This file, knndemo.py, contains a script which will generate random
# data drawn from two gaussian distributions.  It will then perform a
# kNN fit to the data, and then calculate the classification
# boundaries for that fit, given some background grid.  The boundaries
# and data are plotted, and the figure saved.
#

#######################################################################

import numpy as np

from scipy.stats import norm

import sklearn.neighbors as skn

import matplotlib.colors as colors
import matplotlib.pyplot as plt


#######################################################################


def knnplot(num = 100, c1 = -1., sigma1 = 1.5,
            c2 = 1., sigma2 = 1.5, k = 5):

    """
    Generate random data drawn from two gaussian distributions.  The
    function then performs a kNN fit to the data, and calculates the
    classification boundaries for that fit, given a background grid.
    The boundaries and data are plotted, and the figure saved.

    """

    # The number of data drawn from each Gaussian, and the density of
    # the background grid.
    halfnum = int(num / 2)
    gridsize = 0.03

    # Randomly draw from the normal distribution, to get x and y
    # values for the two categories.
    x1 = norm.rvs(size = halfnum, loc = c1, scale = sigma1)
    y1 = norm.rvs(size = halfnum, loc = c1, scale = sigma1)
    x2 = norm.rvs(size = halfnum, loc = c2, scale = sigma2)
    y2 = norm.rvs(size = halfnum, loc = c2, scale = sigma2)

    # Join the data into a single dataset.
    z1 = np.c_[x1, y1]
    z2 = np.c_[x2, y2]
    x = np.concatenate((z1, z2))

    # Create the labels for the data.
    y = np.concatenate(( np.zeros(halfnum), np.zeros(halfnum) + 1))

    # Create the kNN model, and fit it.
    model = skn.KNeighborsClassifier(k)
    model = model.fit(x, y)

    # Get the max and min values of the data.
    minx = min(x[:,0]); maxx = max(x[:,0])
    miny = min(x[:,1]); maxy = max(x[:,1])

    # Create a grid based on the domain, and the grid size.
    xx, yy = np.meshgrid(np.arange(minx, maxx, gridsize),
                         np.arange(miny, maxy, gridsize))

    # Calculate the predicted values for each grid point.
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Create some colour maps.
    cmap_light = colors.ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = colors.ListedColormap(['#FF0000', '#0000FF'])

    # Reshape the predicted data, and plot it as the background.
    z = z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, z, cmap = cmap_light)
    
    # Plot also the training points.
    plt.scatter(x[:, 0], x[:, 1], c = y, cmap = cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # Tighten up, and save the figure.
    plt.tight_layout(0.1)
    plt.savefig('knndemo_k=' + str(k) + '.pdf')

    
#######################################################################


if __name__ == '__main__':

    knnplot(num = 200, c1 = -1., sigma1 = 1.5,
            c2 = 1., sigma2 = 1.5, k = 1)
