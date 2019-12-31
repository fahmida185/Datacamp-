#
# Compute Ontario Summer School
# Machine Learning in Python
# 13 June 2018
# Erik Spence
#
# This file, make_kmeans_iris.py, contains a script runs a kmeans
# analysis on the iris data, and plots the result.
#

#######################################################################

import sklearn.cluster as skc
import sklearn.datasets as skd

import matplotlib.pyplot as plt
import matplotlib.colors as colours

#######################################################################

# Create a colour map, for the clusters.
my_cmap = colours.ListedColormap(['red', 'blue', 'black'])

# Load the iris data.
iris = skd.load_iris()

x = iris.data
y = iris.target

# Create a KMeans model, and fit.
model  = skc.KMeans(n_clusters = 3)
model = model.fit(x)

# Plot the result.
plt.scatter(x[:, 0], x[:, 2], c = model.labels_, cmap = my_cmap)

# Plot the centres of the clusters.
for i in range(3):
    plt.scatter(model.cluster_centers_[i][0],
                model.cluster_centers_[i][2], c = 'ForestGreen')

# Add the labels of the axes.
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[2])

# Tighten up, and save.
plt.tight_layout(0.1)
plt.savefig('kmeans_iris.pdf')
