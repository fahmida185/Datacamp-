#
# Compute Ontario Summer School
# Machine Learning in Python
# 13 June 2018
# Erik Spence
#
# This file, make_kmeans_agg_swiss.py, will perform a kmeans
# clustering analysis, and a agglomerative clustering analysis, on a
# swiss roll data set.  It will plot and save the result.
#

#######################################################################

import numpy as np

import sklearn.neighbors as skn
import sklearn.cluster as skc
import sklearn.datasets as skd

import matplotlib.pyplot as plt
import matplotlib.colors as colours

#######################################################################

# Create a colour map for the clusters.
my_cmap = colours.ListedColormap(['red', 'blue', 'black'])

# Create the data, and remove the extra dimension which we don't want.
x, y = skd.make_swiss_roll(1000, noise = 0.4)
x = np.c_[x[:, 0], x[:, 2]]

# Create the agglomerative model, using kneighbors to calculate the
# connectivity.  Fit the model.
AggModel = skc.AgglomerativeClustering(n_clusters = 3,
                                     connectivity = \
                                     skn.kneighbors_graph(x, 30))
AggModel = AggModel.fit(x)

# Create the kMeans model, and fit.
KMeansModel  = skc.KMeans(n_clusters = 3)
KMeansModel = KMeansModel.fit(x)

# Subplot.
plt.subplot(1,2,1)

# Plot the data, with the colours from the Agg clustering.
plt.scatter(x[:, 0], x[:, 1], c = AggModel.labels_, cmap = my_cmap)
plt.title("Agglomerative")

# Other subplot.
plt.subplot(1,2,2)

# Plot the data, with the colours from the KMeans clustering.
plt.scatter(x[:, 0], x[:, 1], c = KMeansModel.labels_, cmap = my_cmap)
plt.title("K-means")

# Add the KMeans cluster centres.
for i in range(3):
    plt.scatter(KMeansModel.cluster_centers_[i][0],
                KMeansModel.cluster_centers_[i][1], c = 'ForestGreen')

# Tighten up and save.
plt.tight_layout(0.1)
plt.savefig('agg_kmeans_swiss.pdf')
