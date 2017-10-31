from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt


def baseline(descriptors):
    # trying k-means

    print("[Baseline] Clustering")
    k_means = KMeans(n_clusters=7, random_state=0)
    k_means.fit(descriptors)

    predictions = k_means.predict(descriptors)
    print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(descriptors, predictions))
    """
    some results:
    k=7
    Silhouette Coefficient: 0.020
    k=3
    Silhouette Coefficient: 0.013
    k=2
    Silhouette Coefficient: 0.011
    k=10
    Silhouette Coefficient: 0.024
    k=30
    Silhouette Coefficient: 0.037
    k=50
    Silhouette Coefficient: 0.048
    k=100
    Silhouette Coefficient: 0.057
    k=500
    Silhouette Coefficient: 0.053

    """

    """pca = PCA(n_components=2)
    pca.fit(descriptors)
    X_reduced = pca.transform(descriptors)
    print("[Baseline] Plotting")s
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=predictions,
           cmap='RdYlBu')
    plt.show()"""

    # 3d projection
    pca = PCA(n_components=3)
    pca.fit(descriptors)
    X_reduced = pca.transform(descriptors)


    print("[Baseline] Plotting")
    fig = plt.figure(1, figsize=(8,8))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=8, azim=200)
    plt.cla()

    ax.scatter(X_reduced[:, 2], X_reduced[:, 0], X_reduced[:, 1], c=predictions)
    plt.show()
