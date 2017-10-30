import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils.load_dataset import parse_descriptors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import manifold
from time import time


def baseline(argv):
    
    # load dataset and parse descriptors
    descriptors = parse_descriptors(argv)

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


def dbscan(argv):
    # load dataset and parse descriptors
    descriptors = parse_descriptors(argv)

    # trying k-means

    print("[DBSCAN] Clustering")
    db = DBSCAN(eps=0.3).fit(descriptors)
    
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(descriptors, labels))

def tsne_(argv):
    # load dataset and parse descriptors
    descriptors = parse_descriptors(argv)
    n_components = 2

    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    Y = tsne.fit_transform(descriptors)
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))
    fig = plt.figure(1, figsize=(8,8))
    ax = fig.add_subplot(2, 5, 10)
    plt.scatter(Y[:, 0], Y[:, 1], c='red', cmap=plt.cm.Spectral)
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    
    plt.axis('tight')

    plt.show()


if __name__ == "__main__":
    sys.exit(dbscan(sys.argv[1:]))