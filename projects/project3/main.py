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



def baseline(argv):
    
    # load dataset and parse descriptors
    descriptors = parse_descriptors(argv)

    # trying k-means

    print("[Baseline] Clustering")
    k_means = KMeans(n_clusters=7, random_state=0)
    k_means.fit(descriptors)

    predictions = k_means.predict(descriptors)

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


if __name__ == "__main__":
    sys.exit(dbscan(sys.argv[1:]))