from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.cluster import DBSCAN
from sklearn import metrics


def dbscan(descriptors):

    # trying k-means

    print("[DBSCAN] Clustering")
    db = DBSCAN(eps=0.3).fit(descriptors)

    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(descriptors, labels))
