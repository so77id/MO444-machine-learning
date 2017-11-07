from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import numpy as np
import nltk

from utils.load_dataset import load_dataset,get_hash_ids


from sklearn import metrics
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from nltk.cluster.kmeans import KMeansClusterer


def main(argv):
     # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='input file', required=True)
    parser.add_argument('-n_components', '--n_components', help='number of components in pca', required=True)
    parser.add_argument('-k', '--k', help='k of kmeans', required=True)
    ARGS = parser.parse_args()

    descriptors = load_dataset(ARGS.input_file)

    print("PCA")
    pca = PCA(n_components=int(ARGS.n_components))
    descriptors = pca.fit_transform(descriptors)


    # kmeanModel = KMeans(n_clusters=int(ARGS.k), init='k-means++')
    # kmeanModel.fit(descriptors)
    # predictions = kmeanModel.predict(descriptors)
    #cluster_centers_ = kmeanModel.cluster_centers_
    # print(predictions)


    print("Kmeans")
    kclusterer = KMeansClusterer(int(ARGS.k), distance=nltk.cluster.util.cosine_distance)
    predictions = np.array(kclusterer.cluster(descriptors, assign_clusters=True))
    cluster_centers_ = np.array(kclusterer.means())


    print("Distortions")
    distortion_eu = sum(np.min(distance.cdist(descriptors, cluster_centers_, 'euclidean'), axis=1)) / descriptors.shape[0]
    distortion_cos = sum(np.min(distance.cdist(descriptors, cluster_centers_, 'cosine'), axis=1)) / descriptors.shape[0]


    print("Silhouettes")
    silhouette_score_eu = metrics.silhouette_score(descriptors, predictions, metric='euclidean')
    silhouette_score_cos = metrics.silhouette_score(descriptors, predictions, metric='cosine')

    print("EUCLIDEAN K:", ARGS.k, "distortion:", distortion_eu, "silhouette score:", silhouette_score_eu)
    print("COS K:", ARGS.k, "distortion:", distortion_cos, "silhouette score:", silhouette_score_cos)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))