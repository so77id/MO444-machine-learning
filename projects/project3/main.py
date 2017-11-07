from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import numpy as np

from utils.load_dataset import load_dataset


from sklearn import metrics
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def main(argv):
     # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='input file', required=True)
    parser.add_argument('-n_components', '--n_components', help='number of components in pca', required=True)
    parser.add_argument('-k', '--k', help='k of kmeans', required=True)
    ARGS = parser.parse_args()



    descriptors = load_dataset(ARGS.input_file)

    pca = PCA(n_components=int(ARGS.n_components))
    descriptors = pca.fit_transform(descriptors)


    kmeanModel = KMeans(n_clusters=int(ARGS.k), init='k-means++')
    kmeanModel.fit(descriptors)
    predictions = kmeanModel.predict(descriptors)

    distortion = sum(np.min(distance.cdist(descriptors, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / descriptors.shape[0]

    silhouette_score = metrics.silhouette_score(descriptors, predictions)

    print("K:", ARGS.k, "distortion:", distortion, "silhouette score:", silhouette_score)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))