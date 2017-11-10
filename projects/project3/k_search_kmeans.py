from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import nltk
import numpy as np
import matplotlib.pyplot as plt

from utils.load_dataset import load_dataset

# from methods.baseline import baseline
from scipy.spatial import distance

from sklearn import metrics
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from nltk.cluster.kmeans import KMeansClusterer

plt.style.use('./presentation.mplstyle')

#http://www.sthda.com/english/articles/29-cluster-validation-essentials/96-determining-the-optimal-number-of-clusters-3-must-know-methods/

def main(argv):
     # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='input file', required=True)
    parser.add_argument('-s', '--step', help='step', required=True)
    parser.add_argument('-ik', '--init_k', help='K initial', required=True)
    parser.add_argument('-fk', '--final_k', help='K final', required=True)
    parser.add_argument('-od', '--distortion_out_file', help='elbow distortion graph file', required=True)
    parser.add_argument('-os', '--silhouette_out_file', help='elbow silhoutte graph', required=True)
    parser.add_argument('-pca',  '--pca',        help='with pca', action='store_true')
    parser.add_argument('-k_pca', '--k_pca', help='k pca')
    ARGS = parser.parse_args()

    descriptors = load_dataset(ARGS.input_file)
    if ARGS.pca == True:
        print("With pca")
        pca = PCA(n_components=int(ARGS.k_pca))
        descriptors = pca.fit_transform(descriptors)

    ks = []
    distortions = []
    silhouettes = []

    for k in range(int(ARGS.init_k), int(ARGS.final_k), int(ARGS.step)):
        # kmeanModel = KMeans(n_clusters=k, init='k-means++')
        # kmeanModel.fit(descriptors)
        # predictions = kmeanModel.predict(descriptors)
        # cluster_centers_ = kmeanModel.cluster_centers_

        kclusterer = KMeansClusterer(k, distance=nltk.cluster.util.cosine_distance)
        predictions = kclusterer.cluster(descriptors, assign_clusters=True)
        predictions = np.array(predictions)
        cluster_centers_ = np.array(kclusterer.means())

        distortion = sum(np.min(distance.cdist(descriptors, cluster_centers_, 'cosine'), axis=1)) / descriptors.shape[0]

        silhouette_score = metrics.silhouette_score(descriptors, predictions, metric='cosine')

        distortions.append(distortion)
        silhouettes.append(silhouette_score)
        ks.append(k)

        print("k:", k, "distortion:", distortion, "Silhouette Coefficient", silhouette_score)


    # Plot the elbow with distortion
    fig = plt.figure()
    plt.plot(ks, distortions, 'bx-')
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method')
    fig.savefig(ARGS.distortion_out_file)

    # Plot the elbow with distortion
    fig = plt.figure()
    plt.plot(ks, silhouettes, 'bx-')
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score analysis')
    fig.savefig(ARGS.silhouette_out_file)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))