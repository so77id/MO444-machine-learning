import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils.load_dataset import load_dataset
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def baseline(argv):
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='input file', required=True)
    parser.add_argument('-ds', '--descriptor_size', help='descriptor size', required=True)
    parser.add_argument('-nd', '--n_descriptors', help='number of descriptors', required=True)
    ARGS = parser.parse_args()


    print("[Baseline] Loading dataset")
    descriptors = load_dataset(ARGS.input_file)
    # trying k-means

    print("[Baseline] Clustering")
    k_means = KMeans(n_clusters=7, random_state=0)
    k_means.fit(descriptors)

    predictions = k_means.predict(descriptors)

    """pca = PCA(n_components=2)
    pca.fit(descriptors)
    X_reduced = pca.transform(descriptors)
    print("[Baseline] Plotting")
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



if __name__ == "__main__":
    sys.exit(baseline(sys.argv[1:]))