from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from utils.load_dataset import load_dataset

def main(argv):
     # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='input file', required=True)
    parser.add_argument('-t', '--threshold', help='variance threshold', required=True)
    parser.add_argument('-o1', '--out_graph_1_file', help='variance graph file name', required=True)
    parser.add_argument('-o2', '--out_graph_2_file', help='variance accumulated graph file name', required=True)
    ARGS = parser.parse_args()

    descriptors = load_dataset(ARGS.input_file)

    pca = PCA()
    pca.fit(descriptors)

    ks = np.arange(pca.explained_variance_ratio_.size) + 1
    variance_accumulated = np.cumsum(pca.explained_variance_ratio_)

    k_ideal = np.argmax(variance_accumulated > np.float64(ARGS.threshold))

    print(k_ideal)

    # Plot the elbow with distortion
    fig = plt.figure()
    plt.plot(ks, pca.explained_variance_ratio_, 'bx')
    plt.axvline(x=k_ideal, color='r')
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('Variance ratio')
    plt.title('PCA Variance Study')
    # plt.show()
    fig.savefig(ARGS.out_graph_1_file)

    # Plot the elbow with distortion
    fig = plt.figure()
    plt.plot(ks, variance_accumulated, 'bx')
    plt.axvline(x=k_ideal, color='r')
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('Accumulated Variance')
    plt.title('PCA Variance Study')
    # plt.show()
    fig.savefig(ARGS.out_graph_2_file)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))