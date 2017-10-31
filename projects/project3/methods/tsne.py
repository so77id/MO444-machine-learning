from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from time import time
from sklearn import manifold

import matplotlib.pyplot as plt


def tsne(descriptors):
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