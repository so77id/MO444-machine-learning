from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import numpy as np
import nltk
import string

from utils.load_dataset import get_hash_ids

import matplotlib.pyplot as plt

plt.style.use('./presentation.mplstyle')

def main(argv):
     # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-ids', '--ids_file', help='ids file', required=True)
    parser.add_argument('-dp', '--doc_path', help='docs path', required=True)
    parser.add_argument('-oc', '--out_filename', help='out_filename', required=True)
    ARGS = parser.parse_args()

    # with open("{}/{}".format(ARGS.doc_proc_path,id_), 'w') as doc_out:

    ids_list, news_groups = get_hash_ids(ARGS.ids_file)

    print(len(news_groups))

    news_groups_counts = {}
    for id_, ngs in news_groups.items():
        for ng in ngs:
            if ng not in news_groups_counts:
                news_groups_counts[ng] = 1
            else:
                news_groups_counts[ng] +=1

    vals = np.array([v for v in news_groups_counts.values()])
    news_groups_counts = np.array([[k,v] for k,v in news_groups_counts.items()])

    keys = news_groups_counts[:,0]
    values = news_groups_counts[:,1].astype(np.int)

    i = np.argsort(values)[::-1]

    keys = keys[i]
    values = values[i]
    norm_values = values/values.sum()
    sum_norms = norm_values[:30].sum()
    print(keys.shape)
    print("NORM SUM:", sum_norms)

    print("MEAN:", values.mean())
    print("MEDIAN:", np.median(values))
    print("STD:", values.std())
    print("MODA:", np.bincount(values).argmax())

    plot_values = values[:30]
    plot_keys = keys[:30]

    # Plot the elbow with distortion
    fig, ax = plt.subplots()
    # plt.plot(keys, values, 'bx-')
    ind = np.arange(len(plot_values))  # the x locations for the groups
    width = 0.75 # the width of the bars
    ax.barh(ind, plot_values, width, color="blue")
    ax.set_yticks(ind+width/2)
    ax.set_yticklabels(plot_keys, minor=False)
    plt.title('The 30 most frequent labels')
    plt.xlabel('Frequency')
    plt.ylabel('Label')

    plt.show()
    # fig.savefig(ARGS.distortion_out_file)





if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))