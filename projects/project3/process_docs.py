from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import numpy as np
import nltk
import string

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from utils.load_dataset import get_hash_ids


def main(argv):
     # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-ids', '--ids_file', help='ids file', required=True)
    parser.add_argument('-dp', '--doc_path', help='docs path', required=True)
    parser.add_argument('-dpp', '--doc_proc_path', help='docs processed path', required=True)
    ARGS = parser.parse_args()

    ids_list = get_hash_ids(ARGS.ids_file)


    stopWords = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    punctuations = list(string.punctuation)
    punctuations.extend(("''", "\n", "\t", ""))

    for id_ in ids_list:
        id_ = id_.split("\n")[0]
        print("processing:", id_)
        with open("{}/{}".format(ARGS.doc_path,id_), 'r') as doc_in, open("{}/{}".format(ARGS.doc_proc_path,id_), 'w') as doc_out:
            text_in = doc_in.read().lower()
            words = [i.strip("".join(punctuations)) for i in word_tokenize(text_in) if i not in punctuations]
            wordsFiltered = []

            for w in words:
                if w not in stopWords and w is not '':
                    w_ = stemmer.stem(w)
                    wordsFiltered.append(w_)

            nwords = np.array(wordsFiltered)

            uwords, counts = np.unique(nwords, return_counts=True)

            sort_index = np.argsort(counts)[::-1]

            uwords_sorted = uwords[sort_index]
            counts_sorted = counts[sort_index]

            for uw, c in zip(uwords_sorted, counts_sorted):
                doc_out.write("{} {}\n".format(uw, c))



if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))