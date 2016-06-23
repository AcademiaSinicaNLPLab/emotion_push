#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas
import argparse
import os
from feature.extractor import feature_fuse, W2VExtractor, CNNExtractor

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(MODULE_DIR, '..', 'corpus')

def parse_arg(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', help='dataset name')
    return parser.parse_args(argv[1:])

# conjs = ['but', 'although', 'though']
conjs = ["n't"]

def get_only_but_ind(sentences):
    res = []
    for i, sentence in enumerate(sentences):
        s = sentence.split()
        for conj in conjs:
            if conj in s:
                res.append(i)
                break
    return res

if __name__ == "__main__":
    
    args = parse_arg(sys.argv)
    dataset = args.dataset
    corpus = pandas.read_pickle(os.path.join(CORPUS_DIR, dataset+'.pkl'))
    obut_ind = get_only_but_ind(corpus.sentence)

    # corpus.loc[obut_ind].to_csv(os.path.join(CORPUS_DIR, 'data', 'CONJ'+dataset+'.all'), index=False, doublequote=False, sep=' ', header=False, columns=['label','sentence'])
    corpus.loc[obut_ind].to_csv(os.path.join(CORPUS_DIR, 'data', 'NEG'+dataset+'.all'), index=False, doublequote=False, sep=' ', header=False, columns=['label','sentence'])
