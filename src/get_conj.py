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

conj_fol = set(['but', 'however', 'nevertheless', 'otherwise', 'yet', 'still', 'nonetheless'])
conj_infer = set(['therefore', 'furthermore', 'consequently', 'thus', 'subsequently', 'eventually', 'hence'])
conj_prev = set(['till', 'until', 'despite', 'though', 'although'])
mod = set(['if', 'might', 'could', 'can', 'would', 'may'])
neg = set(['n\'t', 'not', 'neither', 'never', 'no', 'nor'])

conjs = conj_fol.union(conj_infer).union(conj_prev).union(neg)

def get_conj_ind(sentences):
    res = []
    for i, sentence in enumerate(sentences):
        sent = set(sentence.split())
        if len(sent.intersection(conjs))>0:
            res.append(i)
    return res

if __name__ == "__main__":
    
    args = parse_arg(sys.argv)
    dataset = args.dataset
    corpus = pandas.read_pickle(os.path.join(CORPUS_DIR, dataset+'.pkl'))
    conj_ind = get_conj_ind(corpus.sentence)

    corpus.loc[conj_ind].to_csv(os.path.join(CORPUS_DIR, 'data', 'CONJ'+dataset+'.all'), index=False, doublequote=False, sep=' ', header=False, columns=['label','sentence'])
