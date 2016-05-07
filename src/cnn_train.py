#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import os
import cPickle
import numpy as np
from model import Model
np.random.seed(10)
from classifier.cnn import Kim_CNN, AdaCNN, RecordTest
from sklearn.cross_validation import StratifiedKFold, train_test_split
from word2vec.word2vec import Word2Vec
import pandas
from feature.extractor import feature_fuse, W2VExtractor, CNNExtractor
from keras.utils.np_utils import to_categorical

import logging
logging.basicConfig(level=logging.DEBUG)


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(MODULE_DIR, '..', 'corpus')
LEXICON_DIR = os.path.join(MODULE_DIR, '..', 'lexicon')
MODEL_DIR = os.path.join(MODULE_DIR, '..', 'model')
CACHE_DIR = os.path.join(MODULE_DIR, '..', 'cache')

def load_embedding(vocabulary, cache_file_name):
    if os.path.isfile(cache_file_name):
        with open(cache_file_name) as f:
            return cPickle.load(f)
    else:
        res = np.random.uniform(low=-0.05, high=0.05, size=(len(vocabulary), 300))
        w2v = Word2Vec()
        for word in vocabulary.keys():
            if word in w2v:
                ind = vocabulary[word]
                res[ind] = w2v[word]
        with open(cache_file_name, 'w') as f:
            cPickle.dump(res, f)
        return res

def parse_arg(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', help='dataset name')
    parser.add_argument('-e', '--epoch', type=int, default=20, help='number of epoch')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('-d', '--debug', action='store_true', help='fast debug mode')
    return parser.parse_args(argv[1:])

if __name__ == "__main__":
    args = parse_arg(sys.argv)
    np.random.seed(args.seed)

    dataset = args.dataset
    corpus = pandas.read_pickle(os.path.join(CORPUS_DIR, dataset+'.pkl'))
    sentences, labels = list(corpus.sentence), list(corpus.label)

    if len(set(corpus.split.values))==1:
        split = None
    else:
        split = corpus.split.values

    cnn_extractor = CNNExtractor(mincount=0)
    X, y = cnn_extractor.extract_train(sentences, labels)

    if args.debug:
        logging.debug('Embedding is None!!!!!!!!!!!!')
        W = None
        embedding_dim = 300
    else:
        logging.debug('loading embedding..')
        W = load_embedding(cnn_extractor.vocabulary,
                           cache_file_name=os.path.join(CACHE_DIR, dataset + '_emb.pkl'))
        embedding_dim=W.shape[1]

    logging.debug('embedding loaded..')

    m=1

    if m==1:
        CLF = Kim_CNN
    elif m==2:
        CLF = AdaCNN
    elif m==3:
        CLF = RNN

    clf = CLF(vocabulary_size=cnn_extractor.vocabulary_size,
              maxlen=X.shape[1],
              embedding_dim=embedding_dim,
              nb_class=len(cnn_extractor.literal_labels),
              embedding_weights=W)

    model = Model(clf, cnn_extractor)
    dump_file = os.path.join(MODEL_DIR, dataset + '_cnn')

    if split is None:
        test_acc = []
        cv = StratifiedKFold(y, n_folds=10, shuffle=True)
        y = to_categorical(y)
        for train_ind, test_ind in cv:
            X_train, X_test = X[train_ind], X[test_ind]
            y_train, y_test = y[train_ind], y[test_ind]
            X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1)
            callback = RecordTest(X_test, y_test)
            clf.fit(X_train, y_train,
                    batch_size=50,
                    nb_epoch=args.epoch,
                    validation_data=(X_dev, y_dev),
                    callbacks=[callback])
            test_acc.append(callback.test_acc)
            print "test_acc: {}".format(callback.test_acc)

        print test_acc, np.average(test_acc)
    else:
        y = to_categorical(y)
        train_ind, dev_ind, test_ind = (split=='train', split=='dev', split=='test')
        X_train, X_dev, X_test = X[train_ind], X[dev_ind], X[test_ind]
        y_train, y_dev, y_test = y[train_ind], y[dev_ind], y[test_ind]
        callback = RecordTest(X_test, y_test)
        clf.fit(X_train, y_train,
                batch_size=50,
                nb_epoch=args.epoch,
                validation_data=(X_dev, y_dev))

    model.dump_to_file(dump_file)
