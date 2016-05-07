#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import os
import cPickle
import numpy as np
from model import Model
from classifier.cnn import Kim_CNN, RNN, AdaCNN
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
    parser.add_argument('-l', '--level', type=int, default=10, help='number of level')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('-d', '--debug', action='store_true', help='fast debug mode')
    parser.add_argument('-nc', '--no_curriculum', action='store_true', help='no curriculum learning')
    return parser.parse_args(argv[1:])
from keras.callbacks import Callback


class IndexByLevel(object):
    def __init__(self, X, sentence):
        ind_level = [(i, l) for i, l in enumerate(self.call(X, sentence))]
        ind_level = sorted(ind_level, key=lambda x: x[1])
        index = [l[0] for l in ind_level]
        self.index = index

class IndexByLength(IndexByLevel):
    def call(self, X, sentence):
        return [(list(x)+[0]).index(0) for x in X]

class ImproveMonitor(object):
    def __init__(self, max_count=2):
        self.best = 0.
        self.counter = 0
        self.max_count = max_count

    @property
    def no_improve_too_long(self):
        return self.counter == self.max_count

    def feed(self, current):
        if current > self.best:
            self.best = current
            self.counter = 0
        else:
            self.counter+=1

    def recount_from(self, best=0.):
        self.best = best 
        self.counter = 0

class EarlyStop(Callback):
    def __init__(self, patience=2):
        self.monitor = ImproveMonitor(patience)

    def on_epoch_end(self, epoch, logs={}):
        acc = logs.get('val_acc')
        self.monitor.feed(acc)
        if self.monitor.no_improve_too_long:
            self.model.stop_training = True

class Resample(Callback):
    def __init__(self, X, y, sentence, index_by_level, max_level=10):
        super(Resample, self).__init__()
        index = index_by_level(X, sentence).index
        self.X = X[index]
        self.y= y[index]
        self.sentence = sentence[index]
        self.max_level = max_level

        self.level = 0
        self.val_monitor = ImproveMonitor()
        self.resample()

    def resample(self):
        self.level+=1
        split = self.level*len(self.X)/self.max_level
        self.X_sample = self.X[:split]
        self.y_sample = self.y[:split]
        print "Level {}".format(self.level)

    def all_data_used(self):
        return self.level == self.max_level

    def on_epoch_end(self, epoch, logs={}):
        val_acc = logs.get('val_acc')
        self.val_monitor.feed(val_acc)
        if self.val_monitor.no_improve_too_long:
            self.resample()
            self.val_monitor.recount_from(val_acc)

class RecordTest(Callback):
    def __init__(self, X_test, y_test):
        super(RecordTest, self).__init__()
        self.X_test, self.y_test = X_test, y_test
        self.best_val_acc =0.
        self.test_acc = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get('val_acc')
        if current is None:
            warnings.warn('val_acc required')

        pred = self.model.predict(self.X_test, verbose=0)
        test_acc = np.mean(np.equal(np.argmax(pred, axis=-1), np.argmax(self.y_test, axis=-1)))
        if current > self.best_val_acc:
            self.best_val_acc = current
            self.test_acc = test_acc
        print
        print "Test acc(this epoch)/Best test acc: {}/{}".format(test_acc, self.test_acc)

class RecordSentenceError(Callback):
    def __init__(self, X, y, sentence, log=os.path.join(MODULE_DIR, '..', 'result', 'senerror')):
        super(RecordSentenceError, self).__init__()
        self.X = X
        self.y = y
        self.sentence = sentence 
        self.log = open(log, 'w')

    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.X, verbose=0)
        ind = np.not_equal(np.argmax(pred, axis=-1), np.argmax(self.y, axis=-1))
        # for i, s in zip(self.y[ind],self.sentence[ind]):
        #     self.log.write('{}|{}\n'.format(i,s))
        aa = [i for i in range(len(ind)) if ind[i]]
        self.log.write('{}\n'.format(aa))
        self.log.write('='*20+'\n')

if __name__ == "__main__":
    args = parse_arg(sys.argv)
    np.random.seed(args.seed)

    dataset = args.dataset
    corpus = pandas.read_pickle(os.path.join(CORPUS_DIR, dataset+'.pkl'))
    sentence, labels = np.array((corpus.sentence)), list(corpus.label)

    if len(set(corpus.split.values))==1:
        split = None
    else:
        split = corpus.split.values

    cnn_extractor = CNNExtractor(mincount=0)
    X, y = cnn_extractor.extract_train(sentence, labels)

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

    CLF = Kim_CNN
    # CLF = RNN

    cv = StratifiedKFold(y, n_folds=10, shuffle=True)
    y = to_categorical(y)

    test_acc = []
    for train_ind, test_ind in cv:
        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]
        sentence_train = sentence[train_ind]

        cv2 = StratifiedKFold(np.argmax(y_train, axis=1), n_folds=10, shuffle=True)
        train_ind, dev_ind = list(cv2)[0]
        X_train, X_dev = X_train[train_ind], X_train[dev_ind]
        y_train, y_dev = y_train[train_ind], y_train[dev_ind] 
        sentence_train = sentence_train[train_ind]

        recordTest = RecordTest(X_test, y_test)
        earlyStop = EarlyStop(3)

        clf = CLF()(vocabulary_size=cnn_extractor.vocabulary_size,
                    maxlen=X.shape[1],
                    embedding_dim=embedding_dim,
                    nb_class=len(cnn_extractor.literal_labels),
                    embedding_weights=W)

        if not args.no_curriculum:
            resample_on_val = Resample(X_train, y_train, sentence_train, IndexByLength, max_level=args.level)
            while not resample_on_val.all_data_used():
                Xs, ys = resample_on_val.X_sample, resample_on_val.y_sample,
                clf.fit(Xs, ys,
                        batch_size=50, nb_epoch=1, verbose=2,
                        validation_data=(X_dev, y_dev),
                        callbacks=[resample_on_val, recordTest])
        clf.fit(X_train, y_train,
                batch_size=50, nb_epoch=100, verbose=2,
                validation_data=(X_dev, y_dev),
                callbacks=[RecordSentenceError(X_test, y_test, sentence[test_ind]), recordTest, earlyStop])

        test_acc.append(recordTest.test_acc)
        break

    print test_acc, np.average(test_acc)
