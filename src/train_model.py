#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import os
import cPickle
import numpy as np
np.random.seed(0)
from sklearn.svm import LinearSVC
from classifier.cnn import Kim_CNN, MyCNN, Multi_OneD_CNN, MyRegularCNN, RecordTest
from sklearn.cross_validation import StratifiedKFold, train_test_split
from model import Model
from word2vec.word2vec import Word2Vec
import dataloader
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


def select_feature(feature_types):
    if len(feature_types) == 1:
        return [feature_types[0]()]
    query = 'Possible Features:\n'
    for i, ftp in enumerate(feature_types):
        query += '{}. {}\n'.format(i, ftp.__name__)
    selection = raw_input(query)
    selection = [int(f) for f in selection.split(',')]
    return [feature_types[i]() for i in selection]


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

def reorder_embedding():
    w2v = Word2Vec()
    lexicon_fname = os.path.join(LEXICON_DIR, 'connotation/connotation.csv'),
    cache_file_name = os.path.join(CACHE_DIR, 'emborder.pkl')

    if os.path.isfile(cache_file_name):
        with open(cache_file_name) as f:
            return cPickle.load(f)
    else:
        import pandas as pd
        df = pd.read_csv(lexicon_fname, header=None)
        df.columns = ['word', 'sent']

        def get_emb(df, sent):
            words = list(df['word'][df['sent'] == sent])
            emb = []
            for word in words:
                if word in w2v:
                    emb.append(w2v[word])
            return np.array(emb)

        pos_mat = get_emb(df, 'positive')
        neg_mat = get_emb(df, 'negative')

        avg_diff = np.average(pos_mat, axis=0) - np.average(neg_mat, axis=0)
        var_sum = np.var(pos_mat, axis=0) + np.var(neg_mat, axis=0)
        score = avg_diff / var_sum
        res = np.argsort(score)

        with open(cache_file_name, 'w') as f:
            cPickle.dump(res, f)
        return res

def transform_embedding(emb):
    import theano.tensor as T
    from theano import function

    EMB = T.dmatrix('EMB')
    AXIS = T.dmatrix('AXIS')
    transform = function([EMB, AXIS], T.dot(EMB, T.transpose(AXIS)))

    w2v = Word2Vec()
    with open(os.path.join(LEXICON_DIR, 'emo12_wordnet.pkl')) as f:
        axis = cPickle.load(f)['embedding']

    return transform(emb, axis)

def parse_arg(argv):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', help='dataset name')
    parser.add_argument('-e', '--epoch', type=int, default=20, help='number of epoch')
    return parser.parse_args(argv[1:])

if __name__ == "__main__":
    args = parse_arg(sys.argv)
    dataset = args.dataset

    corpus = pandas.read_pickle(os.path.join(CORPUS_DIR, dataset+'.pkl'))
    sentences, labels = list(corpus['sentence']), list(corpus['label'])
    if len(set(corpus.split.values))==1:
        split = None
    else:
        split = corpus.split.values

    logging.debug('loading feature..')
    cnn_extractor = CNNExtractor(mincount=0)
    X, y = cnn_extractor.extract_train(sentences, labels)
    logging.debug('feature loaded')

    pretrained = True
    if pretrained:
        logging.debug('loading embedding..')
        W = load_embedding(cnn_extractor.vocabulary,
                           cache_file_name=os.path.join(CACHE_DIR, dataset + '_emb.pkl'))
        logging.debug('embedding loaded..')
    else:
        W = None

    embedding_dim = W.shape[1] if W is not None else 300

    layer=1
    # clf = MyCNN(
    # clf = Multi_OneD_CNN(
    clf = Kim_CNN(
    # clf = Good_Kim_CNN(
    # clf = MyRegularCNN(
        vocabulary_size=cnn_extractor.vocabulary_size,
        nb_filter=100,
        layer=layer,
        embedding_dim=embedding_dim,
        filter_length=[3, 4, 5],
        drop_out_prob=0.5,
        maxlen=X.shape[1],
        use_my_embedding=True,
        nb_class=len(cnn_extractor.literal_labels),
        embedding_weights=W)

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
                    show_accuracy=True,
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
                show_accuracy=True,
                validation_data=(X_dev, y_dev))

    # feature_extractors = [W2VExtractor(use_globve=True)]
    # X, y = feature_fuse(feature_extractors, sentences, labels)
    # clf = LinearSVC()
    # OVO = True
    # parameters = dict(C=np.logspace(-5, 1, 8))
    # dump_file = os.path.join(MODEL_DIR, dataset + '_svm')
    # model = Model(clf, feature_extractors, OVO=OVO)
    # model.grid_search(X, y, parameters=parameters, n_jobs=-1)

    # feature_extractors = [CNNExtractor(mincount=0)]
    # X, y = feature_fuse(feature_extractors, sentences, labels)
    # clf = OneD_CNN(vocabulary_size=cnn_extractor.vocabulary_size,
    #           nb_filters=100,
    #           embedding_dim=300,
    #           drop_out_prob=0.5,
    #           maxlen=X.shape[1],
    #           nb_class=len(cnn_extractor.literal_labels))
    # OVO = False
    # parameters = dict(batch_size=[50], nb_epoch=[20], show_accuracy=[True])
    # dump_file = os.path.join(MODEL_DIR, dataset + '_cnn')
    # model = Model(clf, feature_extractors, OVO=OVO)
    # model.grid_search(X, y, parameters=parameters, n_jobs=1)

    # model.dump_to_file(dump_file)
    # for fe in feature_extractors:
    #     del fe
    # del feature_extractors
