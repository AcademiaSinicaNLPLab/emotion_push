#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pickle
import itertools
import numpy as np
import warnings
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cross_validation import StratifiedKFold
from sklearn import grid_search
from feature.extractor import feature_fuse
import logging


class Model(object):
    '''
    A model is a wrapper of any scikit-learn compatible classifier.
    The classifier to be wrapped should implement 'fit' and 'predict' (or 'predict_proba', 'decision_function')  functions.
    '''
    def __init__(self, clf, feature_extractors, OVA=False):
        '''
        @params:
        clf: the classifier instance to be wrapped.
        feature_extractors: a (or a list of) feature extractor (should inherit from feature.extractors.FeatureExtractor)
        OVA: for grid_search, one-versus-all
        '''
        self.clf = clf
        self.feature_extractors = feature_extractors if type(feature_extractors) == list else [feature_extractors]
        self.OVA = OVA

    @property
    def labels(self):
        '''Return labels of the training data'''
        return self.feature_extractors[0].literal_labels

    @staticmethod
    def load_from_file(fname):
        '''Load model from file, including the classifier(s) and feature_extractor(s)
        @params
        fname: file name of the pickled model

        @Return
        Unickled model object
        '''
        with open(fname) as f:
            model = pickle.load(f)
        for c in model.clf:
            try:
                # After loading from the pickled file, we might want to do other stuffs to initialize wrapped classifier(s).
                # To do so, just implment the 'post_load' function in you classifier class.
                c.post_load()
            except AttributeError as e:
                logging.debug("No post_load for {}".format(c.__class__))
        for fe in model.feature_extractors:
            try:
                # Same as the above
                fe.post_load()
            except AttributeError as e:
                logging.debug("No post_load for {}".format(fe.__class__))
        return model

    def dump_to_file(self, fname):
        '''Dump model to pickled file.
        @params
        fname: file name of the pickled model
        '''
        if not isinstance(self.clf, list):
            self.clf = [self.clf]
        for c in self.clf:
            try:
                # Before dumping the pickled file, we might want to delete some (big) members, or save some mebers in another file.
                # To do so, just implment the 'pre_dump' function in you classifier class.
                c.pre_dump(fname)
            except AttributeError as e:
                logging.debug("No pre_dump for {}".format(c.__class__))
        for fe in self.feature_extractors:
            try:
                # Same as the above
                fe.pre_dump()
            except AttributeError as e:
                logging.debug("No pre_dump for {}".format(fe.__class__))
        with open(fname, 'w') as f:
            pickle.dump(self, f)

    def grid_search(self, X, y, n_folds=10, scoring='accuracy', parameters=None, balance=False, **kwargs):
        ''' Do cross_validation with the classifier
        @params
        X: 2D numpy array. # of row equals to sample size. # of columns equal to featue dimension.
        y: 1D numpy array. # of elements equal to sample size.
        n_folds: cross_validation folds
        scoring: estimation metric, scikit-learn compatible (string or function).
        parameters: grid_search parameters, scikit-learn compatible.
        balance: if set True, force balancing the data. Useful for LJ40K corpus.
        kwargs: scikit-learn grid_search compatible.
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clfs = []
            if self.OVA:
                y = MultiLabelBinarizer().fit_transform([[i] for i in y])
                for i in range(y.shape[1]):
                    XX = X
                    yy = y[:, i]
                    if balance and abs(np.sum(yy)-len(yy)/2)>10:
                        posInd, negInd = yy==1, yy==0
                        XXpos, yypos = X[posInd], yy[posInd]
                        XXneg, yyneg = X[negInd], yy[negInd]
                        np.random.shuffle(XXneg)
                        np.random.shuffle(yyneg)
                        XX = np.concatenate((XXpos, XXneg[:len(XXpos)]))
                        yy = np.concatenate((yypos, yyneg[:len(yypos)]))

                    clfs.append(self._grid_search(XX, yy, n_folds, scoring, parameters, balance, **kwargs))
            else:
                clfs.append(self._grid_search(X, y, n_folds, scoring, parameters, **kwargs))
            self.clf = clfs

    def _grid_search(self, X, y, n_folds, scoring, parameters, balance=True, **kwargs):
        # Use balancing labels for each fold
        cv = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
        clf = grid_search.GridSearchCV(self.clf, parameters, scoring=scoring, cv=cv, **kwargs)
        clf.fit(X, y)

        print clf.best_params_, clf.best_score_
        return clf.best_estimator_

    def predict(self, text):
        ''' Delegate predict functions to the wrapped classifier(s).
        @params
        text: raw sentence

        @Return
        A list of value correspong to probability (not gauranteed if decision_function is adopted) of each class.
        '''
        feature = feature_fuse(self.feature_extractors, text)
        if not feature.any():
            return [0] * len(self.clf)

        fn_list = dir(self.clf[0])
        if 'predict_proba' in fn_list:
            res = [c.predict_proba(feature)[0] for c in self.clf]
        elif 'decision_function' in fn_list:
            res = [c.decision_function(feature)[0] for c in self.clf]
        else:
            res = [c.predict(feature)[0] for c in self.clf]

        if len(res)==1:
            return list(res[0])
        else:
            return list(res)
