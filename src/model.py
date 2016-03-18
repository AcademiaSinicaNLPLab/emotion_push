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
    def __init__(self, clf, feature_extractors, OVO=False):
        self.clf = clf
        self.feature_extractors = feature_extractors if type(feature_extractors) == list else [feature_extractors]
        self.OVO = OVO

    @property
    def labels(self):
        return self.feature_extractors[0].literal_labels

    @staticmethod
    def load_from_file(fname):
        with open(fname) as f:
            model = pickle.load(f)
        for c in model.clf:
            try:
                c.post_load()
            except AttributeError as e:
                logging.debug("No post_load for {}".format(c.__class__))
        for fe in model.feature_extractors:
            try:
                fe.post_load()
            except AttributeError as e:
                logging.debug("No post_load for {}".format(fe.__class__))
        return model

    def dump_to_file(self, fname):
        for c in self.clf:
            try:
                c.pre_dump(fname)
            except AttributeError as e:
                logging.debug("No pre_dump for {}".format(c.__class__))
        for fe in self.feature_extractors:
            try:
                fe.pre_dump()
            except AttributeError as e:
                logging.debug("No pre_dump for {}".format(fe.__class__))
        with open(fname, 'w') as f:
            pickle.dump(self, f)

    def grid_search(self, X, y, n_folds=10, scoring='accuracy', parameters=None, **kwargs):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clfs = []
            if self.OVO:
                y = MultiLabelBinarizer().fit_transform([[i] for i in y])
                for i in range(y.shape[1]):
                    clfs.append(self._grid_search(X, y[:, i], n_folds, scoring, parameters, **kwargs))
            else:
                clfs.append(self._grid_search(X, y, n_folds, scoring, parameters, **kwargs))
            self.clf = clfs

    def _grid_search(self, X, y, n_folds, scoring, parameters, **kwargs):
        cv = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
        clf = grid_search.GridSearchCV(self.clf, parameters, scoring=scoring, cv=cv, **kwargs)
        clf.fit(X, y)

        print clf.best_params_, clf.best_score_
        return clf.best_estimator_

    def predict(self, text):
        feature = feature_fuse(self.feature_extractors, text)
        if not feature.any():
            return [0] * len(self.clf)

        fn_list = dir(self.clf[0])
        if 'predict_pboba' in fn_list:
            return [c.predict_proba(feature)[0] for c in self.clf]
        elif 'decision_function' in fn_list:
            return [c.decision_function(feature)[0] for c in self.clf]
        else:
            return [c.predict(feature)[0] for c in self.clf]
