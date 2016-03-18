#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract articles from doraemon Mongo db"""
import sys
import pickle
import os
from pymongo import MongoClient
import numpy as np
from singleton import Singleton

module_path = os.path.dirname(os.path.abspath(sys.modules[__name__].__file__))
cache_path = os.path.join(module_path, '../../model', 'globve')


class Globve(object):

    def __init__(self, address="doraemon.iis.sinica.edu.tw", dbname="vocabulary_corpus", dimension=300):

        global cache_path
        client = MongoClient(address)
        db = client[dbname]
        self.collection = db["word2vec_" + str(dimension) + 'd']
        self.dimension = dimension
        self.zerovec = np.zeros(self.dimension)
        self.cache_modified = False
        cache_path = cache_path + str(dimension) + '.pkl'
        if os.path.isfile(cache_path):
            with open(cache_path) as f:
                cache = pickle.load(f)
            self.cache, self.zerocache = cache['cache'], cache['zero']
        else:
            self.cache = dict()
            self.zerocache = []

    def __del__(self):
        """
        Cache the word vectors to make the program run more fast next time
        """
        if self.cache_modified:
            print "Writing back globve cache..."
            with open(cache_path, 'w') as f:
                pickle.dump({'cache': self.cache, 'zero': self.zerocache}, f)

    def query_database_and_cache_it(self, key):
        assert type(
            key) == unicode, "The key of Globve is unicode, try 'key'.decode('utf8') to convert it"
        if key in self.cache or key in self.zerocache:  # already cached, do nothing
            return
        cursor = self.collection.find({'word': key})
        if cursor.count() > 0:
            cursor = cursor[0]
            vec = cursor['vec']
            self.cache.update({key: vec})
        else:  # For words not in dataase, store only key instead of zero vector
            vec = self.zerovec
            self.zerocache.append(key)
        self.cache_modified = True

    def __contains__(self, key):  # for 'in' keyword
        self.query_database_and_cache_it(key)
        return key in self.cache

    def __getitem__(self, key):  # for [] operator
        """
        keys: a string
        return: the word vector the word
        """
        self.query_database_and_cache_it(key)
        if key in self.cache:
            return np.array(self.cache[key])
        else:
            raise KeyError
