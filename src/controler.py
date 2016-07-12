#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This module handles the server's API logic.
'''

import sys
import os
import csv
from model import Model
from pymongo import MongoClient
from textblob import TextBlob

sys.path.append('/tools/CKIP/client')
from Tokenizer import Tokenizer

def install_all_model(models, dirname, fnames):
    '''Load all trained model from filename. Store the result in models. This is a call back for os.path.walk.
    @param models: the resulting model list
    @param dirname: current visiting directory
    @param fnames: all file name int current visiting directory
    @see controler.__init__
    '''
    if dirname == '.':
        return
    for fname in fnames:
        path = os.path.join(dirname, fname)
        if '.' in os.path.basename(path):
            continue
        if os.path.isfile(path):
            name = os.path.basename(path)
            print "Loading model {} ...".format(name)
            model = Model.load_from_file(path)
            models[name] = model


class Controler():
    '''Implement Server's APIs' logic.
    '''

    def __init__(self, model_dir):
        '''
        @param model_dir: The directory containing all trained model files
        '''
        self.models = {}
        self.tokenizer = Tokenizer()
        os.path.walk(model_dir, install_all_model, self.models)
        print "All models loaded"

    def list_model(self):
        '''Logic for Server's list_model API
        '''
        print self.models
        return {name: model.labels for name, model in self.models.items()}

    def predict(self, model_name, sentence):
        '''Logic for Server's preict API
        @param model_name: the model to be used to predict the emotion
        @param sentence: the target sentence
        @return: a list of scores, corresponding to the emotion of the selected model
        '''
        try:
            if model_name == 'YAHOO_svm':
                tokenized_us = self.tokenizer.tokenizeStr(sentence.encode('utf8'))[0][0]
                cleanr = re.compile('\([A-Za-z].*?\)')
                tokenized_us = re.sub(cleanr, '', tokenized_us.decode('utf8')).encode('utf8')
                sentence = ' '.join(tokenized_us.split('　'))
            else:
                sentence = str(TextBlob(sentence).translate(to='en'))
        except Exception as e:
            print('Error: '+e)
            pass

        pred = self.models[model_name].predict(sentence)
        if sum(pred) == 0:
            return {'res': pred}
        else:
            return {'res': pred}
        return


class Logger(object):
    '''Implement Server's log API
    '''

    def __init__(self, address="doraemon.iis.sinica.edu.tw", dbname="emotion_push", collection_name='log'):

        client = MongoClient(address)
        db = client[dbname]
        self.collection = db[collection_name]

    def log(self, data):
        try:
            self.collection.insert_one(data)
        except e:
            print e
            return "Failed"
        return "Success"
