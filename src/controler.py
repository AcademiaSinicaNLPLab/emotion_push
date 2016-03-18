#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import csv
from model import Model
from pymongo import MongoClient
from textblob import TextBlob


def install_all_model(models, dirname, fnames):
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

    def __init__(self, model_dir):
        self.models = {}
        os.path.walk(model_dir, install_all_model, self.models)
        print "All models loaded"

    def list_model(self):
        print self.models
        return {name: model.labels for name, model in self.models.items()}

    def predict(self, model_name, text):
        try:
            text = str(TextBlob(text).translate(to='en'))
        except Exception as e:
            pass

        pred = self.models[model_name].predict(text)
        if sum(pred) == 0:
            return {'res': pred}
        else:
            return {'res': pred}
        return


class Logger(object):

    def __init__(self, address="doraemon.iis.sinica.edu.tw", dbname="emotion_visualiztion", collection_name='log'):

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
