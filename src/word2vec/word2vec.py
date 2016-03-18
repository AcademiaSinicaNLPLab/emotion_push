import os
from gensim.models import Word2Vec as W2V
from singleton import Singleton

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(MODULE_DIR, '../../word2vec/google_word2vec_pretrained')

class Word2Vec(object):
    __metaclass__ = Singleton

    def __init__(self):
        self.model = W2V.load(model_path)

    def __contains__(self, key):  # for 'in' keyword
        return key in self.model

    def __getitem__(self, key):  # for [] operator
        return self.model[key]
