import itertools
import numpy as np
from collections import Counter
from word2vec.globvemongo import Globve
from word2vec.word2vec import Word2Vec
import copy
from preprocessor import preprocess
from functools import reduce
import sys


def feature_fuse(feature_extractors, sentences, labels=None):
    Xs = []

    if labels is not None:
        ys = []
        for fe in feature_extractors:
            X, y = fe.extract_train(sentences, labels)
            Xs.append(X)
            ys.append(np.array(y))
        if len(ys) > 1:
            assert(reduce(lambda a, b: (a == b).all, [y for y in ys]))
        X = Xs[0] if len(Xs) == 1 else np.concatenate(Xs, axis=1)
        return X, y
    else:
        for fe in feature_extractors:
            Xs.append(fe.extract(sentences))
        return Xs[0] if len(Xs) == 1 else np.concatenate(Xs, axis=1)


class FeatureExtractor(object):

    def extract_train(self, sentences, labels):
        literal_labels = list(set(labels))
        print "Labels: ", literal_labels
        y = np.array([literal_labels.index(l) for l in labels])

        # sentences = [s.split(' ') for s in sentences]
        sentences = [preprocess(s) for s in sentences]
        self.pre_calculate(sentences)

        Xs = []
        X = np.array([self._extract(s).flatten() for s in sentences])
        self.literal_labels = literal_labels
        return X, y

    def extract(self, text):
        return self._extract(preprocess(text))

    def pre_calculate(self, sentences):
        pass

    def _extract(self, text):
        raise NotImplementedError


class W2VExtractor(FeatureExtractor):
    def __init__(self, use_globve=False):
        self.use_globve = use_globve
        self.post_load()

    def pre_dump(self):
        del self.model

    def post_load(self):
        if self.use_globve:
            self.model = Globve()  # wordvector model
        else:
            self.model = Word2Vec()  # wordvector model

    def _extract(self, text):
        X = np.zeros(300)
        i = 0
        for word in [w.decode('utf8') for w in text]:
            if word in self.model:
                i += 1
                X = X + self.model[word]
        if i > 0:
            X = X / i
        # if float(i)/len(text)<0.6:
        #     X = np.zeros_like(X)

        return X


class CNNExtractor(FeatureExtractor):
    def __init__(self, mincount=0):
        self.padding_word = "<PAD/>"
        self.mincount = mincount

    @property
    def vocabulary_size(self):
        return len(self.vocabulary)

    def pre_calculate(self, sentences):
        maxlen = max(len(x) for x in sentences)
        if maxlen % 2 == 1:
            maxlen+=1

        self.maxlen = maxlen
        pad_sentences = [self.to_given_length(s, self.maxlen) for s in sentences]
        word_counts = Counter(itertools.chain(*pad_sentences))
        # ind -> word
        vocabulary_inv = [x[0] for x in word_counts.most_common() if x[1] > self.mincount]
        assert vocabulary_inv[0] == self.padding_word  # padding should be the most frequent one
        # word -> ind
        self.vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    def _extract(self, text):
        text = self.to_given_length(text, self.maxlen)

        res = np.zeros((1, self.maxlen), dtype=int)
        for i, word in enumerate(text):
            if word in self.vocabulary:
                res[0][i] = self.vocabulary[word]
            else:
                res[0][i] = self.vocabulary[self.padding_word]
        return res

    def to_given_length(self, sentence, length):
        sentence = sentence[:length]
        return sentence + [self.padding_word] * (length - len(sentence))

# TfIdf Not done yet
class TfIdfExtractor(FeatureExtractor):
    def pre_calculate(self, sentences):
        vocabulary = self.build_vocab(sentences)
        word_doc_mat = np.zeros((len(vocabulary), len(sentences)))
        for d, sent in enumerate(sentences):
            for word in sent:
                if word in vocabulary:
                    word_doc_mat[vocabulary[word], d] += 1

        self.F = [0] * len(vocabulary)

    def extract(self, text):
        self.d += 1

    def build_vocab(self, sentences, mincount=1):
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common() if x[1] > mincount]
        # Mapping from word to index
        return {x: i for i, x in enumerate(vocabulary_inv)}
