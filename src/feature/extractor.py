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
    '''
    Concate mutiple feature vectors extracted by a list of feature_extractors. 

    @param feature_extractors: list of feature_extractors, each item shold be an (inherited) class of FeatureExtractor
    @param sentences: list of sentences to be extracted
    @param labels: literal labels of each sentence


    @return X: 2D numpy array, feature vectors, one sentence per row 
    @return y: 1D numpy array, numbered label of each sentence
    '''
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
        if len(feature_extractors)==1:
            return Xs[0]
        else:
            return np.concatenate(Xs, axis=1)

class FeatureExtractor(object):
    '''A feature extractor extract features from raw sentence.'''

    def extract_train(self, sentences, labels):
        ''' Extract feature vectors and numbered labels from training data.
        @param sentences: list of sentences to be extracted
        @param labels: literal labels of each sentence

        @return X: 2D numpy array, feature vectors, one sentence per row 
        @return y: 1D numpy array, numbered label of each sentence
        '''
        literal_labels = list(set(labels))
        print "Labels: ", literal_labels
        y = np.array([literal_labels.index(l) for l in labels])

        sentences = [preprocess(s) for s in sentences]
        self.pre_calculate(sentences)

        Xs = []
        X = np.array([self._extract(s) for s in sentences])
        self.literal_labels = literal_labels
        return X, y

    def extract(self, sentence):
        '''Extract the feature vector for a testing sentence. The sentence is first turned into a list of words and then the feature extraction logic is delegated to _extract.'''
        return self._extract(preprocess(text))

    def pre_calculate(self, sentences):
        '''Implement this function when a extractor needs to calculate some global infomation of the trainig data before extracting feature vector for each sentence.
        @param senteces: a list of training sentences
        '''
        pass

    def _extract(self, wordarray):
        '''The real logic to extract feature vector of a single sentence.
        @param wordarray: a list of words from the original sentence.
        '''
        raise NotImplementedError

class W2VExtractor(FeatureExtractor):
    def __init__(self, use_globve=False):
        self.use_globve = use_globve
        if self.use_globve:
            self.model = Globve()  # wordvector model
        else:
            self.model = Word2Vec()  # wordvector model

    def pre_dump(self):
        # It's silly to pickle the whole word embedding.
        del self.model

    def post_load(self):
        # Load the word embedding bacause we didn't store it when dumping.
        if self.use_globve:
            self.model = Globve()  # wordvector model
        else:
            self.model = Word2Vec()  # wordvector model

    def _extract(self, wordarray):
        X = np.zeros(300)
        i = 0
        for word in [w.decode('utf8') for w in wordarray]:
            if word in self.model:
                i += 1
                X = X + self.model[word]
        if i > 0:
            X = X / i
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
        # if maxlen % 2 == 1:
        #     maxlen+=1

        self.maxlen = maxlen
        pad_sentences = [self.to_given_length(s, self.maxlen) for s in sentences]
        word_counts = Counter(itertools.chain(*pad_sentences))
        # ind -> word
        vocabulary_inv = [x[0] for x in word_counts.most_common() if x[1] > self.mincount]
        assert vocabulary_inv[0] == self.padding_word  # padding should be the most frequent one
        # word -> ind
        self.vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    def _extract(self, wordarray):
        wordarray = self.to_given_length(wordarray, self.maxlen)

        res = np.zeros(self.maxlen, dtype=int)
        for i, word in enumerate(wordarray):
            if word in self.vocabulary:
                res[i] = self.vocabulary[word]
            else:
                res[i] = self.vocabulary[self.padding_word]
        return res

    def to_given_length(self, wordarray, length):
        wordarray = wordarray[:length]
        return wordarray + [self.padding_word] * (length - len(wordarray))

class ConjWeightCNNExtractor(CNNExtractor):
    def __init__(self, mincount=0):
        from nltk.corpus import stopwords
        super(ConjWeightCNNExtractor, self).__init__(mincount)
        self.conj_fol = set(['but', 'however', 'nevertheless', 'otherwise', 'yet', 'still', 'nonetheless'])
        self.conj_infer = set(['therefore', 'furthermore', 'consequently', 'thus', 'subsequently', 'eventually', 'hence'])
        self.conj_prev = set(['till', 'until', 'despite', 'though', 'although'])
        self.mod = set(['if', 'might', 'could', 'can', 'would', 'may'])
        self.neg = set(['n\'t', 'not', 'neither', 'never', 'no', 'nor'])
        # self.no_emotion = set(['.']+stopwords.words('english'))
        self.neg_win = 5

        self.conj = self.conj_fol.union(self.conj_infer).union(self.conj_prev)
        self.All = self.conj.union(self.neg)

    def extract_weight(self, wordarray):
        len_words = len(wordarray)
        wordarray = self.to_given_length(wordarray, self.maxlen)
        wordarray = np.array(wordarray)

        weight = np.zeros(len(wordarray))
        flip = np.zeros(len(wordarray))
        weight[:len_words] = 1
        flip[:len_words] = 1

        periods = [-1] + list(np.where(wordarray=='.')[0]) + [len_words]
        for b, end in zip(periods[:-1], periods[1:]):
            beg = b+1
            sentence = wordarray[beg:end]
            for i, word in enumerate(sentence):
                if word in self.conj_fol or word in self.conj_infer:
                    for j in range(i+1, len(sentence)):
                        if sentence[j] in self.conj:
                            break
                        else:
                            weight[beg+j]+=1
                elif word in self.conj_prev:
                    for j in range(i-1, 0, -1):
                        if sentence[j] in self.conj:
                            break
                        else:
                            weight[beg+j]+=1
                elif word in self.neg:
                    for j in range(i+1, min(i+1+self.neg_win, len(sentence))):
                        if sentence[j] in self.All:
                            break
                        else:
                            flip[beg+j]=-1
                # if word in self.All or word in self.no_emotion:
                if word in self.All:
                    weight[beg+i] = 0
        return weight*flip

    def _extract(self, wordarray):
        sent_res = super(ConjWeightCNNExtractor, self)._extract(wordarray)
        weight = self.extract_weight(wordarray)
        # tmp = zip(wordarray, weight)
        # print tmp
        return tuple((sent_res, weight))

ConjWeightOneVecCNNExtractor = ConjWeightCNNExtractor

class ConjWeightNegVecCNNExtractor(ConjWeightCNNExtractor):
    def extract_weight(self, wordarray):
        len_words = len(wordarray)
        wordarray = self.to_given_length(wordarray, self.maxlen)
        wordarray = np.array(wordarray)

        weight = np.zeros(len(wordarray))
        flip = np.zeros(len(wordarray))
        weight[:len_words] = 1

        periods = [-1] + list(np.where(wordarray=='.')[0]) + [len_words]
        for b, end in zip(periods[:-1], periods[1:]):
            beg = b+1
            sentence = wordarray[beg:end]
            for i, word in enumerate(sentence):
                if word in self.conj_fol or word in self.conj_infer:
                    for j in range(i+1, len(sentence)):
                        if sentence[j] in self.conj:
                            break
                        else:
                            weight[beg+j]+=1
                elif word in self.conj_prev:
                    for j in range(i-1, 0, -1):
                        if sentence[j] in self.conj:
                            break
                        else:
                            weight[beg+j]+=1
                elif word in self.neg:
                    for j in range(i+1, min(i+1+self.neg_win, len(sentence))):
                        if sentence[j] in self.All:
                            break
                        else:
                            flip[beg+j]=1
                # if word in self.All or word in self.no_emotion:
                if word in self.All:
                    weight[beg+i] = 0
        return weight, flip

    def _extract(self, wordarray):
        sent_res = CNNExtractor._extract(self, wordarray)
        weight, flip = self.extract_weight(wordarray)
        # tmp = zip(wordarray, weight)
        # print tmp
        return tuple((sent_res, weight, flip))

ConjWeightTwoVecCNNExtractor = ConjWeightNegVecCNNExtractor

class ConjWeightAllVecCNNExtractor(ConjWeightCNNExtractor):
    def extract_weight(self, wordarray):
        len_words = len(wordarray)
        wordarray = self.to_given_length(wordarray, self.maxlen)
        wordarray = np.array(wordarray)

        weight = np.zeros(len(wordarray))
        flip = np.zeros(len(wordarray))
        weight[:len_words] = 1
        flip[:len_words] = 1

        periods = [-1] + list(np.where(wordarray=='.')[0]) + [len_words]
        for b, end in zip(periods[:-1], periods[1:]):
            beg = b+1
            sentence = wordarray[beg:end]
            for i, word in enumerate(sentence):
                if word in self.conj_fol or word in self.conj_infer:
                    for j in range(i+1, len(sentence)):
                        if sentence[j] in self.conj:
                            break
                        else:
                            weight[beg+j]+=1
                elif word in self.conj_prev:
                    for j in range(i-1, 0, -1):
                        if sentence[j] in self.conj:
                            break
                        else:
                            weight[beg+j]+=1
                elif word in self.neg:
                    for j in range(i+1, min(i+1+self.neg_win, len(sentence))):
                        if sentence[j] in self.All:
                            break
                        else:
                            flip[beg+j]=-1
                # if word in self.All or word in self.no_emotion:
                if word in self.All:
                    weight[beg+i] = 0
                    
        res = weight*flip
        for i in range(len_words):
            if res[i]==-1:
                res[i]=3
            if res[i]==-2:
                res[i]==4
        return res
