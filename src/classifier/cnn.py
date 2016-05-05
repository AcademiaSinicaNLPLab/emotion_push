#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import warnings
import math
import numpy as np
from keras.models import Sequential, Model, model_from_json
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D, ZeroPadding1D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute, Lambda
from keras.layers import Input, Layer, merge
from keras.layers.wrappers import TimeDistributed
from keras import initializations
from keras import activations
from keras import backend as K
from keras import regularizers
from keras.regularizers import l2
from keras import constraints
from keras.constraints import MaxNorm
from keras.callbacks import Callback

import theano
import theano.tensor as T
sys.setrecursionlimit(50000)

import logging
logging.basicConfig(level=logging.DEBUG)

l2norm=1e-3
maxmorm=10

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

class MyEmbedding(Embedding):
    def __init__(self, input_dim, output_dim, use_mask=True, **kwargs):
        self.use_mask = use_mask
        super(MyEmbedding, self).__init__(input_dim, output_dim, **kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.use_mask:
            m = np.ones((self.input_dim, self.output_dim),
                        dtype=theano.config.floatX)
            m[0] = [0] * self.output_dim
            mask = theano.tensor.constant(m, dtype=theano.config.floatX)
            outW = K.gather(self.W, X)
            outM = K.gather(mask, X)
            return outW * outM
        else:
            return K.gather(self.W, X)

class CNNS(KerasClassifier):
    def add_embedding(self, model, vocabulary_size, embedding_dim, maxlen, use_my_embedding, embedding_weights=None):
        if embedding_weights is not None:
            if use_my_embedding:
                model.add(MyEmbedding(vocabulary_size, embedding_dim,
                                      input_length=maxlen, weights=[embedding_weights]))
            else:
                model.add(Embedding(vocabulary_size, embedding_dim,
                                    input_length=maxlen, weights=[embedding_weights]))
        else:
            if use_my_embedding:
                model.add(MyEmbedding(vocabulary_size,
                                      embedding_dim, input_length=maxlen))
            else:
                model.add(Embedding(vocabulary_size,
                                    embedding_dim, input_length=maxlen))

    def add_pad(self, model, maxlen):
        padding_len = self.calc_padding_len(maxlen)
        if padding_len>0:
            model.add(ZeroPadding1D(padding_len))
        maxlen = maxlen+2*padding_len
        logging.debug('maxlen changed to {}'.format(maxlen))
        return maxlen

    def log_params(self, params):
        weights = params.pop('embedding_weights')
        if weights is not None:
            params.update({'embedding_weights': 'given'})
        else:
            params.update({'embedding_weights': 'random'})
        print params

    def add_full(self, model, drop_out_prob, nb_class):
        model.add(Dropout(drop_out_prob))

        assert(nb_class > 1)
        model.add(Dense(nb_class, W_constraint=MaxNorm(m=9, axis=0)))
        model.add(Activation('softmax'))
        model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    def post_load(self):
        with open(self.arch_file) as f:
            self.model = model_from_json(
                f.read(), {"MyEmbedding": MyEmbedding})
        self.model.load_weights(self.weight_file)

    def pre_dump(self, dump_file):
        self.arch_file = dump_file + '_arch.json'
        self.weight_file = dump_file + '_weights.h5'

        with open(self.arch_file, 'w') as f:
            f.write(self.model.to_json())
        self.model.save_weights(self.weight_file, overwrite=True)
        del self.model

    def calc_padding_len(self, ori_len):
        assert ori_len % 2 ==0
        align = [2**i for i in range(10)]
        if ori_len not in align:
            for l in align:
                if l>ori_len:
                    return (l-ori_len)/2

class Kim_CNN(CNNS):
    def __call__(self,
                 vocabulary_size=5000,
                 maxlen=100,
                 embedding_dim=300,
                 nb_filter=100,
                 filter_length=[3],
                 nb_class=2,
                 drop_out_prob=0.5,
                 use_my_embedding=True,
                 embedding_weights=None):

        self.log_params(locals())
        model = Sequential()
        self.add_embedding(model, vocabulary_size, embedding_dim, maxlen, use_my_embedding, embedding_weights)
        # maxlen = self.add_pad(model, maxlen)

        model.add(self.convLayer(maxlen, embedding_dim, nb_filter, filter_length))
        # model.add(MaxPooling1D(maxlen))
        # model.add(Flatten())

        self.add_full(model, drop_out_prob, nb_class)
        model.summary()
        return model

    def convLayer(self, maxlen, embedding_dim, nb_filter, filter_length):
        main_input = Input(shape=(maxlen, embedding_dim), name='main_input')
        convs = []
        for i in filter_length:
            conv = Convolution1D(nb_filter=nb_filter,
                                 filter_length=i,
                                 border_mode='valid',
                                 activation='relu',
                                 subsample_length=1,
                                 input_shape=(maxlen, embedding_dim))(main_input)
            pool = MaxPooling1D(pool_length=maxlen-i+1)(conv)
            convs.append(Flatten()(pool))

        if len(convs)>1:
            output = merge(convs, mode='concat', concat_axis=-1)
        else:
            output = convs[0]
        return Model(main_input, output)

class AdaCNN(CNNS):
    def __call__(self,
                 vocabulary_size=5000,
                 maxlen=100,
                 embedding_dim=300,
                 gr_hidden_dim=50,
                 clf_hidden_dim=100,
                 gr_act='tanh',
                 clf_act='relu',
                 nb_class=2,
                 drop_out_prob=0,
                 use_my_embedding=True,
                 embedding_weights=None):

        model = Sequential()
        self.add_embedding(model, vocabulary_size, embedding_dim, maxlen, use_my_embedding, embedding_weights)

        # Reduce embedding size
        # model.add(TimeDistributed(Dense(gr_hidden_dim, W_regularizer=l2(l2norm), W_constraint=MaxNorm(maxmorm))))
        model.add(TimeDistributed(Dense(gr_hidden_dim)))

        model.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda x:(x[0], x[2])))
        model.add(Dense(nb_class))
        model.add(Activation('softmax'))
        model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

        # Construct hierarchy
        model.add(GrCNNLayer(hidden_dim=gr_hidden_dim,
                             init='glorot_uniform',
                             activation=gr_act,
                             input_length=maxlen,
                             pooling='max',))

        # Gating classifier
        model.add(self.weighted_mlf(maxlen, gr_hidden_dim, clf_hidden_dim, nb_class, clf_act, drop_out_prob))
        # model.add(Lambda(lambda x: x[:, -1, :], output_shape=lambda x: (x[0], x[2])))
        # model.add(Dense(nb_class))

        model.add(Activation('softmax'))
        model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def weighted_mlf(self, maxlen, gr_hidden_dim, clf_hidden_dim, nb_class, clf_act, drop_out_prob):
        hierarchy = Input(shape=(maxlen, gr_hidden_dim), name='hierarchy')
        experts = TimeDistributed(Dense(clf_hidden_dim, activation=clf_act, W_regularizer=l2(l2norm), W_constraint=MaxNorm(maxmorm)))(hierarchy)
        experts = Dropout(drop_out_prob)(experts)
        experts = TimeDistributed(Dense(nb_class, activation='softmax', W_regularizer=l2(l2norm), W_constraint=MaxNorm(maxmorm)))(experts)

        weights = TimeDistributed(Dense(1, W_regularizer=l2(l2norm), W_constraint=MaxNorm(maxmorm)))(hierarchy)
        weights = Lambda(lambda x: K.tile(x, (1,1,2)), output_shape=lambda x: (x[0], x[1], 2))(weights)

        score = merge([experts, weights], mode='mul')
        score = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda x:(x[0],2))(score)

        model = Model(hierarchy, score)
        return model

class GrCNNLayer(Layer):
    ''' Generate pyramid from sentences image

    # Input shape
        3D tensor with shape: `(nb_samples, sequence_length, input_dim)`.

    # Output shape
        3D tensor with shape: `(nb_samples, sequence_length, output_dim)`.

    # Arguments
      init: name of initialization function for the weights
          of the layer (see: [initializations](../initializations.md)),
          or alternatively, Theano function to use for weights initialization.
          This parameter is only relevant if you don't pass a `weights` argument.
      input_length: Length of input sequences, when it is constant.
      pooling: str. Polling method used to compress the pyramid
      hidden_dim: dimensionality of the input (integer).
    '''

    def __init__(self, hidden_dim,
                 init='glorot_uniform',
                 activation='tanh',
                 input_length=None,
                 pooling='max',
                 **kwargs):

        self.hidden_dim = hidden_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.maxlen = input_length
        self.pooling = pooling

        if self.hidden_dim:
            kwargs['input_shape'] = (self.maxlen, self.hidden_dim)
        super(GrCNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        hidden_dim = input_shape[2]

        # W^l, W^r, parameters used to construct the central hidden representation
        self.Wl = self.init((self.hidden_dim, self.hidden_dim), name='{}_Wl'.format(self.name))
        self.Wr = self.init((self.hidden_dim, self.hidden_dim), name='{}_Wr'.format(self.name))
        self.Wb = K.zeros((self.hidden_dim,), name='{}_Wb'.format(self.name))
        # G^l, G^r, parameters used to construct the three-way coefficients
        self.Gl = self.init((self.hidden_dim, 3), name='{}_Gl'.format(self.name))
        self.Gr = self.init((self.hidden_dim, 3), name='{}_Gr'.format(self.name))
        self.Gb = K.zeros((3,), name='{}_Gb'.format(self.name))
        
        self.trainable_weights = [self.Wl, self.Wr, self.Wb, self.Gl, self.Gr, self.Gb]

        self.regularizers=[]
        for _ in range(4):
            self.regularizers.append(l2(l2norm))
        self.regularizers[0].set_param(self.Wl)
        self.regularizers[1].set_param(self.Wr)
        self.regularizers[2].set_param(self.Gl)
        self.regularizers[3].set_param(self.Gr)

        self.constraints={}
        self.constraints[self.Wl]=MaxNorm(maxmorm)
        self.constraints[self.Wr]=MaxNorm(maxmorm)
        self.constraints[self.Gl]=MaxNorm(maxmorm)
        self.constraints[self.Gr]=MaxNorm(maxmorm)

    def call(self, x, mask=None):
        maxlen = x.shape[1]

        hidden0 = x
        # shape: (batch_size, maxlen, hidden_dim) 
        pyramid, _ = theano.scan(fn=self.build_pyramid, 
                                 sequences=T.arange(maxlen-1),
                                 outputs_info=[hidden0],
                                 non_sequences=maxlen)
        # shape: (maxlen-1, batch_size, maxlen, hidden_dim)

        hidden0 = K.expand_dims(hidden0, dim=0)
        # shape: (1, batch_size, maxlen, hidden_dim)

        pyramid = K.concatenate([hidden0, pyramid], axis=0)
        # shape: (maxlen, batch_size, maxlen, hidden_dim)

        hierarchy, _ = theano.scan(fn=self.compress_pyramid,
                                   sequences=[T.arange(maxlen, 0, -1), 
                                              pyramid])
        # shape: (maxlen, batch_size, hidden_dim)

        hierarchy = K.permute_dimensions(hierarchy, (1, 0, 2))
        # shape: (batch_size, maxlen, hidden_dim)
        
        return hierarchy

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.maxlen, self.hidden_dim)

    def build_pyramid(self, t, sent_mat, maxlen):
        '''
        # Arguments
            t: level index.
            sent_mat: input tensor at level t.
                shape (batch_size, maxlen, hidden_dim)
            maxlen: max level number; sentence length
        '''
        # Left, right, and center hidden representations.
        # shape: (batch_size, maxlen-1, hidden_dim)
        left_mat = sent_mat[:, :maxlen-t-1]
        right_mat = sent_mat[:, 1:maxlen-t]
        central_raw = (K.dot(left_mat, self.Wl) + 
                       K.dot(right_mat, self.Wr) + 
                       self.Wb)
        central_mat = self.activation(central_raw)

        # Gates for the hidden representations
        # shape: (batch_size, maxlen-1, 3)
        gates_raw = K.dot(left_mat, self.Gl) + K.dot(right_mat, self.Gr) + self.Gb
        gates = activations.softmax(gates_raw)
        # Equivalent implementation: gates, _ = theano.scan(T.nnet.softmax, sequences=gates_raw)

        # Extract left, right, and central gates.
        # shape: (batch_size, maxlen-1, 1). Note that we extend the tensors' dimension for them to be broadcastable
        left_gate = gates[:, :, 0].dimshuffle(0, 1, 'x')
        central_gate = gates[:, :, 1].dimshuffle(0, 1, 'x')
        right_gate = gates[:, :, 2].dimshuffle(0, 1, 'x')

        # Build next level of hidden representation using soft combination
        # shape: (batch_size, maxlen-1, hidden_dim). Note that gates are broadcast along the last dimension.
        next_level = left_gate * left_mat + \
            right_gate * right_mat + \
            central_gate * central_mat
    
        # Set the (maxlen-1) rows but return all the maxlen rows back
        return T.set_subtensor(sent_mat[:, :maxlen-t-1], next_level)

    def compress_pyramid(self, t, sent_mat):
        '''
        # Arguments
            t: level index
            sent_mat: input tensor at level t
            shape: (batch_size, maxlen, hidden_dim), only the (batch_size, :t, :) sub-tensor is meaningful
        '''
        if self.pooling == 'max':
            return K.max(sent_mat[:, :t, :], axis=1)
        elif self.pooling == 'averaging':
            return K.mean(sent_mat[:, :t, :], axis=1)
        else:
            raise NotImplementedError('The pooling method {} is not implemented in {}'.format(self.pooling, AdaSent.__name__))
