#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import warnings
import math
import numpy as np
from keras.models import Sequential, Model, model_from_json
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Layer, Input, merge, Dense, Embedding, Dropout, Activation, Flatten, Reshape, Permute, Lambda, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras import initializations
from keras import regularizers
from keras import constraints
from keras import activations
from keras import backend as K
from keras.constraints import MaxNorm
from keras.regularizers import l1l2

if K._BACKEND=='tensorflow':
    import tensorflow as tf
else:
    import theano.tensor as T

class MyEmbedding(Embedding):
    def __init__(self, input_dim, output_dim,
                 init='uniform', input_length=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None,
                 mask_zero=False,
                 weights=None, dropout=0., fill_zero=None, **kwargs):
        super(MyEmbedding, self).__init__(input_dim, output_dim,
                                          init=init,
                                          input_length=input_length,
                                          W_regularizer=W_regularizer,
                                          activity_regularizer=activity_regularizer,
                                          W_constraint=W_constraint,
                                          mask_zero=mask_zero,
                                          weights=weights,
                                          dropout=dropout,
                                          **kwargs)
        if fill_zero is None:
            self.fill_zero = [0]*self.output_dim
        else:
            assert(len(fill_zero))==self.output_dim
            self.fill_zero = fill_zero

    def call(self, X, mask=None):
        m1 = np.ones((self.input_dim, self.output_dim))
        m1[0] = [0]*self.output_dim
        m2 = np.zeros((self.input_dim, self.output_dim))
        m2[0] = self.fill_zero
        if K._BACKEND=='theano':
            M1 = T.constant(m1, dtype=self.W.dtype)
            M2 = T.constant(m2, dtype=self.W.dtype)
        else:
            M1 = tf.constant(m1, dtype=self.W.dtype)
            M2 = tf.constant(m2, dtype=self.W.dtype)
        outW = K.gather(self.W, X)
        outM1 = K.gather(M1, X)
        outM2 = K.gather(M2, X)
        return outW*outM1+outM2

class MaskEmbedding(Embedding):
    def __init__(self, input_dim, output_dim,
                 init='uniform', input_length=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None,
                 mask_zero=False,
                 weights=None, dropout=0., fixed_index=None, **kwargs):
        super(MaskEmbedding, self).__init__(input_dim, output_dim,
                                            init=init,
                                            input_length=input_length,
                                            W_regularizer=W_regularizer,
                                            activity_regularizer=activity_regularizer,
                                            W_constraint=W_constraint,
                                            mask_zero=mask_zero,
                                            weights=weights,
                                            dropout=dropout,
                                            **kwargs)
        self.fixed_index = fixed_index

    def call(self, X, mask=None):
        m1 = np.ones((self.input_dim, self.output_dim))
        m2 = np.array(self.initial_weights[0]).copy() 
        for i in range(m2.shape[0]):
            if i in self.fixed_index:
                # fixxed, should mask out original embedding
                m1[i] = 0
            else:
                # not fixxed, should keep the original embedding (add zero to the origin) 
                m2[i] = 0

        if K._BACKEND=='theano':
            M1 = T.constant(m1, dtype=self.W.dtype)
            M2 = T.constant(m2, dtype=self.W.dtype)
        else:
            M1 = tf.constant(m1, dtype=self.W.dtype)
            M2 = tf.constant(m2, dtype=self.W.dtype)
        outW = K.gather(self.W, X)
        outM1 = K.gather(M1, X)
        outM2 = K.gather(M2, X)
        return outW*outM1+outM2

class CNNS(KerasClassifier):
    def get_emb_layer(self, vocabulary_size, embedding_dim, maxlen, embedding_weights=None, fill_zero=None, **kwargs):

        if embedding_weights is not None:
            return MyEmbedding(vocabulary_size, embedding_dim, input_length=maxlen, weights=[embedding_weights], fill_zero=fill_zero, **kwargs)
        else:
            return MyEmbedding(vocabulary_size, embedding_dim, input_length=maxlen, fill_zero=fill_zero, **kwargs)

    def log_params(self, params):
        weights = params.pop('embedding_weights')
        if weights is not None:
            params.update({'embedding_weights': 'given'})
        else:
            params.update({'embedding_weights': 'random'})
        print params

    def add_full(self, model, drop_out_prob, nb_class):
        ''' For keras Sequential api'''
        model.add(Dropout(drop_out_prob))
        assert(nb_class > 1)
        model.add(Dense(nb_class, W_constraint=MaxNorm(m=9, axis=0)))
        model.add(Activation('softmax'))

    def get_full(self, drop_out_prob, nb_class):
        ''' For keras functional api'''
        def pseudo_layer(layer):
            clf = Dropout(drop_out_prob)(layer)
            assert(nb_class > 1)
            clf = Dense(nb_class, W_constraint=MaxNorm(m=9, axis=0))(clf)
            clf = Activation('softmax')(clf)
            return clf
        return pseudo_layer

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

    def compile(self, model):
        model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

class Kim_CNN(CNNS):
    def __call__(self,
                 vocabulary_size=5000,
                 maxlen=100,
                 embedding_dim=300,
                 embedding_weights=None,
                 nb_filter=100,
                 filter_length=[3,4,5],
                 nb_class=2,
                 drop_out_prob=0.):

        self.log_params(locals())
        model = Sequential()
        model.add(self.get_emb_layer(vocabulary_size, embedding_dim, maxlen, embedding_weights))
        model.add(self.conv_Layer(maxlen, embedding_dim, nb_filter, filter_length))
        self.add_full(model, drop_out_prob, nb_class)
        self.compile(model)
        return model

    def conv_Layer(self, maxlen, embedding_dim, nb_filter, filter_length, border_mode='same'):
        main_input = Input(shape=(maxlen, embedding_dim), name='main_input')
        convs = []
        for i in filter_length:
            conv = Convolution1D(nb_filter=nb_filter,
                                 filter_length=i,
                                 border_mode=border_mode,
                                 activation='relu',
                                 subsample_length=1,
                                 input_shape=(maxlen, embedding_dim))(main_input)
            convs.append(conv)

        if len(convs)>1:
            output = merge(convs, mode='concat', concat_axis=-1)
        else:
            output = convs[0]
        output = MaxPooling1D(pool_length=maxlen)(output)
        output = Flatten()(output)
        return Model(main_input, output)

class ConjWeight_CNN(Kim_CNN):
    def __call__(self,
                 vocabulary_size=5000,
                 maxlen=100,
                 embedding_dim=300,
                 embedding_weights=None,
                 nb_filter=100,
                 filter_length=[3,4,5],
                 nb_class=2,
                 drop_out_prob=0.):

        self.log_params(locals())
        sent_input = Input(shape=(maxlen,), dtype='int32')
        emb = self.get_emb_layer(vocabulary_size, embedding_dim, maxlen, embedding_weights)(sent_input)

        weight_input = Input((maxlen,)) 
        weight = RepeatVector(embedding_dim)(weight_input)
        weight = Permute((2,1))(weight)

        weighted_emb = merge([emb, weight], mode='mul')
        conv = self.conv_Layer(maxlen, embedding_dim, nb_filter, filter_length)(weighted_emb)
        clf = self.get_full(drop_out_prob, nb_class)(conv)

        model = Model([sent_input, weight_input], clf)
        self.compile(model)
        return model

class ConjWeightNegVec_CNN(Kim_CNN):
    def __call__(self,
                 vocabulary_size=5000,
                 maxlen=100,
                 embedding_dim=300,
                 embedding_weights=None,
                 nb_filter=100,
                 filter_length=[3,4,5],
                 nb_class=2,
                 drop_out_prob=0.,
                 l1 = 0.,
                 l2 = 0.):

        self.log_params(locals())

        sent_input = Input(shape=(maxlen,), dtype='int32')
        emb = self.get_emb_layer(vocabulary_size, embedding_dim, maxlen, embedding_weights)(sent_input)

        weight_input = Input((maxlen,)) 
        weight = RepeatVector(embedding_dim)(weight_input)
        weight = Permute((2,1))(weight)

        neg_input = Input((maxlen,), dtype='int32')
        neg_init = -1*np.ones((2, embedding_dim))
        neg_weight = self.get_emb_layer(2, embedding_dim, maxlen, embedding_weights=neg_init, fill_zero=[1]*embedding_dim, W_regularizer=l1l2(l1,l2))(neg_input)

        weighted_emb = merge([emb, weight, neg_weight], mode='mul')
        conv = self.conv_Layer(maxlen, embedding_dim, nb_filter, filter_length)(weighted_emb)
        clf = self.get_full(drop_out_prob, nb_class)(conv)

        model = Model([sent_input, weight_input, neg_input], clf)
        self.compile(model)
        return model

class ConjWeightAllVec_CNN(Kim_CNN):
    def __call__(self,
                 vocabulary_size=5000,
                 maxlen=100,
                 embedding_dim=300,
                 embedding_weights=None,
                 nb_filter=100,
                 filter_length=[3,4,5],
                 nb_class=2,
                 drop_out_prob=0.,
                 l1 = 0.,
                 l2 = 0.):

        self.log_params(locals())

        sent_input = Input(shape=(maxlen,), dtype='int32')
        emb = self.get_emb_layer(vocabulary_size, embedding_dim, maxlen, embedding_weights)(sent_input)

        weight_input = Input((maxlen,), dtype='int32') 
        weight_init = np.array([0,1,2,-1,-2])[:, None]*np.ones(embedding_dim)
        weight = self.get_emb_layer(5, embedding_dim, maxlen, embedding_weights=weight_init, W_regularizer=l1l2(l1,l2))(weight_input)

        weighted_emb = merge([emb, weight], mode='mul')
        conv = self.conv_Layer(maxlen, embedding_dim, nb_filter, filter_length)(weighted_emb)
        clf = self.get_full(drop_out_prob, nb_class)(conv)

        model = Model([sent_input, weight_input], clf)
        self.compile(model)
        return model

class Scale(Layer):
    def __init__(self, input_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, W_constraint=None, **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights

        kwargs['input_shape'] = (self.input_dim,)
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.W = self.init((input_dim,),
                           name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        output = x*self.W
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ConjWeightOneVec_CNN(Kim_CNN):
    def __call__(self,
                 vocabulary_size=5000,
                 maxlen=100,
                 embedding_dim=300,
                 embedding_weights=None,
                 nb_filter=100,
                 filter_length=[3,4,5],
                 nb_class=2,
                 drop_out_prob=0.,
                 l1 = 0.,
                 l2 = 0.):

        self.log_params(locals())

        sent_input = Input(shape=(maxlen,), dtype='int32')
        emb = self.get_emb_layer(vocabulary_size, embedding_dim, maxlen, embedding_weights)(sent_input)
        scaled_emb = TimeDistributed(Scale(embedding_dim, weights=[np.ones(embedding_dim)], W_regularizer=l1l2(l1,l2)))(emb)

        weight_input = Input((maxlen,)) 
        weight = RepeatVector(embedding_dim)(weight_input)
        weight = Permute((2,1))(weight)

        weighted_emb = merge([scaled_emb, weight], mode='mul')
        conv = self.conv_Layer(maxlen, embedding_dim, nb_filter, filter_length)(weighted_emb)
        clf = self.get_full(drop_out_prob, nb_class)(conv)

        model = Model([sent_input, weight_input], clf)
        self.compile(model)
        return model

class ConjWeightTwoVec_CNN(Kim_CNN):
    def __call__(self,
                 vocabulary_size=5000,
                 maxlen=100,
                 embedding_dim=300,
                 embedding_weights=None,
                 nb_filter=100,
                 filter_length=[3,4,5],
                 nb_class=2,
                 drop_out_prob=0.,
                 l1 = 0.,
                 l2 = 0.):

        self.log_params(locals())

        sent_input = Input(shape=(maxlen,), dtype='int32')
        emb = self.get_emb_layer(vocabulary_size, embedding_dim, maxlen, embedding_weights)(sent_input)

        weight_input = Input((maxlen,)) 
        weight = RepeatVector(embedding_dim)(weight_input)
        weight = Permute((2,1))(weight)

        neg_input = Input((maxlen,), dtype='int32')
        neg_init = np.ones((2, embedding_dim))
        neg_init[1] = -1
        neg_weight = Embedding(2, embedding_dim, input_length=maxlen, weights=[neg_init], W_regularizer=l1l2(l1,l2))(neg_input)

        weighted_emb = merge([emb, weight, neg_weight], mode='mul')
        conv = self.conv_Layer(maxlen, embedding_dim, nb_filter, filter_length)(weighted_emb)
        clf = self.get_full(drop_out_prob, nb_class)(conv)

        model = Model([sent_input, weight_input, neg_input], clf)
        self.compile(model)
        return model
