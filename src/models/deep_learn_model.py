import logging
import pandas as pd
import numpy as np
from official.nlp import optimization

import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from keras.models import Sequential, Model


def cnn_model_config(maxlen, vocab_size):
    embedding_dim = 50

    text_input = layers.Input(shape=(maxlen,), name='docs')

    text_embeddings = layers.Embedding(input_dim=vocab_size,
                                       output_dim=embedding_dim,
                                       input_length=maxlen)(text_input)

    conv_1 = layers.Conv1D(embedding_dim, 2, strides=2,
                           name='cnn_2')(text_embeddings)
    conv_2 = layers.Conv1D(embedding_dim, 3, strides=3,
                           name='cnn_3')(text_embeddings)
    conv_3 = layers.Conv1D(embedding_dim, 4, strides=4,
                           name='cnn_4')(text_embeddings)
    conv_4 = layers.Conv1D(embedding_dim, 5, strides=5,
                           name='cnn_5')(text_embeddings)

    pool_1 = layers.AveragePooling1D(pool_size=10, name='pooling_1')(conv_1)
    pool_2 = layers.AveragePooling1D(pool_size=10, name='pooling_2')(conv_2)
    pool_3 = layers.AveragePooling1D(pool_size=10, name='pooling_3')(conv_3)
    pool_4 = layers.AveragePooling1D(pool_size=10, name='pooling_4')(conv_4)

    hybrid_layer = layers.Concatenate(axis=1, name='hybrid_layer')([
        pool_1, pool_2, pool_3, pool_4])

    flatten = layers.Flatten(name='faltten')(hybrid_layer)

    dense = layers.Dense(512, activation='relu', name='Dense')(flatten)
    dropout = layers.Dropout(0.5)(dense)

    dense = layers.Dense(64, activation='relu', name='Dense_2')(dropout)
    dropout = layers.Dropout(0.5)(dense)

    output = layers.Dense(5, activation='softmax', name='classifier')(dropout)

    model_CNN = Model(inputs=text_input, outputs=output)

    return model_CNN


def cnn_model_builder(num_train_steps, vocab_size, maxlen):
    init_lr = 7e-3
    num_warmup_steps = int(0.1*num_train_steps)
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    cnn_model = cnn_model_config(maxlen, vocab_size)
    cnn_model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    return cnn_model


def lstm_model_builder(vocab_size, maxlen):
    embedding_dim = 16
    model_lstm = Sequential()
    model_lstm.add(layers.Embedding(input_dim=vocab_size,
                                    output_dim=embedding_dim,
                                    input_length=maxlen))
    model_lstm.add(layers.Conv1D(16, 2, strides=2))
    model_lstm.add(layers.AveragePooling1D(pool_size=3))
    model_lstm.add(layers.Bidirectional(layers.LSTM(16)))
    model_lstm.add(layers.Dropout(rate=0.5, seed=21))
    model_lstm.add(layers.Dense(256, activation='relu'))
    model_lstm.add(layers.Dense(5, activation='softmax'))
    model_lstm.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    return model_lstm
