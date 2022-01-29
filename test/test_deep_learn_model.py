import pytest
import logging
from src.models.deep_learn_model import *


def test_cnn_model_builder():
    num_train_steps = 2000
    vocab_size = 20000
    maxlen = 700
    model = cnn_model_builder(num_train_steps, vocab_size, maxlen)

    assert isinstance(model, keras.engine.functional.Functional)
    assert len(model.layers) == 17
    assert model.loss == 'sparse_categorical_crossentropy'


def test_lstm_model_builder():
    vocab_size = 20000
    maxlen = 700
    model = lstm_model_builder(vocab_size, maxlen)

    assert isinstance(model, keras.engine.functional.Functional)
    assert len(model.layers) == 7
    assert model.loss == 'sparse_categorical_crossentropy'
