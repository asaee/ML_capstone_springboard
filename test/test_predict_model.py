import pytest
import os
import pandas as pd
import pickle5 as pickle
import keras
from keras.models import load_model

from src.models.predict_model import *


@pytest.fixture
def get_DL_data():
    sample_data_DL = os.path.join(os.path.dirname(
        __file__), "data/sample_for_deeplearning.csv")
    corpus = pd.read_csv(sample_data_DL, usecols=['content', 'topic_area'])

    return corpus


@pytest.fixture
def get_tokenizer():
    path_tokenizer = os.path.join(os.path.dirname(
        __file__), "../src/lib/tokenizer.pickle")
    with open(path_tokenizer, 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    assert isinstance(loaded_tokenizer, keras.preprocessing.text.Tokenizer)

    return loaded_tokenizer


@pytest.fixture
def get_model():
    path_model = os.path.join(os.path.dirname(
        __file__), "../src/lib/cnn_model")

    loaded_model = load_model(path_model, compile=False)
    assert isinstance(loaded_model, keras.engine.functional.Functional)

    return loaded_model


def test_process_input_text(get_DL_data, get_tokenizer):
    processed_corpus = process_input_text(
        get_DL_data['content'].values, get_tokenizer)

    assert isinstance(processed_corpus, np.ndarray)
    assert processed_corpus.shape == (get_DL_data.shape[0], 700)


def test_predict_label_prob(get_DL_data, get_tokenizer, get_model):
    processed_corpus = process_input_text(
        get_DL_data['content'].values, get_tokenizer)

    label_pred_prob = predict_label_prob(
        processed_corpus, get_model)

    assert isinstance(label_pred_prob, np.ndarray)
    assert label_pred_prob.shape == (get_DL_data.shape[0], 5)
    assert label_pred_prob.max() <= 1.0
    assert label_pred_prob.min() >= 0.0
