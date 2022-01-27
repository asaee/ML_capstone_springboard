import logging
import pickle5 as pickle
import numpy as np

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

logging.basicConfig(level=logging.INFO)


def process_input_text(corpus, path_tokenizer="./lib/tokenizer.pickle", maxlen=700):
    with open(path_tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)

    corpus = tokenizer.texts_to_sequences(corpus)
    corpus = pad_sequences(corpus, padding='post', maxlen=maxlen)

    return corpus


def predict_label_prob(corpus, path_model="./lib/cnn_model"):
    model_cnn = load_model(path_model, compile=False)
    label_pred_prob = model_cnn.predict(corpus)
    return label_pred_prob


def predict_label(label_pred_prob):
    lablel_map = {0: 'business', 1: 'finance',
                  2: 'general', 3: 'science', 4: 'tech'}

    label_pred = np.vectorize(lablel_map.get)(
        np.argmax(label_pred_prob, axis=1))

    return label_pred
