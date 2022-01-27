import logging
import pickle

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def process_input_text(corpus, maxlen=700):
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    corpus = tokenizer.texts_to_sequences(corpus)
    corpus = pad_sequences(corpus, padding='post', maxlen=maxlen)

    return corpus
