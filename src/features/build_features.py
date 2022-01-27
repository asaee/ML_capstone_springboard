import logging
from optparse import Values
import pickle

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def label_encoder(array):
    enc = LabelEncoder()
    array = enc.fit_transform(array)
    encoded_label_map = dict(zip(enc.transform(enc.classes_), enc.classes_))
    return array, encoded_label_map


def onehot_encoder(array):
    enc = OneHotEncoder()
    array = enc.fit_transform(array.reshape(-1, 1)).toarray()
    n_classes = len(enc.categories_[0])
    return array, n_classes, enc.categories_[0]


def merge_tags(df):
    tag_map = {'consumer': 'general',
               'healthcare': 'science',
               'automotive': 'business',
               'environment': 'science',
               'construction': 'business',
               'ai': 'tech'}

    df['tags'] = [(lambda tags: tag_map[tags] if tags in tag_map.keys() else tags)(tags)
                  for tags in df['topic_area']]
    return df


def data_preparation(df, test_size, random_state=None):
    logging.info("Splitting the data-frame into train and test parts")

    df = df[['content', 'topic_area']]

    df = merge_tags(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df['content'].tolist(),
        df['tags'].values,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def tokenize_text_deeplearning(df, maxlen=700):
    logging.info(
        "Splitting the data-frame into train, test, and validation parts")

    df = df[['content', 'topic_area']]

    df = merge_tags(df)

    X = df.content.values
    y, enc_label_map = label_encoder(df['tags'].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=21)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=0.2, random_state=21)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    # saving the Tokenizer
    with open('.\lib\tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    X_validation = tokenizer.texts_to_sequences(X_validation)

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    X_validation = pad_sequences(X_validation, padding='post', maxlen=maxlen)

    return X_train, X_test, X_validation, y_train, y_test, y_validation, vocab_size, enc_label_map
