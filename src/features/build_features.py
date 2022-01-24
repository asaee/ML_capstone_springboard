import logging

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

from sklearn.model_selection import train_test_split


def label_encoder(array):
    enc = LabelEncoder()
    array = enc.fit_transform(array)
    n_classes = len(enc.classes_[0])
    return array, n_classes


def ordinal_encoder(array):
    enc = OrdinalEncoder()
    array = enc.fit_transform(array.reshape(-1, 1))
    n_classes = len(enc.categories_[0])
    return array, n_classes


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
