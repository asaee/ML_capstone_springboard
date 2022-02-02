from concurrent.futures import process
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from src.features.build_features import data_preparation
from classification_models import config_naive_bayes, config_logistic_regression, bayes_search_fit, bayes_search_eval
from src.features.build_features import tokenize_text_deeplearning
from src.visualization.plot_metrics import plot_neuralnet_history
from deep_learn_model import cnn_model_builder, lstm_model_builder
import tensorflow as tf
from keras.callbacks import EarlyStopping


def train_classifier(filename, clf="nb"):

    logging.info('Starting the data analysis pipeline')
    classifiers = {'nb': config_naive_bayes(
    ), 'lr': config_logistic_regression()}
    processed_data = pd.read_csv(filename, usecols=['content', 'topic_area'])

    X_train, X_test, y_train, y_test = data_preparation(processed_data,
                                                        test_size=0.3,
                                                        random_state=21)

    bs = classifiers[clf]
    bs = bayes_search_fit(bs, X_train, y_train)
    bayes_search_eval(bs, X_test, y_test)
    logging.info('The data analysis pipeline has terminated')

    return


def train_cnn(filename):
    maxlen = 700
    epochs = 20
    batch_size = 500

    corpus = pd.read_csv(filename, usecols=['content', 'topic_area'])
    X_train, X_test, X_validation, y_train, y_test, y_validation, vocab_size, enc_label_map = tokenize_text_deeplearning(
        corpus, maxlen)

    steps_per_epoch = tf.data.experimental.cardinality(
        tf.data.Dataset.from_tensor_slices(X_train)).numpy()
    num_train_steps = steps_per_epoch * epochs

    cnn_model = cnn_model_builder(num_train_steps, vocab_size, maxlen)

    my_callbacks = [EarlyStopping(monitor='val_loss',
                                  patience=4,
                                  restore_best_weights=False),
                    ]

    history_cnn_model = cnn_model.fit(X_train, y_train,
                                      epochs=epochs,
                                      verbose=True,
                                      validation_data=(
                                          X_validation, y_validation),
                                      batch_size=batch_size,
                                      callbacks=[my_callbacks])

    y_pred_prob = cnn_model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    logging.info(classification_report(y_test, y_pred,
                                       target_names=list(enc_label_map.values())))

    plot_neuralnet_history(history_cnn_model)
    logging.info('The deep learning data analysis pipeline has terminated')
    return


def train_lstm(filename):
    maxlen = 700
    epochs = 20
    batch_size = 500

    corpus = pd.read_csv(filename, usecols=['content', 'topic_area'])
    X_train, X_test, X_validation, y_train, y_test, y_validation, vocab_size, enc_label_map = tokenize_text_deeplearning(
        corpus, maxlen)

    lstm_model = lstm_model_builder(vocab_size, maxlen)

    my_callbacks = [EarlyStopping(monitor='val_loss',
                                  patience=4,
                                  restore_best_weights=False),
                    ]

    history_lstm_model = lstm_model.fit(X_train, y_train,
                                        epochs=epochs,
                                        verbose=True,
                                        validation_data=(
                                            X_validation, y_validation),
                                        batch_size=batch_size,
                                        callbacks=[my_callbacks])

    y_pred_prob = lstm_model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    logging.info(classification_report(y_test, y_pred,
                                       target_names=list(enc_label_map.values())))

    plot_neuralnet_history(history_lstm_model)
    logging.info('The deep learning data analysis pipeline has terminated')
    return


def save_keras_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("../lib/model/model_config.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("../lib/model/model_weights.h5")

    return
