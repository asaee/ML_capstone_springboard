from concurrent.futures import process
import logging
import pandas as pd
from src.features.build_features import data_preparation
from classification_models import config_naive_bayes, config_logistic_regression, bayes_search_fit, bayes_search_eval


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
