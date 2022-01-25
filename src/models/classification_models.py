import logging

import numpy as np
import pandas as pd

from pprint import pprint
import dill as pickle

# Model selection
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Model optimization
from skopt import BayesSearchCV


# Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# performance metrics
from sklearn.metrics import classification_report

from src.visualization.plot_metrics import ROCPlot, PrecisionRecallPlot


def bayes_search_fit(search, X_train, y_train):
    logging.info("Performing grid search...")
    search.fit(X_train, y_train)
    logging.info("Best score: %0.3f" % search.best_score_)
    print("Best parameters set:")
    best_parameters = search.best_estimator_.get_params()

    param_grid = search.get_params()['search_spaces']
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return search


def bayes_search_eval(search, X_test, y_test):
    y_pred = search.predict(X_test)
    y_pred_prob = search.predict_proba(X_test)

    logging.info(classification_report(y_test, y_pred))

    roc = ROCPlot(y_test, y_pred_prob)
    roc.roc_calc()
    roc.roc_plot()
    pr = PrecisionRecallPlot(y_test, y_pred_prob)
    pr.precision_recall_calc()
    pr.precision_recall_plot()

    return


def config_naive_bayes():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])

    param_grid = {
        'clf__alpha': (1e-4, 1.0, "log-uniform"),
        'vect__max_features': [10000, 30000, None],
        'vect__max_df': (0.5, 0.75, 1.0),
        'tfidf__use_idf': (True, False),
    }

    search = BayesSearchCV(estimator=pipeline, search_spaces=param_grid,
                           n_iter=32, n_jobs=5, verbose=1, random_state=21)

    logging.info("pipeline:", [name for name, _ in pipeline.steps])
    logging.info("parameters:")
    logging.info(param_grid)

    return search


def config_logistic_regression():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(solver='liblinear', n_jobs=5))
    ])

    param_grid = {
        'vect__max_features': [10000, 30000, None],
        'vect__max_df': (0.5, 0.75, 1.0),
        'tfidf__use_idf': (True, False),
        'clf__dual': (True, False),
        'clf__max_iter': [100, 110, 120, 130, 140],
        'clf__C': (1e-5, 1e2, "log-uniform"),
    }

    search = BayesSearchCV(estimator=pipeline, search_spaces=param_grid,
                           n_iter=32, n_jobs=5, verbose=1, random_state=21)

    logging.info("pipeline:", [name for name, _ in pipeline.steps])
    logging.info("parameters:")
    logging.info(param_grid)

    return search
