import pytest

from src.models.classification_models import *


def test_config_naive_bayes():
    model = config_naive_bayes()

    assert model.estimator.steps[0][0] == 'vect'
    assert model.estimator.steps[1][0] == 'tfidf'
    assert model.estimator.steps[2][0] == 'clf'

    assert [x for x in model.search_spaces.keys()] == ['clf__alpha',
                                                       'vect__max_features', 'vect__max_df', 'tfidf__use_idf']


def test_config_logistic_regression():
    model = config_logistic_regression()

    assert model.estimator.steps[0][0] == 'vect'
    assert model.estimator.steps[1][0] == 'tfidf'
    assert model.estimator.steps[2][0] == 'clf'

    assert [x for x in model.search_spaces.keys()] == ['vect__max_features',
                                                       'vect__max_df',
                                                       'tfidf__use_idf',
                                                       'clf__dual',
                                                       'clf__max_iter',
                                                       'clf__C']
