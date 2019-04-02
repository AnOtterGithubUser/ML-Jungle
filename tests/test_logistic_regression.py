import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from models.logistic_regression import BinaryLogisticRegression


def test_binary_logistic_regression_should_score_more_than_90_percent_accuracy_on_dummy_training_set_with_20_features():
    # Given
    X, y = make_classification(random_state=42)
    model = BinaryLogisticRegression()

    # When
    model.fit(X, y)
    y_pred = model.predict(X)
    model_accuracy_score = accuracy_score(y, y_pred)

    # Then
    assert(model_accuracy_score > 0.9)


def test_binary_logistic_regression_should_score_more_than_90_percent_accuracy_on_dummy_training_set_with_1_feature():
    # Given
    X, y = make_classification(n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=42)
    model = BinaryLogisticRegression()

    # When
    model.fit(X, y)
    y_pred = model.predict(X)
    model_accuracy_score = accuracy_score(y, y_pred)

    # Then
    assert(model_accuracy_score > 0.9)


def test_binary_logistic_regression_should_raise_ValueError_when_given_different_number_of_training_samples_and_labels():
    # Given
    X, y = make_classification(n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=42)
    X = X[:10]
    model = BinaryLogisticRegression()

    # When
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_binary_logistic_regression_should_raise_ValueError_when_given_empty_training_set():
    # Given
    X, y = np.array([]), np.array([])
    model = BinaryLogisticRegression()

    # When
    with pytest.raises(ValueError):
        model.fit(X, y)




