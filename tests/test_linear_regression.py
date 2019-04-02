import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from models.linear_regression import BinaryLinearRegression


def test_binary_linear_regression_should_have_rmse_equal_to_zero_on_dummy_training_set_with_two_dimensions():
    # Given
    X, y = make_regression(random_state=42)
    model = BinaryLinearRegression()

    # When
    model.fit(X, y)
    y_pred = model.predict_proba(X)
    model_rmse = mean_squared_error(y, y_pred)

    # Then
    assert (model_rmse < 1e-20)


def test_binary_linear_regression_without_bias_should_have_rmse_equal_to_zero_on_dummy_training_set_with_two_dimensions():
    # Given
    X, y = make_regression(random_state=42)
    model = BinaryLinearRegression(with_bias=False)

    # When
    model.fit(X, y)
    y_pred = model.predict_proba(X)
    model_rmse = mean_squared_error(y, y_pred)

    # Then
    assert (model_rmse < 1e-19)


def test_binary_linear_regression_should_have_rmse_equal_to_zero_on_dummy_training_set_with_one_dimension():
    # Given
    X, y = make_regression(n_features=1, random_state=42)
    model = BinaryLinearRegression()

    # When
    model.fit(X, y)
    y_pred = model.predict_proba(X)
    model_rmse = mean_squared_error(y, y_pred)

    # Then
    assert (model_rmse < 1e-20)


def test_binary_linear_regression_without_bias_should_have_rmse_equal_to_zero_on_dummy_training_set_with_one_dimension():
    # Given
    X, y = make_regression(n_features=1, random_state=42)
    model = BinaryLinearRegression(with_bias=False)

    # When
    model.fit(X, y)
    y_pred = model.predict_proba(X)
    model_rmse = mean_squared_error(y, y_pred)

    # Then
    assert (model_rmse < 1e-20)


def test_binary_linear_regression_fit_should_raise_warning_when_the_graam_matrix_is_singular():
    # Given
    X, y = make_regression(random_state=42)
    model = BinaryLinearRegression()

    # When
    with pytest.warns(UserWarning):
        model.fit(X, y)


def test_binary_linear_regression_fit_should_raise_ValueError_when_given_different_number_of_samples_and_labels():
    # Given
    X, y = make_regression(random_state=42)
    X = X[:10]
    model = BinaryLinearRegression()

    # When
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_binary_linear_regression_fit_should_raise_ValueError_when_given_empty_set():
    # Given
    X, y = np.array([]), np.array([])
    model = BinaryLinearRegression()

    # When
    with pytest.raises(ValueError):
        model.fit(X, y)



