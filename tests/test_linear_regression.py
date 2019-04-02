import pytest
import numpy as np
from sklearn.datasets import make_classification
from models.linear_regression import BinaryLinearRegression


def test_binary_linear_regression_zero_check():
    assert(1==1)