import pytest
import numpy as np
from utils.kernels import linear_kernel



def test_linear_kernel_should_return_inner_product_between_vectors():
    # Given
    x = np.array([1, 2, 3])
    y = np.array([4, 5 ,6])
    expected_result = 32
    # When
    actual_result = linear_kernel(x, y)
    # Then
    assert(expected_result == actual_result)