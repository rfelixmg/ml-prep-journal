import numpy as np
import pytest
from mljournal.metrics.losses.mean_square_error import MeanSquareError


@pytest.fixture
def mse():
    return MeanSquareError()


def test_forward(mse):
    # Test data
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    # Calculate expected MSE manually
    expected_mse = np.mean((y_true - y_pred) ** 2)

    # Calculate MSE using the class
    actual_mse = mse.forward(y_true, y_pred)

    # Assert
    assert np.isclose(actual_mse, expected_mse)
    assert isinstance(actual_mse, float)


def test_backward(mse):
    # Test data
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    # First call forward to set internal state
    mse.forward(y_true, y_pred)

    # Calculate expected gradient manually
    expected_grad = 2 * (y_pred - y_true) / len(y_true)

    # Calculate gradient using the class
    actual_grad = mse.backward()

    # Assert
    assert np.allclose(actual_grad, expected_grad)
    assert isinstance(actual_grad, np.ndarray)


def test_backward_with_different_shapes(mse):
    # Test data
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    # First call forward to set internal state
    mse.forward(y_true, y_pred)

    # Calculate expected gradient manually
    expected_grad = 2 * (y_pred - y_true) / len(y_true)

    # Calculate gradient using the class
    actual_grad = mse.backward()

    # Assert
    assert np.allclose(actual_grad, expected_grad)
    assert isinstance(actual_grad, np.ndarray)


def test_forward_with_different_shapes(mse):
    # Test data
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    # Calculate expected MSE manually
    expected_mse = np.mean((y_true - y_pred) ** 2, axis=-1)

    # Calculate MSE using the class
    actual_mse = mse(y_true, y_pred)

    # Assert
    assert np.isclose(actual_mse, expected_mse)
    assert isinstance(actual_mse, float)
