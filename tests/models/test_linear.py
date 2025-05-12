import numpy as np
import pytest
from mljournal.models.linear import LinearRegression


@pytest.fixture
def linear_model():
    return LinearRegression()


def test_normal_equation_solution(linear_model):
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 3

    # Generate random features
    X = np.random.randn(n_samples, n_features)
    # Generate true weights
    true_weights = np.array([2.0, -1.0, 0.5])
    # Generate target values with some noise
    y = X @ true_weights + np.random.randn(n_samples) * 0.1

    # Fit using normal equation
    linear_model.fit(X, y, method="normal")

    # Get predicted weights
    predicted_weights = linear_model.weights

    # Assert
    assert np.allclose(predicted_weights, true_weights, rtol=0.1)
    assert predicted_weights.shape == (n_features,)


def test_normal_equation_with_bias(linear_model):
    # Generate synthetic data with bias term
    np.random.seed(42)
    n_samples, n_features = 100, 3

    # Generate random features
    X = np.random.randn(n_samples, n_features)
    # Generate true weights including bias
    true_weights = np.array([2.0, -1.0, 0.5])
    true_bias = 1.0
    # Generate target values with some noise
    y = X @ true_weights + true_bias + np.random.randn(n_samples) * 0.1

    # Fit using normal equation
    linear_model.fit(X, y, method="normal", bias=True)

    # Get predicted weights and bias
    predicted_weights = linear_model.weights
    predicted_bias = linear_model.bias

    # Assert
    assert np.allclose(predicted_weights, true_weights, rtol=0.1)
    assert np.isclose(predicted_bias, true_bias, rtol=0.1)
    assert predicted_weights.shape == (n_features,)


def test_normal_equation_singular_matrix(linear_model):
    # Generate data that will create a singular matrix
    np.random.seed(42)
    n_samples, n_features = 10, 3

    # Create linearly dependent features
    X = np.random.randn(n_samples, n_features)
    X[:, 2] = X[:, 0] + X[:, 1]  # Make third column linearly dependent
    y = np.random.randn(n_samples)

    # Test that it raises appropriate error
    with pytest.raises(np.linalg.LinAlgError):
        linear_model.fit(X, y, method="normal")


def test_normal_equation_prediction(linear_model):
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 3

    # Generate random features
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([2.0, -1.0, 0.5])
    y = X @ true_weights + np.random.randn(n_samples) * 0.1

    # Fit using normal equation
    linear_model.fit(X, y, method="normal")

    # Generate new test data
    X_test = np.random.randn(10, n_features)
    y_test = X_test @ true_weights

    # Make predictions
    y_pred = linear_model.predict(X_test)

    # Assert
    assert np.allclose(y_pred, y_test, rtol=0.1)
    assert y_pred.shape == (10,)


def test_gradient_descent_forward(linear_model):
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 3

    # Generate random features and weights
    X = np.random.randn(n_samples, n_features)
    weights = np.random.randn(n_features)
    bias = 1.0

    # Set model parameters
    linear_model.weights = weights
    linear_model.bias = bias

    # Make predictions
    y_pred = linear_model.forward(X)

    # Calculate expected predictions
    y_expected = X @ weights + bias

    # Assert
    assert np.allclose(y_pred, y_expected)
    assert y_pred.shape == (n_samples,)


def test_gradient_descent_backward(linear_model):
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 3

    # Generate random features, weights, and targets
    X = np.random.randn(n_samples, n_features)
    weights = np.random.randn(n_features)
    bias = 1.0
    y_true = np.random.randn(n_samples)

    # Set model parameters
    linear_model.weights = weights
    linear_model.bias = bias

    # Forward pass
    y_pred = linear_model.forward(X)

    # Calculate gradients
    grad_weights, grad_bias = linear_model.backward(X, y_true, y_pred)

    # Calculate expected gradients manually
    error = y_pred - y_true
    expected_grad_weights = X.T @ error / n_samples
    expected_grad_bias = np.mean(error)

    # Assert
    assert np.allclose(grad_weights, expected_grad_weights)
    assert np.isclose(grad_bias, expected_grad_bias)
    assert grad_weights.shape == (n_features,)


def test_gradient_descent_fit(linear_model):
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 3

    # Generate random features
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([2.0, -1.0, 0.5])
    true_bias = 1.0
    y = X @ true_weights + true_bias + np.random.randn(n_samples) * 0.1

    # Fit using gradient descent
    linear_model.fit(X, y, method="gradient", learning_rate=0.01, n_iterations=1000)

    # Get predicted weights and bias
    predicted_weights = linear_model.weights
    predicted_bias = linear_model.bias

    # Assert
    assert np.allclose(predicted_weights, true_weights, rtol=0.1)
    assert np.isclose(predicted_bias, true_bias, rtol=0.1)
    assert predicted_weights.shape == (n_features,)


def test_gradient_descent_convergence(linear_model):
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 3

    # Generate random features
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([2.0, -1.0, 0.5])
    true_bias = 1.0
    y = X @ true_weights + true_bias + np.random.randn(n_samples) * 0.1

    # Track loss during training
    losses = []

    def loss_callback(loss):
        losses.append(loss)

    # Fit using gradient descent with callback
    linear_model.fit(
        X,
        y,
        method="gradient",
        learning_rate=0.01,
        n_iterations=1000,
        callback=loss_callback,
    )

    # Assert that loss decreases over time
    assert losses[-1] < losses[0]
    # Assert that loss stabilizes (no large jumps at the end)
    assert np.std(losses[-100:]) < np.std(losses[:100])
