from typing import Dict, List, Optional

import numpy as np
from numpy.typing import ArrayLike
from typing_extensions import override  # type: ignore
from .core import CoreModel


class LinearRegression(CoreModel):
    _input: ArrayLike
    _output: ArrayLike
    _gtruth: ArrayLike

    def __init__(self, num_features: int = 2, *args: List, **kwargs: Dict) -> None:
        super().__init__(*args, **kwargs)
        self.num_features = num_features
        self.load_model(checkpoint=kwargs.get("checkpoint", None))  # type: ignore
        self.__dict__.update(kwargs)

    def load_model(
        self, checkpoint: Optional[str] = None, *args: List, **kwargs: Dict
    ) -> None:
        if checkpoint is not None:
            raise NotImplementedError(
                "Loading model from checkpoint is not implemented"
            )

        self.weights = np.random.random((self.num_features, 1))
        self.bias = np.random.random()

    @override
    def forward(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        (shape1, shape2), self._input = x.shape, x
        if shape2 != self.num_features and shape1 == self.num_features:
            x = np.transpose(x)

        assert (
            x.shape[1] == self.num_features
        ), f"Number of features {x.shape[1]} does not match expected number of features {self.num_features}"
        self._output = np.dot(x, self.weights) + self.bias
        return self._output

    def backward(self, *args, **kwargs) -> ArrayLike:
        lr = kwargs.get("learning_rate", 0.01)
        m = len(self._output)
        dw = (2 / m) * self._input.T @ (self._output - self._gtruth)
        db = (2 / m) * np.sum(self._output - self._gtruth)

        self.weights -= lr * dw
        self.bias -= lr * db

    @override
    def fit(self, X: ArrayLike, y: ArrayLike, *args, **kwargs) -> None:
        if kwargs.get("method", "normal") == "normal":
            self._analytical_solution(X, y)
        else:
            raise ValueError("Method must be either 'normal' or 'gradient'")

    def _analytical_solution(self, X: ArrayLike, y: ArrayLike) -> ArrayLike:
        """
        Analytical solution to the linear regression problem.
        Solves the Normal Equation:
        theta = (X^T X)^-1 X^T y
        where:
            X is the input matrix
            y is the target vector
            theta is the parameter vector

        Considerations:
        - No iteration needed, exact solution assuming that X^T X is invertible.
        - Computationally expensive for large datasets (O(n^3) complexity)
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.bias = theta[0]
        self.weights = theta[1:].reshape(-1, 1)
