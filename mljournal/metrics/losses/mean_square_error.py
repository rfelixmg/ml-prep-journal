import numpy as np
from numpy.typing import ArrayLike
from typing_extensions import override  # type: ignore
from ..core import CoreMetric


class MeanSquareError(CoreMetric):
    namespace = "mean_square_error"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__.update(kwargs)

    @override
    def forward(self, y_true: ArrayLike, y_pred: ArrayLike, *args, **kwargs) -> float:
        self._y_true, self._y_pred = y_true, y_pred
        return 1 / len(y_true) * np.sum(np.square(y_true - y_pred), axis=-1)

    def backward(self, *args, **kwargs) -> ArrayLike:
        return 2 * (self._y_pred - self._y_true) / len(self._y_true)
