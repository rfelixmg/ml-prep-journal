import numpy as np
from ..core import CoreMetric


class MeanSquareError(CoreMetric):
    namespace = 'mean_square_error'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__.update(kwargs)

    def forward(self, y_true, y_pred, *args, **kwargs):
        self._y_true, self._y_pred = y_true, y_pred
        return 1 / len(y_true) * np.sum((y_true - y_pred) ** 2)

    def backward(self, *args, **kwargs):
        return 2 * (self._y_pred - self._y_true) / len(self._y_true)
    
