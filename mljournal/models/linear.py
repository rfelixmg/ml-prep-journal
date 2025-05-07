import numpy as np

from typing import List, Dict, Optional
from numpy.typing import ArrayLike

from .core import CoreModel
from ..metrics.losses import MeanSquareError

class LinearRegression(CoreModel):
    def __init__(self, num_features: int = 2, *args: List, **kwargs: Dict) -> None:
        self.num_features = num_features
        self.load_model(kwargs.get('checkpoint', None))
        self.__dict__.update(kwargs)

    def load_model(self, checkpoint: Optional[str] = None, *args: List, **kwargs: Dict) -> None:
        if checkpoint is not None:
            raise NotImplemented()

        self.weights = np.random.random((self.num_features, 1))
        self.bias = np.random.random()

    def forward(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        (shape1, shape2), self._input = x.shape, x
        if shape2 != self.num_features and shape1 == self.num_features:
            x = np.transpose(x)
        
        assert x.shape[1] == self.num_features, f"Number of features {x.shape[1]} does not match expected number of features {self.num_features}"
        return np.dot(x, self.weights) + self.bias

    def backward(self, *args, **kwargs) -> ArrayLike:
        raise NotImplemented()