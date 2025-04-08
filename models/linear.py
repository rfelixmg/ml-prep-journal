import numpy as np

from typing import List, Dict, Optional
from numpy.typing import ArrayLike


class LinearRegression:
    def __init__(self, num_features: int = 2, *args: List, **kwargs: Dict) -> None:
        self.num_features = num_features
        self.load_model(kwargs.get('checkpoint', None))
        self.__dict__.update(kwargs)

    def __call__(self, *args: List, **kwargs: Dict) -> ArrayLike:
        return self.forward(*args, **kwargs)

    def load_model(self, checkpoint: Optional[str] = None, *args: List, **kwargs: Dict) -> None:
        if checkpoint is not None:
            raise NotImplemented

        self.weights = np.random.random((self.num_features, 1))
        self.bias = np.random.random()


    def forward(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        return x.dot(self.weights) + self.bias
    
    # def backward(self, *args, **kwargs) -> ArrayLike:
        # raise NotImplemented()