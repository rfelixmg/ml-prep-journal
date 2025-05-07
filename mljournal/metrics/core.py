import numpy as no

from typing import List, Dict
from numpy.typing import ArrayLike

from abc import abstractmethod

class CoreMetric:
    def __init__(self, *args: List, **kwargs: Dict) -> None:
        pass

    def __call__(self, *args: List, **kwargs: Dict) -> ArrayLike:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> ArrayLike:
        raise NotImplemented()
    
    @abstractmethod
    def backward(self, *args, **kwargs) -> ArrayLike:
        raise NotImplemented()
