from abc import abstractmethod
from typing import Dict, List
from numpy.typing import ArrayLike


class CoreMetric:
    def __init__(self, *args: List, **kwargs: Dict) -> None:
        pass

    def __call__(self, *args: List, **kwargs: Dict) -> ArrayLike:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> ArrayLike:
        raise NotImplementedError()

    @abstractmethod
    def backward(self, *args, **kwargs) -> ArrayLike:
        raise NotImplementedError()
