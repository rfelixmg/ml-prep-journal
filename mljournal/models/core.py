from abc import abstractmethod
from typing import Dict, List, Optional

from numpy.typing import ArrayLike


class CoreModel:
    def __init__(self, *args: List, **kwargs: Dict) -> None:
        pass

    def __call__(self, *args: List, **kwargs: Dict) -> ArrayLike:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> ArrayLike:
        raise NotImplementedError()

    def load_model(
        self, checkpoint: Optional[str] = None, *args: List, **kwargs: Dict
    ) -> None:
        """Base implementation of model loading. Override in subclasses if needed."""
        if checkpoint is not None:
            raise NotImplementedError(
                "Checkpoint loading not implemented in base class"
            )
