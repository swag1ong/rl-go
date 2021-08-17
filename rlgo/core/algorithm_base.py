from abc import ABC, abstractmethod


class AlgorithmBase(ABC):
    """algorithm base class"""

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def learn(self, *args, **kwargs):
        pass

    def sample(self, *args, **kwargs):
        pass
