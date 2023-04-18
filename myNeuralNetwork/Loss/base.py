from abc import ABC, abstractmethod


class BaseLoss(ABC):
    @abstractmethod
    def error(self, A, y):
        pass

    @abstractmethod
    def error_derivative(self, A, y):
        pass
