from abc import ABC, abstractmethod


class BaseActivation(ABC):
    """
    Base abstract class for defining activation functions.
    It has been designed to be inherited from.
    """
    @abstractmethod
    def value_at(self, z):
        """
        Evaluates the activation function at the points specified by z.
        """

    @abstractmethod
    def derivative_at(self, z):
        """
        Evaluates the derivative of the activation function at the points specified by z.
        """
