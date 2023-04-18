"""
Abstract Base Class (ABC) for implementing an loss function.
Implementing classes should inherit from `BaseLoss
"""

from abc import ABC, abstractmethod


class BaseLoss(ABC):
    @abstractmethod
    def error(self, A, y):
        """
        Evaluates the loss function.

        Args:
            A: The obtained output of the network
            y: The predicted output
        """

    @abstractmethod
    def error_derivative(self, A, y):
        """
        Evaluates the derivative of the loss function.

        Args:
            A: The obtained output of the network
            y: The predicted output
        """
