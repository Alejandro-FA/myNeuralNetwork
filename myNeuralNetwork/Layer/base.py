from abc import ABC, abstractmethod
from myNeuralNetwork.Activation.base import BaseActivation


class BaseLayer(ABC):
    def __init__(self, weights, biases, activation: BaseActivation):
        """
        Initializes a new instance of a Layer class.

        Args:
            weights: The weights of the neurons of the layer.
            biases: The biases of the neurons of the layer.
            activation: The activation function of the layer.
        """
        self.activation = activation
        self.W = weights
        self.b = biases

    @abstractmethod
    def forward_propagation(self, input_data, debug=False):
        pass

    @abstractmethod
    def backward_propagation(self, dA, debug=False):
        pass

    @abstractmethod
    def update_weights(self, learning_rate, debug=False):
        pass