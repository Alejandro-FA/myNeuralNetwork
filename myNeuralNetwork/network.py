"""
This module includes basic Neural Networks functionalities.
It has been created to practice basic implementations of Neural Networks.
The TensorFlow library is only used for algebraic operations.
"""
from myNeuralNetwork.Layer.base import BaseLayer


class NeuralNetwork:
    """
    Custom class for creating neural networks and using forward_propagation
    and backward_propagation to update the parameters.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.layers = []

    def add_layer(self, layer: BaseLayer):
        self.layers.append(layer)

    def forward_propagation(self, debug=False):
        layer_output = self.X
        for layer in self.layers:
            layer_output = layer.forward_propagation(layer_output, debug)
        return layer_output

    def backward_propagation(self, output_error, debug=False):
        for layer in reversed(self.layers):
            output_error = layer.backward_propagation(output_error, debug)

    def update_weights(self, learning_rate, debug=False):
        for layer in self.layers:
            layer.update_weights(learning_rate, debug)
