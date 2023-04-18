import tensorflow as tf
from myNeuralNetwork.Activation.base import BaseActivation

class Sigmoid(BaseActivation):
    """
    This class implements the value_at and derivative_at methods for
    a sigmoid activation function.
    """
    def value_at(self, z):
        return tf.sigmoid(z)

    def derivative_at(self, z):
        g = tf.sigmoid(z)
        return tf.multiply(g, tf.subtract(1, g))
