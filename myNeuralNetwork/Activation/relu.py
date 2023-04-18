import tensorflow as tf
from myNeuralNetwork.Activation.base import BaseActivation

class ReLU(BaseActivation):
    """
    This class implements the value_at and derivative_at methods for
    a ReLU activation function.
    """
    def value_at(self, z):
        return tf.maximum(0, z)

    def derivative_at(self, z):
        g = tf.maximum(0, z)
        return tf.sign(g)
