import tensorflow as tf
from myNeuralNetwork.Loss.base import BaseLoss


class L2(BaseLoss):
    def error(self, A, y):
        diff = tf.subtract(A, y)
        return tf.multiply(diff, diff)

    def error_derivative(self, A, y):
        diff = tf.subtract(A, y)
        return tf.multiply(2, diff)
        