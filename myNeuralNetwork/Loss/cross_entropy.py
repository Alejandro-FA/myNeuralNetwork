import tensorflow as tf
from myNeuralNetwork.Loss.base import BaseLoss


class CrossEntropy(BaseLoss):
    def error(self, A, y):
        log_a = tf.math.log(A)
        log_1a = tf.math.log(1 - A)
        first = tf.negative( tf.multiply(y, log_a) )
        second = tf.multiply( tf.subtract(1, y), log_1a )
        return tf.subtract(first, second)

    def error_derivative(self, A, y):
        first = tf.divide( tf.negative(y), A )
        second = tf.divide( tf.subtract(1, y), tf.subtract(1, A) )
        return tf.add(first, second)