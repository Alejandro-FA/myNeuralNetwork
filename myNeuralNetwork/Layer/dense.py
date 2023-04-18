import tensorflow as tf
from myNeuralNetwork.Activation.base import BaseActivation
from myNeuralNetwork.Layer.base import BaseLayer

class Dense(BaseLayer):
    """
    Class to represent Neural Networks layers and use gradient descent to update
    parameters.

    Attributes:
        W: A matrix representation of the weights of the layer.
        b: A vector representation of the biases of the layer.
    """
    def __init__(self, weights, biases, activation: BaseActivation):
        super().__init__(weights, biases, activation)
        self.A_prev = None
        self.Z = None
        self.dW = None
        self.db = None


    def forward_propagation(self, input_data, debug=False):
        self.A_prev = input_data # A[l-1]
        self.Z = tf.add( tf.matmul(self.W, input_data), self.b )
        A = self.activation.value_at(self.Z)

        if debug: 
            print(f"Forward propagation output:\n{A}\n")
        return A


    def backward_propagation(self, dA, debug=False):
        dg = self.activation.derivative_at(self.Z)
        dZ = tf.multiply(dA, dg)
        self.dW = tf.matmul(dZ, self.A_prev, transpose_b=True)
        self.db = dZ
        dA_prev = tf.matmul(self.W, dZ, transpose_a=True) # dL / dA[l-1] 

        if debug:
            print(f"dZ[l]:\n{dZ}")
            print(f"dW:\n{self.dW}")
            print(f"db:\n{self.db}")
            print(f"dA[l-1]:\n{dA_prev}\n")
        return dA_prev


    def update_weights(self, learning_rate, debug=False):
        self.W = tf.subtract( self.W, tf.multiply(learning_rate, self.dW) )
        self.b = tf.subtract( self.b, tf.multiply(learning_rate, self.db) )

        if debug:
            print(f"Updated weights:\n{self.W}")
            print(f"Updated bias:\n{self.b}\n")