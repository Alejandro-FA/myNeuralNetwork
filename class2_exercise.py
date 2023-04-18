import tensorflow as tf
import myNeuralNetwork as NN


X = tf.constant([2, -1, 2], shape=[3,1], dtype=tf.double)
y = tf.constant(1, dtype=tf.double)
W_1 = tf.constant([[1, 2, 0], [0, 3, 0], [4, 4, 0], [0, 0, -2]], dtype=tf.double)
b_1 = tf.constant([2, 1, 1, 3], shape=[4,1], dtype=tf.double)
W_2 = tf.constant([3, 2, 1, 1], shape=[1, 4], dtype=tf.double)
b_2 = tf.constant(-11, dtype=tf.double)

l1 = NN.Layer.Dense(weights=W_1, biases=b_1, activation=NN.ReLU())
l2 = NN.Layer.Dense(weights=W_2, biases=b_2, activation=NN.Sigmoid())

network = NN.NeuralNetwork(X, y)
network.add_layer(l1)
network.add_layer(l2)


output = network.forward_propagation(debug=True)
error = NN.CrossEntropy().error_derivative(output, y)
network.backward_propagation(error, debug=True)
network.update_weights(learning_rate=0.01, debug=True)

print("\nFinished")
