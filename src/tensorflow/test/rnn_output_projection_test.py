import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

n_steps = 2
n_inputs = 3
n_neurons = 3
n_outputs = 5

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# X_seqs is a Python list of n_steps tensors of shape [None, n_inputs]

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)

wrapperd_cell = tf.contrib.rnn.OutputProjectionWrapper(
    cell=basic_cell,
    output_size=n_outputs
)

output_seqs, states = tf.nn.dynamic_rnn(
    cell=wrapperd_cell,
    inputs=X,
    dtype=tf.float32
)

init = tf.global_variables_initializer()

X_batch = np.array([
    # t=0         t=1
    [[0, 1, 2], [9, 8, 7]],  # instance 0
    [[3, 4, 5], [0, 0, 0]],  # instance 1
    [[6, 7, 8], [6, 5, 4]],  # instance 2
    [[9, 0, 1], [3, 2, 1]],  # instance 3
])

with tf.Session() as sess:
    sess.run(init)
    result_seq, result_states = sess.run([output_seqs, states], feed_dict={X: X_batch})
    print(result_seq, result_seq.shape)

