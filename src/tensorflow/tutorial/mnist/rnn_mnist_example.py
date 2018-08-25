from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST-data", one_hot=True)


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# Training Parameters
learning_rate = 0.001
training_steps = 5000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28  # MNIST data input (img shape: 28*28)
timesteps = 28  # timesteps
num_hidden = 128  # hidden layer num of features
num_classes = 10  # MNIST total classes (0-9 digits)


def RNN_BLOCK_LSTM(X):
    with tf.variable_scope('RNN'):
        lstm_cell = rnn.LSTMBlockCell(num_hidden, forget_bias=1.0)

        # lstm_cell = rnn.DropoutWrapper(
        #     rnn.LSTMBlockCell(num_hidden, forget_bias=1.0),
        #     input_keep_prob=0.5,
        #     output_keep_prob=0.5,
        #     state_keep_prob=0.5,
        #     dtype=tf.float32
        # )

        outputs, state = tf.nn.dynamic_rnn(
            cell=lstm_cell,
            inputs=X,
            dtype=tf.float32
        )

    batch_norm = tf.layers.batch_normalization(outputs[:, -1, :])

    logits = tf.layers.dense(
        inputs=batch_norm,
        units=num_classes,
        activation=None,
        kernel_initializer=tf.orthogonal_initializer()
    )

    return logits


# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

logits = RNN_BLOCK_LSTM(X)
prediction = tf.nn.softmax(logits)
xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
loss = tf.reduce_mean(xentropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

best_eval = 0.8

with tf.Session() as sess:
    sess.run(init)
    test_len = 128
    for step in range(1, training_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        if step % display_step == 0 or step == 1:
            loss_val, acc = sess.run([loss, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss_val) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            test_X = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
            test_Y = mnist.test.labels[:test_len]
            test_accuracy = sess.run(accuracy, feed_dict={X: test_X, Y: test_Y})
            if test_accuracy > best_eval:
                save_path = saver.save(sess, save_path='./model/model.ckpt', global_step=step)
                print('Saved Model with path {}'.format(save_path))
