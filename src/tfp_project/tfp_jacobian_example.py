import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tensorflow.contrib import eager as tfe

from diag_jacobian import diag_jacobian
from diag_jacobian_pfor import diag_jacobian_pfor

X = tf.Variable(tf.random_normal([3, 3]))

A = tf.Variable(tf.random_uniform(minval=1, maxval=10, shape=[3,3]))
y = tf.multiply(X, A)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    sample_shape = [3, 3]

    # Now let's try to compute the jacobian
    # dydx = diag_jacobian(xs=X, ys=y)
    dydx_with_ss = diag_jacobian(xs=X, ys=y, sample_shape=sample_shape)
    # print(sess.run([dydx, A]))
    print(sess.run([dydx_with_ss, A]))
    # print(sess.run(diag_jacobian_pfor(xs=xtf, ys=ytf)))
