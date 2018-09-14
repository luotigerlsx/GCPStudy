# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utility function to construct the diagonal of a Jacobian matrix."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf

from tensorflow.contrib import eager as tfe
from tensorflow.python.ops.parallel_for import control_flow_ops

__all__ = [
    'diag_jacobian_pfor',
]

# TODO: Add sample_shape
"""
If we know that some y's are only affected by single x's, i.e., one-to-one corresponding. Then, call tf.gradients
once can generate correct individual partial derivative. For example
X = tf.Variable(tf.random_normal([3, 3]))

A = tf.Variable(tf.random_uniform(minval=1, maxval=10, shape=[3,3]))
y = tf.multiply(X, A)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    sample_shape = [3, 3]

    # Now let's try to compute the jacobian
    dydx = diag_jacobian(xs=X, ys=y)
    dydx_with_ss = diag_jacobian(xs=X, ys=y, sample_shape=sample_shape)
    print(sess.run([dydx, A]))
    print(sess.run([dydx_with_ss, A]))
"""


# TODO: Add support for eager

def diag_jacobian_pfor(xs,
                       ys=None,
                       fn=None,
                       sample_shape=None,
                       use_pfor=True,
                       name=None):
    with tf.name_scope(name, 'jacobians_diag', [xs, ys]):

        if sample_shape is None:
            sample_shape = [1]

        # Output Jacobian diagonal
        jacobians_diag_res = []
        # Convert input `xs` to a list
        xs = list(xs) if _is_list_like(xs) else [xs]
        xs = [tf.convert_to_tensor(x) for x in xs]
        if ys is None:
            if fn is None:
                raise ValueError('Both `ys` and `fn` can not be `None`')
            else:
                ys = fn(*xs)
        # Convert ys to a list
        ys = list(ys) if _is_list_like(ys) else [ys]
        if len(xs) != len(ys):
            raise ValueError('`xs` and `ys` should have the same length')

        for y, x in zip(ys, xs):
            shape_x = tf.shape(x)
            # Broadcast `y` to the shape of `x`.
            y_ = y + tf.zeros_like(x)
            # Change `event_shape` to one-dimension
            flat_y = tf.reshape(y_, shape=tf.concat([[-1], sample_shape], -1))

            n = tf.size(x) / tf.to_int32(tf.reduce_prod(sample_shape))
            n = tf.to_int32(n)

            def grad_fn(i):
                res = tf.gradients(tf.gather(flat_y, i), x)[0]
                if res is None:
                    res = tf.zeros(shape_x, dtype=x.dtype)  # pylint: disable=cell-var-from-loop

                flat_res = tf.reshape(res, tf.concat([[-1], sample_shape], -1))
                return tf.gather(flat_res, i)

            if use_pfor:
                jacobian_diag_res = control_flow_ops.pfor(grad_fn, n)
            else:
                jacobian_diag_res = control_flow_ops.for_loop(grad_fn, [y.dtype], n)

            reshaped_jacobian_diag = tf.reshape(jacobian_diag_res, shape_x)
            jacobians_diag_res.append(reshaped_jacobian_diag)

        return jacobians_diag_res


def _is_list_like(x):
    """Helper which returns `True` if input is `list`-like."""
    return isinstance(x, (tuple, list))


if __name__ == '__main__':
    import timeit


    def run_test(use_sample_shape, use_pfor):

        with tf.Session(graph=tf.Graph()) as sess:

            sample_shape = [30, 30]

            X = tf.Variable(tf.random_normal(sample_shape))

            A = tf.Variable(tf.random_uniform(minval=1, maxval=10, shape=sample_shape))
            y = tf.multiply(X, A)

            init = tf.global_variables_initializer()

            sess.run(init)

            if use_pfor:
                # Now let's try to compute the jacobian
                if use_sample_shape:
                    dydx = diag_jacobian_pfor(xs=X, ys=y, sample_shape=sample_shape, use_pfor=True)
                else:
                    dydx = diag_jacobian_pfor(xs=X, ys=y, use_pfor=True)
            else:
                from tensorflow_probability.python.math import diag_jacobian
                if use_sample_shape:
                    dydx = diag_jacobian(xs=X, ys=y, sample_shape=sample_shape)
                else:
                    dydx = diag_jacobian(xs=X, ys=y)
            sess.run([dydx, A])


    result_nss = timeit.timeit(
        'run_test(use_sample_shape=False, use_pfor=True)', setup='from __main__ import run_test',
        number=20)
    result_ss = timeit.timeit(
        'run_test(use_sample_shape=True, use_pfor=True)', setup='from __main__ import run_test',
        number=20)
    result_ss_npr = timeit.timeit(
        'run_test(use_sample_shape=True, use_pfor=False)', setup='from __main__ import run_test',
        number=20)
    result_nss_npr = timeit.timeit(
        'run_test(use_sample_shape=False, use_pfor=False)', setup='from __main__ import run_test',
        number=20)
    print(result_ss, result_nss, result_ss_npr, result_nss_npr)
