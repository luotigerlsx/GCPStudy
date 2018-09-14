import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tensorflow.contrib import eager as tfe

tfd = tfp.distributions

from diag_jacobian_pfor import diag_jacobian_pfor as diag_jacobian

dtype = np.float32
with tf.Session(graph=tf.Graph()) as sess:
    true_mean = dtype([0, 0, 0])
    true_cov = dtype([[1, 0.25, 0.25], [0.25, 2, 0.25], [0.25, 0.25, 3]])
    chol = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)


    # Assume that the state is passed as a list of tensors `x` and `y`.
    # Then the target function is defined as follows:
    def target_fn(x, y):
        # Stack the input tensors together
        z = tf.concat([x, y], axis=-1) - true_mean
        return target.log_prob(z)


    sample_shape = [3, 5]
    state = [tf.ones(sample_shape + [2], dtype=dtype),
             tf.ones(sample_shape + [1], dtype=dtype)]
    fn_val = target_fn(*state)
    grad_fn = tfe.gradients_function(target_fn)
    if tfe.executing_eagerly():
        grads = grad_fn(*state)
    else:
        grads = tf.gradients(fn_val, state)

    # We can either pass the `sample_shape` of the `state` or not, which impacts
    # computational speed of `diag_jacobian`
    _, diag_jacobian_shape_passed = diag_jacobian(
        xs=state, ys=grads, sample_shape=tf.shape(fn_val))
    _, diag_jacobian_shape_none = diag_jacobian(
        xs=state, ys=grads)

    diag_jacobian_shape_passed_ = sess.run(diag_jacobian_shape_passed)
    diag_jacobian_shape_none_ = sess.run(diag_jacobian_shape_none)

print('hessian computed through `diag_jacobian`, sample_shape passed: ',
      np.concatenate(diag_jacobian_shape_passed_, -1))
print('hessian computed through `diag_jacobian`, sample_shape skipped',
      np.concatenate(diag_jacobian_shape_none_, -1))