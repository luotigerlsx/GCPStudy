# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load("/var/data/training_data.npy") as data:
    features = data["features"]
    labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

dataset = tf.data.Dataset.from_tensor_slices((features, labels))

# Note that the above code snippet will embed the features and labels arrays in your
# TensorFlow graph as tf.constant() operations.


"""
An alternative way is to define dataset in terms of placeholder
"""
# Load the training data into two NumPy arrays, for example using `np.load()`.
with np.load("/var/data/training_data.npy") as data:
    features = data["features"]
    labels = data["labels"]

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
# dataset = ...
iterator = dataset.make_initializable_iterator()

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                              labels_placeholder: labels})

"""
Read from text data
"""
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]

dataset = tf.data.Dataset.from_tensor_slices(filenames)

# Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
# and then concatenate their contents sequentially into a single "flat" dataset.
# * Skip the first line (header row).
# * Filter out lines beginning with "#" (comments).
dataset = dataset.flat_map(
    lambda filename: (
        tf.data.TextLineDataset(filename)
            .skip(1)
            .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))
    )
)
