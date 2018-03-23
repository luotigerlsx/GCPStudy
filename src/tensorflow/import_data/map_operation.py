# -*- coding:utf-8 -*-


import tensorflow as tf

"""
The Dataset.map(f) transformation produces a new dataset by applying a given function f to 
each element of the input dataset. It is based on the map() function that is commonly applied 
to lists (and other structures) in functional programming languages. The function f takes the 
tf.Tensor objects that represent a single element in the input, and returns the tf.Tensor objects 
that will represent a single element in the new dataset. 

******
Its implementation uses standard TensorFlow operations to transform one element into another.
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
            .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))

"""
It is possible to use arbitraty python logic with tf.py_func()

"""

import cv2


# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_py_function(filename, label):
    image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
    return image_decoded, label


# Use standard TensorFlow operations to resize the image to a fixed shape.
def _resize_function(image_decoded, label):
    image_decoded.set_shape([None, None, None])
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized, label


filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
labels = [0, 37, 29, 1, ...]

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(
    lambda filename, label: tuple(tf.py_func(
        _read_py_function, [filename, label], [tf.uint8, label.dtype])))
dataset = dataset.map(_resize_function)
