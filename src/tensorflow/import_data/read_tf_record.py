# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

tf_file = './mnist.tfrecords'


def parse_tf_record(example_proto):
    features = {"image_raw": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features["image_raw"], parsed_features["label"]


def decode_image(image_raw):
    return np.fromstring(image_raw, np.float32)


filenames = [tf_file]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(parse_tf_record) \
    .map(lambda record, label: (tf.py_func(decode_image, [record], tf.float32), label))
dataset.batch(4)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    for i in range(1):
        print(sess.run(next_element))
