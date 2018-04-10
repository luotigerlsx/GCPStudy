# -*- coding: utf-8 -*-

import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


tfrecords_filename = './mnist.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
features = mnist.train.images
labels = mnist.train.labels

for i in range(features.shape[0]):
    image = features[i].tostring()
    label = labels[i]

    example = tf.train.Example(
        features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image),
            'label': _int64_feature(label)
        })
    )

    writer.write(example.SerializeToString())

writer.close()
