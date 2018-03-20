# -*- coding: utf-8 -*-

import google.datalab.ml as ml
import tensorflow as tf
from tensorflow.contrib import layers
import tensorflow.contrib.learn as tflearn
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import shutil

"""
Read data created in Lab1a, but this time make it more general, 
so that we are reading in batches. Instead of using Pandas, 
we will add a filename queue to the TensorFlow graph. This queue will be cycled through 
num_epochs times.
"""

CSV_COLUMNS = ['fare_amount', 'pickuplon', 'pickuplat',
               'dropofflon', 'dropofflat', 'passengers', 'key']
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [[0.0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]


def read_dataset(filename, num_epochs=None,
                 batch_size=512,
                 mode=tf.contrib.learn.ModeKeys.TRAIN):
    def _input_fn():
        filename_queue = tf.train.string_input_producer(
            [filename],
            num_epochs=num_epochs,
            shuffle=True)
        reader = tf.TextLineReader()
        _, value = reader.read_up_to(filename_queue, num_records=batch_size)

        value_column = tf.expand_dims(value, -1)
        columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        label = features.pop(LABEL_COLUMN)
        return features, label

    return _input_fn


def get_train():
    return read_dataset('../../data/taxi-train.csv', num_epochs=100,
                        mode=tf.contrib.learn.ModeKeys.TRAIN)


def get_valid():
    return read_dataset('../../data/taxi-valid.csv', num_epochs=1,
                        mode=tf.contrib.learn.ModeKeys.EVAL)


def get_test():
    return read_dataset('../../data/taxi-test.csv', num_epochs=1,
                        mode=tf.contrib.learn.ModeKeys.EVAL)


INPUT_COLUMNS = [
    layers.real_valued_column('pickuplon'),
    layers.real_valued_column('pickuplat'),
    layers.real_valued_column('dropofflat'),
    layers.real_valued_column('dropofflon'),
    layers.real_valued_column('passengers'),
]

feature_cols = INPUT_COLUMNS


def experiment_fn(output_dir):
    return tflearn.Experiment(
        tflearn.LinearRegressor(feature_columns=feature_cols, model_dir=output_dir),
        train_input_fn=get_train(),
        eval_input_fn=get_valid(),
        eval_metrics={
            'rmse': tflearn.MetricSpec(
                metric_fn=metrics.streaming_root_mean_squared_error
            )
        }
    )


shutil.rmtree('taxi_trained', ignore_errors=True)  # start fresh each time
learn_runner.run(experiment_fn, 'taxi_trained')
