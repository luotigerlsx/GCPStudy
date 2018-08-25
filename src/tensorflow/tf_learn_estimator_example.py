# -*- coding:utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import shutil

tf.logging.set_verbosity(tf.logging.INFO)

# in CSV, target is the first column, after the features, followed by the key
CSV_COLUMNS = ['fare_amount', 'pickuplon', 'pickuplat', 'dropofflon',
               'dropofflat', 'passengers', 'key']
FEATURES = CSV_COLUMNS[1:len(CSV_COLUMNS) - 1]
TARGET = CSV_COLUMNS[0]


# def make_input_fn(df):
#     def pandas_to_tf(pdcol):
#         # convert the pandas column values to float
#         t = tf.constant(pdcol.astype('float32').values)
#         # take the column which is of shape (N) and make it (N, 1)
#         # t = tf.expand_dims(t, -1)
#         tf.logging.info(t.shape)
#         return t
#
#     def input_fn():
#         features = {k: pandas_to_tf(df[k]) for k in FEATURES}
#         labels = tf.constant(df[TARGET].values)
#         return features, labels
#
#     return input_fn


def make_input_fn(df):
    return tf.estimator.inputs.pandas_input_fn(
        x=df[FEATURES],
        y=df[TARGET],
        shuffle=False)


def make_feature_cols():
    input_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
    return input_columns


def print_rmse(model, name, input_fn):
    metrics = model.evaluate(input_fn=input_fn, steps=1)
    print('RMSE on {} dataset = {}'.format(name, np.sqrt(metrics['loss'])))


df_train = pd.read_csv('../../data/taxi-train.csv', header=None, names=CSV_COLUMNS)
df_valid = pd.read_csv('../../data//taxi-valid.csv', header=None, names=CSV_COLUMNS)

tf.logging.set_verbosity(tf.logging.INFO)
shutil.rmtree('taxi_trained', ignore_errors=True)

model = tf.estimator.LinearRegressor(feature_columns=make_feature_cols(),
                                     model_dir='taxi_trained')
model.train(input_fn=make_input_fn(df_train), steps=10)

tf.logging.set_verbosity(tf.logging.INFO)
shutil.rmtree('taxi_trained', ignore_errors=True)  # start fresh each time
model = tf.estimator.DNNRegressor(hidden_units=[32, 8, 2],
                                  feature_columns=make_feature_cols(), model_dir='taxi_trained')
model.train(input_fn=make_input_fn(df_train), steps=100)
print_rmse(model, 'validation', make_input_fn(df_valid))
