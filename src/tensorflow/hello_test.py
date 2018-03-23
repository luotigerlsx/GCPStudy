# -*- coding: utf-8 -*-

import tensorflow as tf

feature_columns = [tf.contrib.layers.real_valued_column("sq_footage")]

estimator = tf.contrib.learn.LinearRegressor(feature_columns=feature_columns)


def input_fn_train():
    feature_data = {
        "sq_footage": tf.constant([[1000], [2000]])
    }

    label_data = tf.constant([[10000], [20000]])

    return feature_data, label_data


estimator.fit(input_fn=input_fn_train, steps=10)
