# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

RAW_CSV_COLUMNS = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday',
                   'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance']

INPUT_CSV_DEFAULT = [[''], [''], [''], [0.0], [0], [''], [''], [''],
                     [''], [''], [''], [0.0]]

CSV_COLUMN_NOT_USE = ['Customers']

LABEL_COLUMN = 'Sales'


def build_estimator(config, embedding_size=8, hidden_units=None):
    store = tf.feature_column.categorical_column_with_hash_bucket(
        key='Store',
        hash_bucket_size=20,
        dtype=tf.string)

    day_of_week = tf.feature_column.categorical_column_with_vocabulary_list(
        key='DayOfWeek',
        vocabulary_list=['1', '2', '3', '4', '5', '6', '7'])

    month = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Month',
        vocabulary_list=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])

    open = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Open',
        vocabulary_list=['0', '1'])

    promo = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Promo',
        vocabulary_list=['0', '1'])

    state_holiday = tf.feature_column.categorical_column_with_vocabulary_list(
        key='StateHoliday',
        vocabulary_list=['0', 'a', 'b', 'c'])

    store_type = tf.feature_column.categorical_column_with_vocabulary_list(
        key='StoreType',
        vocabulary_list=['a', 'b', 'c', 'd'])

    assortment = tf.feature_column.categorical_column_with_vocabulary_list(
        key='Assortment',
        vocabulary_list=['a', 'b', 'c'])

    comp_distance = tf.feature_column.numeric_column(
        key='CompetitionDistance',
        normalizer_fn=lambda x: (x - tf.reduce_mean(x)))

    store_embedding = tf.feature_column.embedding_column(
        categorical_column=store,
        dimension=embedding_size)

    store_type_assortment = tf.feature_column.crossed_column(
        keys=['StoreType', 'Assortment'],
        hash_bucket_size=int(1e4))

    comp_distance_bucket = tf.feature_column.bucketized_column(
        source_column=comp_distance,
        boundaries=[0, 3000, 6000, 8000])

    dense_columns = [store_embedding, comp_distance]

    wide_columns = [day_of_week, month, state_holiday,
                    store_type, assortment, store_type_assortment,
                    open, promo, comp_distance_bucket]

    estimator = tf.estimator.DNNLinearCombinedRegressor(
        config=config,
        dnn_feature_columns=dense_columns,
        linear_feature_columns=wide_columns,
        dnn_hidden_units=hidden_units or [100, 70, 50, 25]
    )

    # return estimator

    def my_rmse(labels, predictions):
        pred_values = predictions['predictions']

        new_metric = tf.metrics.root_mean_squared_error(labels, pred_values)

        return {'rmspe': new_metric}

    return tf.contrib.estimator.add_metrics(estimator, my_rmse)


def parse_csv(csv_string):
    """Parse a input string with csv format for a dictionary {feature_name: feature}
    """
    columns = tf.decode_csv(csv_string, record_defaults=INPUT_CSV_DEFAULT)
    features = dict(zip(RAW_CSV_COLUMNS, columns))
    for col in CSV_COLUMN_NOT_USE:
        features.pop(col)

    # 2012-12-11
    features['Month'] = tf.substr(features['Date'], 5, 2)

    return features, features.pop(LABEL_COLUMN)


def input_fn(filename,
             num_epochs=None,
             shuffle=True,
             batch_size=200):
    def _input_fn():
        dataset = tf.data.TextLineDataset(filename).map(parse_csv, num_parallel_calls=3)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 10)

        dataset = dataset.repeat(num_epochs).batch(batch_size).prefetch(1)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    return _input_fn


if __name__ == '__main__':
    tf.logging.set_verbosity('INFO')
    run_config = tf.estimator.RunConfig()

    regressor = build_estimator(run_config)

    train_file = '/Users/luoshixin/Downloads/data/rossmann/clean_train_csv.csv'
    eval_file = '/Users/luoshixin/Downloads/data/rossmann/clean_test_csv.csv'

    regressor.train(input_fn=input_fn(train_file, 10), max_steps=1000)

    eval_results = regressor.evaluate(input_fn=input_fn(eval_file, 1, False))
    print(eval_results)
