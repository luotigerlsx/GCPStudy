# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

# Define the format of your input data including unused columns
CSV_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
               'marital_status', 'occupation', 'relationship', 'race', 'gender',
               'capital_gain', 'capital_loss', 'hours_per_week',
               'native_country', 'income_bracket', 'key']
CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                       [0], [0], [0], [''], [''], ['nonkey']]

KEY_COLUMN = 'key'

LABEL_COLUMN = 'income_bracket'
LABELS = [' <=50K', ' >50K']


INPUT_COLUMNS = [
    # Categorical based column
    # Vocabulary List
    tf.feature_column.categorical_column_with_vocabulary_list(
        key='workclass',
        vocabulary_list=[' Self-emp-not-inc', ' Private', ' State-gov',
                         ' Federal-gov', ' Local-gov', ' ?', ' Self-emp-inc',
                         ' Without-pay', ' Never-worked']
    ),
    tf.feature_column.categorical_column_with_vocabulary_list(
        key='education',
        vocabulary_list=[' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th',
                         ' Some-college', ' Assoc-acdm', ' Assoc-voc', ' 7th-8th',
                         ' Doctorate', ' Prof-school', ' 5th-6th', ' 10th', ' 1st-4th',
                         ' Preschool', ' 12th']
    ),
    tf.feature_column.categorical_column_with_vocabulary_list(
        key='marital_status',
        vocabulary_list=[' Never-married', ' Married-civ-spouse', ' Divorced',
                         ' Married-spouse-absent', ' Separated', ' Married-AF-spouse',
                         ' Widowed']
    ),

    tf.feature_column.categorical_column_with_vocabulary_list(
        key='relationship',
        vocabulary_list=[' Not-in-family', ' Husband', ' Wife', ' Own-child', ' Unmarried',
                         ' Other-relative']
    ),

    tf.feature_column.categorical_column_with_vocabulary_list(
        key='race',
        vocabulary_list=[' White', ' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo',
                         ' Other']
    ),

    tf.feature_column.categorical_column_with_vocabulary_list(
        key='gender',
        vocabulary_list=['Male', 'Female']
    ),

    # Hash Bucket
    tf.feature_column.categorical_column_with_hash_bucket(
        key='occupation',
        hash_bucket_size=100,
        dtype=tf.string
    ),
    tf.feature_column.categorical_column_with_hash_bucket(
        key='native_country',
        hash_bucket_size=100,
        dtype=tf.string
    ),

    # Numerical based column
    tf.feature_column.numeric_column('age'),
    tf.feature_column.numeric_column('education_num'),
    tf.feature_column.numeric_column('capital_gain'),
    tf.feature_column.numeric_column('capital_loss'),
    tf.feature_column.numeric_column('hours_per_week'),

]

UNUSED_COLUMNS = set(CSV_COLUMNS) - {col.name for col in INPUT_COLUMNS} - \
                 {LABEL_COLUMN}

UNUSED_COLUMNS.remove(KEY_COLUMN)


def forward_key_to_export(estimator):
    estimator = tf.contrib.estimator.forward_features(estimator, KEY_COLUMN)
    # return estimator

    ## This shouldn't be necessary (I've filed CL/187793590 to update extenders.py with this code)
    config = estimator.config

    def model_fn2(features, labels, mode):
        estimatorSpec = estimator._call_model_fn(features, labels, mode, config=config)
        if estimatorSpec.export_outputs:
            for ekey in ['predict', 'serving_default']:
                if (ekey in estimatorSpec.export_outputs and
                        isinstance(estimatorSpec.export_outputs[ekey],
                                   tf.estimator.export.PredictOutput)):
                    estimatorSpec.export_outputs[ekey] = \
                        tf.estimator.export.PredictOutput(estimatorSpec.predictions)
        return estimatorSpec

    return tf.estimator.Estimator(model_fn=model_fn2, config=config)


def build_estimator(config, embedding_size=8, hidden_units=None):
    (gender, race, education, marital_status, relationship,
     workclass, occupation, native_country, age,
     education_num, capital_gain, capital_loss, hours_per_week) = INPUT_COLUMNS

    wide_columns = [
        gender,
        race,
        education,
        marital_status,
        relationship,
        workclass,
        occupation,
        native_country,
        tf.feature_column.crossed_column(
            ['education', 'occupation'], hash_bucket_size=int(1e4)),
    ]

    dense_columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.embedding_column(occupation, dimension=embedding_size),
        tf.feature_column.embedding_column(native_country, dimension=embedding_size)
    ]

    def my_metric(labels, predictions):
        pred_values = predictions['class_ids']
        return {'hp_auc': tf.metrics.auc(labels, pred_values)}

    estimator = tf.estimator.DNNLinearCombinedClassifier(
        config=config,
        dnn_feature_columns=dense_columns,
        linear_feature_columns=wide_columns,
        dnn_hidden_units=hidden_units or [100, 70, 50, 25]
    )

    estimator = tf.contrib.estimator.add_metrics(estimator, my_metric)

    estimator = forward_key_to_export(estimator)

    return estimator


def parse_label_column(label_string_tensor):
    """Parses a string tensor into the label tensor
    Args:
      label_string_tensor: Tensor of dtype string. Result of parsing the
      CSV column specified by LABEL_COLUMN
    Returns:
      A Tensor of the same shape as label_string_tensor, should return
      an int64 Tensor representing the label index for classification tasks,
      and a float32 Tensor representing the value for a regression task.
    """
    # Build a Hash Table inside the graph
    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(LABELS))

    # Use the hash table to convert string labels to ints and one-hot encode
    return table.lookup(label_string_tensor)


def parse_csv(csv_string):
    """Parse a input string with csv format for a dictionary {feature_name: feature}
    """
    columns = tf.decode_csv(csv_string + ',', record_defaults=CSV_COLUMN_DEFAULTS)
    features = dict(zip(CSV_COLUMNS, columns))
    for col in UNUSED_COLUMNS:
        features.pop(col)
    return features


def input_fn(filenames,
             num_epochs=None,
             shuffle=True,
             skip_header_lines=0,
             batch_size=200):
    file_name_dataset = tf.data.Dataset.from_tensor_slices(filenames)

    tf.data.TFRecordDataset

    dataset = (file_name_dataset
               .flat_map(lambda file_name: tf.data.TextLineDataset(file_name).skip(skip_header_lines))
               .map(parse_csv)
               )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)

    dataset = dataset.repeat(num_epochs).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features, parse_label_column(features.pop(LABEL_COLUMN))


def csv_serving_input_fn():
    # This is for reading a simple CSV line from input
    csv_row = tf.placeholder(
        shape=[None],
        dtype=tf.string
    )
    # The output of this line is supposed to be a dictionary of {feature_name: feature_tensor}
    features = parse_csv(csv_row)
    features.pop(LABEL_COLUMN)
    return tf.estimator.export.ServingInputReceiver(features, {'csv_row': csv_row})


# [START serving-function]
def json_serving_input_fn():
    """Build the serving inputs."""
    inputs = {}
    for feat in INPUT_COLUMNS:
        inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

    inputs[KEY_COLUMN] = tf.placeholder_with_default(tf.constant(['nokey']), [None])

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in inputs.items()
    }

    return tf.estimator.export.ServingInputReceiver(features, inputs)


# [END serving-function]

SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'CSV': csv_serving_input_fn
}
