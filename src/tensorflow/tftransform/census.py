# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import os
import shutil
import pprint
import tempfile

import tensorflow as tf
import tensorflow_transform as tft

from tensorflow.contrib import lookup

from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.coders import csv_coder
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io

import apache_beam as beam

ORDERED_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'label'
]

CATEGORICAL_FEATURE_KEYS = [
    'workclass', 'education', 'marital-status', 'occupation', 'relationship',
    'race', 'sex', 'native-country'
]

NUMERIC_FEATURE_KEYS = [
    'age', 'education-num', 'capital-gain', 'capital-loss',
    'hours-per-week'
]

LABEL_KEY = 'label'

UNUSED_COLUMN = ['fnlwgt', LABEL_KEY]

CSV_COLUMN_DEFAULTS = [[0.0], [''], [0.0], [''], [0.0], [''], [''], [''], [''], [''],
                       [0.0], [0.0], [0.0], [''], ['']]

TRANSFORMED_TRAIN_DATA_FILEBASE = 'train_transformed'
TRANSFORMED_TEST_DATA_FILEBASE = 'test_transformed'
EXPORTED_MODEL_DIR = 'exported_model_dir'

BUCKET_SIZES = [9, 17, 8, 15, 17, 6, 3, 43]


def create_raw_meta():
    column_schemas = {}

    column_schemas.update({
        fea: dataset_schema.ColumnSchema(tf.string, [], dataset_schema.FixedColumnRepresentation())
        for fea in CATEGORICAL_FEATURE_KEYS
    })

    column_schemas.update({
        fea: dataset_schema.ColumnSchema(tf.float32, [], dataset_schema.FixedColumnRepresentation())
        for fea in NUMERIC_FEATURE_KEYS
    })

    column_schemas[LABEL_KEY] = dataset_schema.ColumnSchema(
        tf.string, [], dataset_schema.FixedColumnRepresentation())

    return dataset_metadata.DatasetMetadata(dataset_schema.Schema(column_schemas))


RAW_DATA_META = create_raw_meta()


def transform_data(train_data_file, test_data_file, working_dir):
    def pre_processing_fun(inputs):
        outputs = {}

        for fea in NUMERIC_FEATURE_KEYS:
            outputs[fea] = tft.scale_to_0_1(inputs[fea])

        for fea in CATEGORICAL_FEATURE_KEYS:
            outputs[fea] = tft.string_to_int(inputs[fea])

        def convert_label(label):
            table = lookup.index_table_from_tensor(['>50K', '<=50K'])
            return table.lookup(label)

        outputs[LABEL_KEY] = tft.apply_function(convert_label, inputs[LABEL_KEY])

        return outputs

    with beam.Pipeline() as pipeline:
        with beam_impl.Context(temp_dir=tempfile.mktemp()):
            converter = csv_coder.CsvCoder(ORDERED_COLUMNS, RAW_DATA_META.schema)

            '''
            Transform and save test data
            '''
            raw_train_data = (
                    pipeline
                    | "Read raw train input" >> beam.io.textio.ReadFromText(train_data_file)
                    | "Filter train line" >> beam.Filter(lambda x: x)
                    | "Fix commas train data" >> beam.Map(lambda x: x.replace(', ', ','))
                    | "Decode train as csv" >> beam.Map(converter.decode)
            )

            raw_train_dataset = (raw_train_data, RAW_DATA_META)

            transformed_train_dataset, transform_fn = (
                    raw_train_dataset | beam_impl.AnalyzeAndTransformDataset(pre_processing_fun))

            transformed_train_data, transformed_train_meta = transformed_train_dataset

            # Save transformed training data
            (transformed_train_data
             | "Save transformed train data" >> beam.io.tfrecordio.WriteToTFRecord(
                        os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE),
                        coder=example_proto_coder.ExampleProtoCoder(
                            transformed_train_meta.schema))
             )

            '''
            Transform and save test data
            '''
            raw_test_data = (
                    pipeline
                    | "Read raw test input" >> beam.io.textio.ReadFromText(test_data_file)
                    | "Filter test line" >> beam.Filter(lambda x: x)
                    | "Fix commas test data" >> beam.Map(lambda x: x.replace(', ', ','))
                    | "Decode test as csv" >> beam.Map(converter.decode)
            )

            raw_test_dataset = (raw_test_data, RAW_DATA_META)

            transformed_test_dataset = (raw_test_dataset, transform_fn) | beam_impl.TransformDataset()

            transformed_test_data, _ = transformed_test_dataset

            # Save transformed training data
            (transformed_test_data
             | "Save transformed test data" >> beam.io.tfrecordio.WriteToTFRecord(
                        os.path.join(working_dir, TRANSFORMED_TEST_DATA_FILEBASE),
                        coder=example_proto_coder.ExampleProtoCoder(
                            transformed_train_meta.schema))
             )

            '''
            Save transform function
            '''
            (transform_fn
             | 'WriteTransformFn' >> transform_fn_io.WriteTransformFn(working_dir))


def make_input_function(working_dir,
                        filebase,
                        num_epochs=None,
                        shuffle=True,
                        batch_size=200):
    transformed_metadata = metadata_io.read_metadata(
        os.path.join(
            working_dir, transform_fn_io.TRANSFORMED_METADATA_DIR))
    transformed_feature_spec = transformed_metadata.schema.as_feature_spec()

    def parse_tf_record(example_proto):
        parsed_features = tf.parse_single_example(example_proto, transformed_feature_spec)
        return parsed_features

    def input_func():
        file_pattern = os.path.join(working_dir, filebase + '-*')
        file_names = tf.data.TFRecordDataset.list_files(file_pattern)
        dataset = file_names.flat_map(lambda x: tf.data.TFRecordDataset(x)).map(parse_tf_record)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 10)

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        return features, features.pop(LABEL_KEY)

    return input_func


def make_serving_input_fn(working_dir):
    """Creates an input function reading from raw data.

    Args:
      working_dir: Directory to read transformed metadata from.

    Returns:
      The serving input function.
    """
    raw_feature_spec = RAW_DATA_META.schema.as_feature_spec()
    # Remove label since it is not available during serving.
    raw_feature_spec.pop(LABEL_KEY)

    def parse_csv(csv_string):
        """Parse a input string with csv format for a dictionary {feature_name: feature}
        """
        columns = tf.decode_csv(csv_string, record_defaults=CSV_COLUMN_DEFAULTS)
        features = dict(zip(ORDERED_COLUMNS, columns))
        for col in UNUSED_COLUMN:
            features.pop(col)
        return features

    def csv_serving_input_fn():
        """Build the serving inputs."""
        csv_row = tf.placeholder(
            shape=[None],
            dtype=tf.string
        )
        features = parse_csv(csv_row)

        _, transformed_features = (
            saved_transform_io.partially_apply_saved_transform(
                os.path.join(working_dir, transform_fn_io.TRANSFORM_FN_DIR),
                features))

        return tf.estimator.export.ServingInputReceiver(transformed_features, {'csv_row': csv_row})

    return csv_serving_input_fn


def train_and_evaluate(working_dir):
    real_valued_columns = [tf.feature_column.numeric_column(key, shape=())
                           for key in NUMERIC_FEATURE_KEYS]

    # Wrap categorical columns.  Note the combiner is irrelevant since the input
    # only has one value set per feature per instance.
    one_hot_columns = [
        tf.feature_column.categorical_column_with_identity(
            key, num_buckets=num_buckets)
        for key, num_buckets in zip(CATEGORICAL_FEATURE_KEYS, BUCKET_SIZES)]

    lestimator = tf.estimator.LinearRegressor(real_valued_columns + one_hot_columns)

    # lestimator.train(
    #     input_fn=make_input_function(
    #         working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE),
    #     max_steps=1000
    # )

    train_input = make_input_function(
        working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE)

    # Don't shuffle evaluation data
    eval_input = make_input_function(
        working_dir, TRANSFORMED_TEST_DATA_FILEBASE)

    train_spec = tf.estimator.TrainSpec(train_input,
                                        max_steps=1000)

    exporter = tf.estimator.FinalExporter('census',
                                          make_serving_input_fn(working_dir))

    eval_spec = tf.estimator.EvalSpec(eval_input,
                                      steps=100,
                                      exporters=[exporter],
                                      name='census-eval')

    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(model_dir=working_dir)
    print('model dir {}'.format(run_config.model_dir))
    estimator = lestimator

    tf.estimator.train_and_evaluate(estimator,
                                    train_spec,
                                    eval_spec)


if __name__ == '__main__':
    working_dir = '/Users/luoshixin/Personal/GCPStudy/src/tensorflow/tftransform/tmp1/'
    data_dir = '/Users/luoshixin/Downloads/data'
    train_data_file = os.path.join(data_dir, 'adult.data.csv')
    test_data_file = os.path.join(data_dir, 'adult.test.csv')

    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)

    tf.logging.set_verbosity('INFO')

    transform_data(train_data_file, test_data_file, working_dir)
    train_and_evaluate(working_dir)
