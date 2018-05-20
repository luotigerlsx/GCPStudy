# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint
import tempfile
import logging
import sys
import argparse

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam.impl as beam_impl
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main(p=None):
    def preprocessing_fn(inputs):
        """Preprocess input columns into transformed columns."""
        x = inputs['x']
        y = inputs['y']
        s = inputs['s']
        x_centered = x - tft.mean(x)
        y_normalized = tft.scale_to_0_1(y)
        s_integerized = tft.string_to_int(s)
        x_centered_times_y_normalized = (x_centered * y_normalized)
        return {
            'x_centered': x_centered,
            'y_normalized': y_normalized,
            'x_centered_times_y_normalized': x_centered_times_y_normalized,
            's_integerized': s_integerized
        }

    raw_data = [
        {'x': 1, 'y': 1, 's': 'hello'},
        {'x': 2, 'y': 2, 's': 'world'},
        {'x': 3, 'y': 3, 's': 'hello'}
    ]

    # raw_data_p = p | beam.Create(raw_data)

    raw_data_metadata = dataset_metadata.DatasetMetadata(dataset_schema.Schema({
        's': dataset_schema.ColumnSchema(
            tf.string, [], dataset_schema.FixedColumnRepresentation()),
        'y': dataset_schema.ColumnSchema(
            tf.float32, [], dataset_schema.FixedColumnRepresentation()),
        'x': dataset_schema.ColumnSchema(
            tf.float32, [], dataset_schema.FixedColumnRepresentation())
    }))

    with beam_impl.Context(temp_dir=tempfile.mkdtemp()):
        transformed_dataset, transform_fn = (  # pylint: disable=unused-variable
                (raw_data, raw_data_metadata) | beam_impl.AnalyzeAndTransformDataset(
            preprocessing_fn))

        transformed_data, transformed_metadata = transformed_dataset  # pylint: disable=unused-variable

        pprint.pprint(transformed_data)
        (transformed_data
         | beam.io.WriteToText('/Users/luoshixin/Personal/GCPStudy/src/tensorflow/tftransform/tmp'))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='First Trial')
    #
    # options, pipeline_args = parser.parse_known_args()
    #
    # with beam.Pipeline(argv=pipeline_args) as p:
    #     main(p)
    main()
