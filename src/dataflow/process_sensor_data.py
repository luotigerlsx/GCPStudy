# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import logging

import apache_beam as beam
import apache_beam.transforms.window as window
import apache_beam.transforms.trigger as trigger
import apache_beam.transforms.combiners as comb
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import StandardOptions
from google.cloud import bigquery


class LineInfo(object):
    def __init__(self, record):
        self.record_split = record.strip().split(',')
        self.timestamp = self.record_split[0]
        self.latitude = float(self.record_split[1])
        self.longitude = float(self.record_split[2])
        self.freeway_id = int(self.record_split[3])
        self.freeway_dir = self.record_split[4]
        self.lan = int(self.record_split[5])
        self.speed = float(self.record_split[6])

    def __str__(self):
        return ','.join(self.record_split)

    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'highway': self.freeway_id,
            'direction': self.freeway_dir,
            'lane': self.lan,
            'speed': self.speed,
            'sensorId': self.get_sensor_key()
        }

    def get_sensor_key(self):
        return ','.join(self.record_split[1:-2])


class ToString(beam.DoFn):
    def process(self, element, *args, **kwargs):
        return element.__str__()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    """Build and run the pipeline."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--project', required=True)
    parser.add_argument('--topic', required=True)
    parser.add_argument('--speedFactor', required=True, type=int)
    parser.add_argument('--averageInterval', required=True, type=int)

    options, pipeline_args = parser.parse_known_args()

    # We use the save_main_session option because one or more DoFn's in this
    # workflow rely on global context (e.g., a module imported at module level).
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    pipeline_options.view_as(StandardOptions).streaming = True

    schema = [
        bigquery.SchemaField('timestamp', 'TIMESTAMP'),
        bigquery.SchemaField('latitude', 'FLOAT'),
        bigquery.SchemaField('longitude', 'FLOAT'),
        bigquery.SchemaField('highway', 'STRING'),
        bigquery.SchemaField('direction', 'STRING'),
        bigquery.SchemaField('lane', 'INTEGER'),
        bigquery.SchemaField('speed', 'FLOAT'),
        bigquery.SchemaField('sensorId', 'STRING'),
    ]

    schema = ','.join([x.name + ':' + x.field_type for x in schema])

    p = beam.Pipeline(options=pipeline_options)

    # process command line parameters to get topic path and average interval
    full_topic_path = 'projects/' + options.project + "/topics/" + options.topic
    average_interval = round((options.averageInterval / options.speedFactor))
    average_frequency = round(average_interval / 2)

    lines = p | beam.io.ReadStringsFromPubSub(topic=full_topic_path)

    parsed_lines = lines | 'ParseAsLine' >> beam.Map(lambda x: LineInfo(x))
    (parsed_lines
     | 'Window' >> beam.WindowInto(window
                                   .SlidingWindows(average_interval, average_frequency))
     | 'BySensor' >> beam.Map(lambda x: (x.get_sensor_key(), x.speed))
     | 'AvgBySensor' >> comb.Mean.PerKey()
     | 'Print' >> beam.ParDo(print)
     )

    (parsed_lines
     | 'TranformForBigQuery' >> beam.Map(lambda x: x.to_dict())
     | 'WriteToBigQuery' >> beam.io.WriteToBigQuery(
                table='speed_analysis',
                dataset='demos',
                project=options.project,
                schema=schema,  # Pass the defined table_schema
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND))


    result = p.run()
    result.wait_until_finish()
