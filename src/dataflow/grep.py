#!/usr/bin/env python

"""
Copyright Google Inc. 2016
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import apache_beam as beam
import re
import sys


def my_grep(line, term):
    if re.match(r'^' + re.escape(term), line):
        yield line


class ToString(beam.DoFn):
    def process(self, element, *args, **kwargs):
        return [element[0] + str(element[1])]


if __name__ == '__main__':
    p = beam.Pipeline(argv=sys.argv)
    input = './javahelp/src/main/java/com/google/cloud/training/dataanalyst/javahelp/*.java'
    output_prefix = '/tmp/output'
    searchTerm = 'import'

    # find all lines that contain the searchTerm
    (p
     | 'GetJava' >> beam.io.ReadFromText(input)
     | 'Grep' >> beam.FlatMap(lambda line: my_grep(line, searchTerm))
     | 'Split' >> beam.FlatMap(lambda line: [(item, 1) for item in line.split(' ')])
     | beam.GroupByKey()
     | beam.Map(lambda x: (x[0], list(x[1])))
     # | 'Count' >> beam.CombinePerKey(lambda values: sum(values))
     # | 'ToString' >> beam.Map(lambda x: x[0] + str(x[1]))
     # | 'ToString' >> beam.FlatMap(lambda x: [x[0] + str(x[1])])
     # | 'ToString' >> beam.ParDo(ToString())
     | 'write' >> beam.io.WriteToText(output_prefix)
     )


    p.run().wait_until_finish()
