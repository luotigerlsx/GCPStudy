from google.cloud import bigquery

client = bigquery.Client()
dataset_id = 'my_dataset'
dataset_ref = client.dataset(dataset_id)

'''
Load data from cloud storage
'''
job_config = bigquery.LoadJobConfig()
job_config.schema = [
    bigquery.SchemaField('name', 'STRING'),
    bigquery.SchemaField('post_abbr', 'STRING')
]
job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON

load_job = client.load_table_from_uri(
    'gs://cloud-samples-data/bigquery/us-states/us-states.json',
    dataset_ref.table('us_states'),
    job_config=job_config)  # API request

assert load_job.job_type == 'load'

load_job.result()  # Waits for table load to complete.

assert load_job.state == 'DONE'
assert client.get_table(dataset_ref.table('us_states')).num_rows > 0

'''
Save query result to a table
'''
job_config = bigquery.QueryJobConfig()

# Set the destination table. Here, dataset_id is a string, such as:
# dataset_id = 'your_dataset_id'
table_ref = client.dataset(dataset_id).table('your_table_id')
job_config.destination = table_ref

# The write_disposition specifies the behavior when writing query results
# to a table that already exists. With WRITE_TRUNCATE, any existing rows
# in the table are overwritten by the query results.
job_config.write_disposition = 'WRITE_TRUNCATE'

# Start the query, passing in the extra configuration.
query_job = client.query(
    'SELECT 17 AS my_col;', job_config=job_config)

rows = list(query_job)  # Waits for the query to finish
assert len(rows) == 1
row = rows[0]
assert row[0] == row.my_col == 17

# In addition to using the results from the query, you can read the rows
# from the destination table directly.
iterator = client.list_rows(
    table_ref, selected_fields=[bigquery.SchemaField('my_col', 'INT64')])

rows = list(iterator)
assert len(rows) == 1
row = rows[0]
assert row[0] == row.my_col == 17

'''
Insert rows
'''
rows_to_insert = [
    (u'Phred Phlyntstone', 32),
    (u'Wylma Phlyntstone', 29),
]

errors = client.insert_rows(table, rows_to_insert)  # API request

assert errors == []
