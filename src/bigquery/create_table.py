# Imports the Google Cloud client library
from google.cloud import bigquery

client = bigquery.Client(project="woven-rush-197905")
dataset_ref = client.dataset('my_dataset')

schema = [
    bigquery.SchemaField('full_name', 'STRING', mode='REQUIRED'),
    bigquery.SchemaField('age', 'INTEGER', mode='REQUIRED'),
]
table_ref = dataset_ref.table('my_table')
table = bigquery.Table(table_ref, schema=schema)
table = client.create_table(table)  # API request
