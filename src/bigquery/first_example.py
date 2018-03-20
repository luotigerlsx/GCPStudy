# Imports the Google Cloud client library
from google.cloud import bigquery

# Instantiates a client
bigquery_client = bigquery.Client(project="woven-rush-197905")

# # The name for the new dataset
# dataset_id = 'demos'
#
# # Prepares a reference to the new dataset
# dataset_ref = bigquery_client.dataset(dataset_id)
# dataset = bigquery.Dataset(dataset_ref)
#
# tables = list(bigquery_client.list_tables(dataset_ref))
#
# for table in tables:
#     print(table.__dict__)

QUERY = (
    'SELECT name FROM `bigquery-public-data.usa_names.usa_1910_2013` '
    'WHERE state = "TX" '
    'LIMIT 100')

TIMEOUT = 30  # in seconds
query_job = bigquery_client.query(QUERY)  # API request - starts the query

# Waits for the query to finish
iterator = query_job.result(timeout=TIMEOUT)
rows = list(iterator)

print(rows[0:2])
