from google.cloud import bigquery

client = bigquery.Client(project="woven-rush-197905")
dataset_ref = client.dataset('demos')

# If table not exists
schema = [
    bigquery.SchemaField('full_name', 'STRING', mode='REQUIRED'),
    bigquery.SchemaField('age', 'INTEGER', mode='REQUIRED'),
]
table_ref = dataset_ref.table('my_table')
table = bigquery.Table('table_ref', schema=schema)
table = client.create_table(table)

# If table already exists
table_ref = dataset_ref.table('my_table')
table = client.get_table(table_ref)

for row in client.list_rows(table):  # API request
    print(row)

rows = client.list_rows(table, max_results=10)
assert len(list(rows)) == 10

# Specify selected fields to limit the results to certain columns
fields = table.schema[:2]  # first two columns
rows = client.list_rows(table, selected_fields=fields, max_results=10)
assert len(rows.schema) == 2
assert len(list(rows)) == 10

# Use the start index to load an arbitrary portion of the table
rows = client.list_rows(table, start_index=10, max_results=10)
