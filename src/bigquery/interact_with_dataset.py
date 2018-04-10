# Imports the Google Cloud client library
from google.cloud import bigquery

client = bigquery.Client(project="woven-rush-197905")
client.list_datasets()
dataset_ref = client.dataset('demos_create')

# For creating dataset
dataset = bigquery.Dataset(dataset_ref)
dataset.location = 'US'
dataset = client.create_dataset(dataset)

# IF the dataset is there already
dataset = client.get_dataset(dataset_ref)
