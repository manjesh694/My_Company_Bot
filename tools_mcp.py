import os
from google.cloud import bigquery
from google.cloud import storage

# 1. BigQuery Tool: For company data analysis
def query_company_data(sql_query: str) -> str:
    """Executes a SQL query on BigQuery to answer questions about company metrics, sales, or users."""
    try:
        client = bigquery.Client()
        query_job = client.query(sql_query)
        results = query_job.result()
        rows = [dict(row) for row in results]
        return str(rows[:10]) # return top 10 results
    except Exception as e:
        return f"Error executing query: {str(e)}"

# 2. Cloud Storage (GCS) Tool: Read company documents/policies
def read_company_doc(bucket_name: str, file_name: str) -> str:
    """Reads a file or document from Google Cloud Storage."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        return blob.download_as_text()
    except Exception as e:
        return f"Error reading document: {str(e)}"