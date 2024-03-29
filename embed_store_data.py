# Step 2: Embedding and Storing Data (embed_and_store_data.py)

from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import pandas as pd

def embed_texts(model, texts):
    """
    Generates embeddings for the given list of texts using the specified model.
    """
    return model.encode(texts, show_progress_bar=True)

def store_in_elasticsearch(es, index_name, records):
    """
    Stores the given records in the specified Elasticsearch index.
    Each record should be a dictionary with at least an 'id' and 'embedding' field.
    """
    for record in records:
        es.index(index=index_name, id=record['id'], body=record)

def main(prepared_data_path, es_host='localhost', es_port=9200, index_name='movies'):
    """
    Main function to prepare and store embeddings in Elasticsearch.
    """
    # Load prepared data
    movies = pd.read_csv(prepared_data_path)
    
    # Initialize the sentence transformer model for embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings for the prepared text data
    print("Generating embeddings...")
    movies['embedding'] = embed_texts(model, movies['search_text'].tolist())
    
    # Initialize Elasticsearch client
    es = Elasticsearch([{'host': es_host, 'port': es_port}])
    
    # Convert DataFrame to list of dictionaries for Elasticsearch
    records = movies.to_dict(orient='records')
    
    # Store the records in Elasticsearch
    print("Storing data in Elasticsearch...")
    store_in_elasticsearch(es, index_name, records)
    
    print("Data successfully embedded and stored.")

# Example usage (commented out):
# main(prepared_data_path='prepared_movies.csv')

# This script embeds text data using a sentence transformer model and stores the embeddings in Elasticsearch.
# Before running this script, ensure that Elasticsearch is running and accessible at the specified host and port.
# Adjust the 'prepared_data_path' variable according to the actual path of your prepared data CSV file.
