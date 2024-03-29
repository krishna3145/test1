# Step 3: Creating the Search Application (search_app.py)

import streamlit as st
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import numpy as np

def search_embeddings(es, index_name, query_embedding, top_k=5):
    """
    Searches the Elasticsearch index for the top_k most similar records to the query_embedding.
    """
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_embedding, 'embedding') + 1.0",
                "params": {"query_embedding": query_embedding}
            }
        }
    }
    
    response = es.search(
        index=index_name,
        body={
            "size": top_k,
            "query": script_query,
            "_source": ["Series_Title", "Overview"]
        }
    )
    
    return [(hit['_source']['Series_Title'], hit['_source']['Overview']) for hit in response['hits']['hits']]

def main(es_host='localhost', es_port=9200, index_name='movies'):
    """
    Main function to run the Streamlit app.
    """
    st.title("Semantic Search Engine for Movies")
    
    # User query input
    user_query = st.text_input("Enter your search query:", "")
    
    if user_query:
        # Initialize the sentence transformer model for embedding
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Embed the user query
        query_embedding = model.encode(user_query)
        
        # Normalize the query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Initialize Elasticsearch client
        es = Elasticsearch([{'host': es_host, 'port': es_port}])
        
        # Perform the search
        results = search_embeddings(es, index_name, query_embedding, top_k=5)
        
        # Display the results
        for title, overview in results:
            st.write(f"**{title}**")
            st.write(overview)
            st.write("----")

# Example usage (commented out):
# if __name__ == "__main__":
#     main()

# This script creates a Streamlit app that allows users to search for movies using semantic search.
# The app embeds the user's query, searches for the most similar embeddings in Elasticsearch, and displays the results.
# Before running this script, ensure that the sentence-transformers library and Streamlit are installed,
# and that Elasticsearch is running and accessible at the specified host and port.
