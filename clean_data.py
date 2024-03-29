# Step 1: Data Cleaning and Preparation (clean_data.py)

import pandas as pd
import re

def clean_text(text):
    """
    Function to clean text data by removing special characters and converting to lowercase.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    return text

def prepare_dataset(dataset_path):
    """
    Prepares the dataset by cleaning and concatenating relevant columns for embedding.
    """
    # Read the dataset
    movies = pd.read_csv('imdb_top_1000.csv')
    
    # Select and clean the relevant columns
    movies['Series_Title'] = movies['Series_Title'].apply(clean_text)
    movies['Overview'] = movies['Overview'].apply(clean_text)
    
    # Concatenate the columns
    movies['search_text'] = movies['Series_Title'] + " " + movies['Overview']
    
    return movies[['Series_Title', 'Overview', 'search_text']]

# Assuming the dataset path is known (for the sake of example, it's hard-coded here)
# dataset_path = 'imdb_top_1000.csv'
# prepared_data = prepare_dataset(dataset_path)
# prepared_data.to_csv('prepared_movies.csv', index=False)

# Uncomment the above lines and adjust the dataset_path variable as needed to run the code.

# Note: This code is meant to be run as a standalone script and includes comments to guide its usage.
#       The actual dataset path might need to be adjusted based on the file's location.