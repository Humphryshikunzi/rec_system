import pandas as pd
from sentence_transformers import SentenceTransformer
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
DATA_DIR = "artifacts/data/train/"
OUTPUT_DIR = "artifacts/data/train/"
USER_DATA_FILE = os.path.join(DATA_DIR, "users.csv")
POST_DATA_FILE = os.path.join(DATA_DIR, "posts.csv")
USER_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "users_with_embeddings.parquet")
POST_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "posts_with_embeddings.parquet")
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

def generate_embeddings():
    """
    Loads user and post data, generates text embeddings, and saves the updated data to Parquet files.
    """
    try:
        # Load data
        logging.info(f"Loading data from {USER_DATA_FILE} and {POST_DATA_FILE}")
        users_df = pd.read_csv(USER_DATA_FILE)
        posts_df = pd.read_csv(POST_DATA_FILE)
        logging.info("Data loaded successfully.")
        # Ensure 'creation_timestamp' in users_df is in datetime format
        if 'creation_timestamp' in users_df.columns:
            try:
                users_df['creation_timestamp'] = pd.to_datetime(users_df['creation_timestamp'])
                logging.info("Successfully converted 'creation_timestamp' in users_df to datetime.")
            except Exception as e:
                logging.error(f"Error converting 'creation_timestamp' in users_df to datetime: {e}. Please check the format in the source CSV.")
        else:
            logging.warning("'creation_timestamp' column not found in users_df. Skipping datetime conversion for it.")
# Ensure 'creation_timestamp' in posts_df is in datetime format
        if 'creation_timestamp' in posts_df.columns:
            try:
                posts_df['creation_timestamp'] = pd.to_datetime(posts_df['creation_timestamp'])
                logging.info("Successfully converted 'creation_timestamp' in posts_df to datetime.")
            except Exception as e:
                logging.error(f"Error converting 'creation_timestamp' to datetime: {e}. Please check the format in the source CSV.")
        else:
            logging.warning("'creation_timestamp' column not found in posts_df. Skipping datetime conversion for it.")

        # Initialize sentence transformer model
        logging.info(f"Initializing sentence transformer model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)
        logging.info("Model initialized successfully.")

        # Generate embeddings for users
        logging.info("Generating embeddings for user 'about_text' and 'headline_text'")
        if 'about_text' in users_df.columns and not users_df['about_text'].isnull().all():
            users_df['about_embedding'] = users_df['about_text'].fillna('').astype(str).apply(lambda x: model.encode(x).tolist())
            logging.info("Generated 'about_embedding'.")
        else:
            logging.warning("'about_text' column not found or is empty in users.csv. Skipping 'about_embedding'.")
            users_df['about_embedding'] = [[] for _ in range(len(users_df))]


        if 'headline_text' in users_df.columns and not users_df['headline_text'].isnull().all():
            users_df['headline_embedding'] = users_df['headline_text'].fillna('').astype(str).apply(lambda x: model.encode(x).tolist())
            logging.info("Generated 'headline_embedding'.")
        else:
            logging.warning("'headline_text' column not found or is empty in users.csv. Skipping 'headline_embedding'.")
            users_df['headline_embedding'] = [[] for _ in range(len(users_df))]


        # Generate embeddings for posts
        logging.info("Generating embeddings for post 'description_text'")
        if 'description_text' in posts_df.columns and not posts_df['description_text'].isnull().all():
            posts_df['description_embedding'] = posts_df['description_text'].fillna('').astype(str).apply(lambda x: model.encode(x).tolist())
            logging.info("Generated 'description_embedding'.")
        else:
            logging.warning("'description_text' column not found or is empty in posts.csv. Skipping 'description_embedding'.")
            posts_df['description_embedding'] = [[] for _ in range(len(posts_df))]

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Save updated dataframes to Parquet
        logging.info(f"Saving users data with embeddings to {USER_OUTPUT_FILE}")
        users_df.to_parquet(USER_OUTPUT_FILE, index=False)
        logging.info(f"Saving posts data with embeddings to {POST_OUTPUT_FILE}")
        posts_df.to_parquet(POST_OUTPUT_FILE, index=False)
        logging.info("Embeddings generated and files saved successfully.")

    except FileNotFoundError as e:
        logging.error(f"Error: Input file not found. {e}")
    except Exception as e:
        logging.error(f"An error occurred during embedding generation: {e}")

if __name__ == "__main__":
    generate_embeddings()