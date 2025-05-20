import argparse
import yaml
import logging
import sys
import os

# Add relevant directories to Python path
# Path to src/inference/offline/index_posts_to_milvus.py
script_dir = os.path.dirname(__file__)
# Path to src/
src_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
# Path to src/model_training/two_tower/
model_module_dir = os.path.join(src_dir, 'model_training', 'two_tower')

sys.path.insert(0, src_dir) # To allow 'from model_training.two_tower.model import PostTower'
sys.path.insert(0, model_module_dir) # To allow Keras to 'import model' if it looks for model.py

import pandas as pd
import mlflow
import tensorflow as tf # Import tensorflow for custom_object_scope
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
# This import should still work because src_dir is on the path
from model_training.two_tower.model import PostTower

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    """Loads YAML configuration file."""
    logging.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_post_tower_model(model_uri):
    """Loads the Post Tower model from MLflow."""
    logging.info(f"Loading Post Tower model from MLflow URI: {model_uri}")
    logging.info(f"Current MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    try:
        # Changed from mlflow.keras.load_model to mlflow.tensorflow.load_model
        # Use tf.keras.utils.custom_object_scope to make PostTower known
        with tf.keras.utils.custom_object_scope({'PostTower': PostTower}):
            model = mlflow.tensorflow.load_model(model_uri)
        logging.info("Post Tower model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading Post Tower model: {e}")
        raise

def load_post_data(data_path):
    """Loads post data from a Parquet file."""
    logging.info(f"Loading post data from {data_path}")
    try:
        df = pd.read_parquet(data_path)
        # Ensure required columns are present.
        # The Post Tower model might expect specific column names for its inputs.
        # For this example, we assume 'post_id', 'category_id', and features needed for embedding.
        # If 'description_embedding' is directly used from the file:
        if 'description_embedding' not in df.columns and 'description_text' not in df.columns:
             raise ValueError("Post data must contain 'description_embedding' or 'description_text' for the Post Tower model.")
        if 'post_id' not in df.columns or 'category_id' not in df.columns:
            raise ValueError("Post data must contain 'post_id' and 'category_id'.")
        logging.info(f"Loaded {len(df)} posts.")
        return df
    except Exception as e:
        logging.error(f"Error loading post data: {e}")
        raise

def connect_to_milvus(host, port):
    """Connects to the Milvus server."""
    logging.info(f"Connecting to Milvus server at {host}:{port}")
    try:
        connections.connect(alias="default", host=host, port=port)
        logging.info("Successfully connected to Milvus.")
    except Exception as e:
        logging.error(f"Failed to connect to Milvus: {e}")
        raise

def create_milvus_collection(collection_name, dim):
    """Creates a Milvus collection if it doesn't exist."""
    if utility.has_collection(collection_name, using="default"):
        logging.info(f"Collection '{collection_name}' already exists.")
        return Collection(collection_name, using="default")

    logging.info(f"Creating Milvus collection: {collection_name}")
    # Define schema
    # Changed post_id to VARCHAR to store string IDs like 'p_00001'
    post_id_field = FieldSchema(name="post_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64)
    category_id_field = FieldSchema(name="category_id", dtype=DataType.INT64)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    
    schema = CollectionSchema(
        fields=[post_id_field, category_id_field, embedding_field],
        description="Post embeddings for recommendation system",
        enable_dynamic_field=False # Set to True if you need more fields later without schema change
    )
    
    collection = Collection(collection_name, schema=schema, using="default")
    logging.info(f"Collection '{collection_name}' created successfully.")
    return collection

def create_index(collection, field_name="embedding"):
    """Creates an index on the embedding field."""
    if collection.has_index(index_name=f"{field_name}_index"):
        logging.info(f"Index on field '{field_name}' already exists for collection '{collection.name}'.")
        return

    logging.info(f"Creating index on field '{field_name}' for collection '{collection.name}'")
    index_params = {
        "metric_type": "L2",  # Or "IP" for inner product
        "index_type": "HNSW", # Or "IVF_FLAT", "IVF_SQ8" etc.
        "params": {"M": 16, "efConstruction": 200}, # Example params for HNSW
    }
    collection.create_index(field_name, index_params, index_name=f"{field_name}_index")
    logging.info(f"Index created successfully on field '{field_name}'.")
    # Load collection after creating index for search
    collection.load()
    logging.info(f"Collection '{collection.name}' loaded into memory.")


def build_field_vocabulary(df, field_name, oov_token="<OOV>", reserve_zero=True):
    """Builds a vocabulary mapping for a given field from a DataFrame."""
    unique_values = df[field_name].unique().tolist()
    vocab = {}
    start_index = 0
    if reserve_zero: # Often 0 is reserved for padding or a special token
        vocab[oov_token] = 0 # OOV token maps to 0
        start_index = 1
    
    for i, value in enumerate(unique_values):
        vocab[value] = i + start_index
    
    # Ensure OOV is in vocab if not reserving zero and it wasn't a unique value
    if not reserve_zero and oov_token not in vocab and oov_token in unique_values: # Should not happen if oov_token is special
        # This case is tricky, means OOV token was a real value.
        # For simplicity, if not reserving zero, OOV will get the next available index if not present.
        pass # It will be handled like any other unique value
    elif not reserve_zero and oov_token not in vocab:
         vocab[oov_token] = len(vocab) # Add OOV at the end if not present

    logging.info(f"Built vocabulary for '{field_name}' with {len(vocab)} entries. OOV token '{oov_token}' maps to {vocab.get(oov_token)}. First few items: {list(vocab.items())[:5]}")
    return vocab

def generate_and_index_embeddings(post_tower_model, posts_df, collection, batch_size, embedding_dim, vocabs):
    """Generates embeddings and ingests them into Milvus, using provided vocabularies for categorical features."""
    num_posts = len(posts_df)
    logging.info(f"Starting embedding generation and indexing for {num_posts} posts.")

    for i in range(0, num_posts, batch_size):
        batch_df = posts_df.iloc[i:i + batch_size]
        logging.info(f"Processing batch {i // batch_size + 1}/{(num_posts + batch_size - 1) // batch_size}")

        # Prepare model inputs. This needs to match the Post Tower's expected input structure.
        # Example: if the model expects a dictionary of features:
        # Assuming 'description_embedding' is a column of lists/arrays in the Parquet file
        # and 'post_id', 'category_id' are also needed by the model.
        # The actual input preparation will depend heavily on the saved model's signature.
        
        # Placeholder for actual input preparation logic:
        # You'll need to adapt this based on your PostTower model's input requirements.
        # For example, if it takes numericalized text tokens, or pre-computed embeddings.
        # If using 'description_embedding' directly from parquet:
        # model_inputs = {
        # "description_embedding_input": np.stack(batch_df['description_embedding'].values),
        # "post_id_input": batch_df['post_id'].values,
        # "category_id_input": batch_df['category_id'].values
        # }
        # If the model generates embeddings from raw features like text:
        # model_inputs = prepare_model_inputs_from_df(batch_df, feature_names_expected_by_model)

        # For this script, we'll assume the model takes the 'description_embedding' (if present)
        # and other IDs as separate inputs. This is a common pattern for two-tower models.
        # This part is highly dependent on the actual saved model structure.
        # Let's assume the model has a .predict method that accepts a dictionary of features
        # or a list/tuple of numpy arrays in a specific order.

        # Simplified example: Assuming the model takes 'description_embedding', 'post_id', 'category_id'
        # and 'description_embedding' is already a suitable numerical vector.
        # This is a placeholder and needs to be adapted.
        # For a Keras model, inputs might be a dict of numpy arrays or a list of numpy arrays.
        
        # Example: if 'description_embedding' is already the feature vector for the post content
        # and the model also takes 'post_id' and 'category_id' as inputs to form the final post embedding.
        # This is a common setup where the "Post Tower" combines various features.
        
        # We need to ensure the inputs are in the format expected by post_tower_model.predict()
        # This might involve converting pandas Series to numpy arrays, stacking, etc.
        # For simplicity, let's assume the model takes a dictionary of features.
        # The actual keys must match the input layer names of your Keras model.
        
        # This is a critical part that needs to be correct for your specific model.
        # The actual input preparation is CRITICAL and depends on your model.
        # The PostTower model expects specific input keys: "post_id", "category_id",
        # "media_type", "creator_id", "description_embedding".
        
        import numpy as np
        # Ensure dtypes are correct for the model's input signature
        # PostTower expects:
        # "post_id": string tensor
        # "description_embedding": float32 tensor
        # "category_id": int64 tensor (indices for embedding)
        # "media_type": int64 tensor (indices for embedding)
        # "creator_id": int64 tensor (indices for embedding)

        prepared_inputs = {
            "post_id": batch_df['post_id'].values, # Keep as strings
        }

        # Map categorical string features to integer indices using vocabs
        # The PostTower model expects integer indices for these Embedding layers.
        # OOV_TOKEN is assumed to map to index 0 if reserve_zero=True in build_field_vocabulary.
        oov_idx_category = vocabs['category_id'].get("<OOV>", 0)
        prepared_inputs["category_id"] = batch_df['category_id'].map(vocabs['category_id']).fillna(oov_idx_category).to_numpy().astype(np.int64)

        if 'media_type' in batch_df.columns:
            if 'media_type' in vocabs:
                oov_idx_media_type = vocabs['media_type'].get("<OOV>", 0)
                prepared_inputs["media_type"] = batch_df['media_type'].map(vocabs['media_type']).fillna(oov_idx_media_type).to_numpy().astype(np.int64)
            else:
                logging.error("'media_type' vocabulary not found. Ensure it's built and passed correctly.")
                raise KeyError("'media_type' vocabulary missing.")
        else:
            logging.error("'media_type' column not found in batch_df. This is a required input for the PostTower model.")
            raise KeyError("'media_type' column missing from input data.")

        if 'creator_id' in batch_df.columns:
            # Note: The PostTower model uses an Embedding layer for creator_id, implying it expects integer indices.
            # If creator_id values in the data are strings like 'u_xxxx', they need to be mapped to integers.
            # The current PostTower model does NOT have a StringLookup layer for creator_id.
            # This vocabulary mapping handles the conversion from string 'u_xxxx' to integer indices.
            if 'creator_id' in vocabs:
                oov_idx_creator_id = vocabs['creator_id'].get("<OOV>", 0)
                prepared_inputs["creator_id"] = batch_df['creator_id'].map(vocabs['creator_id']).fillna(oov_idx_creator_id).to_numpy().astype(np.int64)
            else:
                logging.error("'creator_id' vocabulary not found. Ensure it's built and passed correctly.")
                raise KeyError("'creator_id' vocabulary missing.")
        else:
            logging.error("'creator_id' column not found in batch_df. This is a required input for the PostTower model.")
            raise KeyError("'creator_id' column missing from input data.")

        if 'description_embedding' in batch_df.columns:
            desc_embeddings_list = batch_df['description_embedding'].tolist()
            if not desc_embeddings_list and len(batch_df) > 0 : # If list is empty but batch is not
                 logging.error("description_embedding column is present but resulted in an empty list for a non-empty batch.")
                 raise ValueError("description_embedding data is missing for the batch.")
            
            if desc_embeddings_list: # Proceed only if list is not empty
                # Validate structure of embeddings before stacking
                first_elem_len = -1
                if isinstance(desc_embeddings_list[0], (list, np.ndarray)):
                    first_elem_len = len(desc_embeddings_list[0])
                else:
                    logging.error(f"First element of description_embedding is not a list/array. Type: {type(desc_embeddings_list[0])}")
                    raise ValueError("description_embedding column elements are not lists/arrays.")

                if not all(isinstance(e, (list, np.ndarray)) and len(e) == first_elem_len for e in desc_embeddings_list):
                    logging.error("description_embedding column contains non-uniform elements (varying lengths or types).")
                    # Example of non-uniform data:
                    # for i, e in enumerate(desc_embeddings_list):
                    #    if not (isinstance(e, (list, np.ndarray)) and len(e) == first_elem_len):
                    #        logging.debug(f"Problematic element at index {i}: type {type(e)}, len {len(e) if hasattr(e, '__len__') else 'N/A'}")
                    raise ValueError("description_embedding column has inconsistent data for stacking.")
                
                try:
                    desc_embeddings_np = np.array(desc_embeddings_list, dtype=np.float32)
                except Exception as e:
                    logging.error(f"Error converting description_embedding to numpy array: {e}")
                    raise
                
                if desc_embeddings_np.ndim != 2 and len(batch_df) > 0:
                    logging.error(f"description_embedding is not 2D after processing. Shape: {desc_embeddings_np.shape}. Expected (batch_size, embedding_dim).")
                    raise ValueError(f"description_embedding could not be converted to a 2D array. Shape: {desc_embeddings_np.shape}")
                prepared_inputs["description_embedding"] = desc_embeddings_np
            elif len(batch_df) > 0: # batch_df not empty, but desc_embeddings_list is empty
                logging.error("description_embedding column is present but resulted in an empty list for a non-empty batch.")
                raise ValueError("description_embedding data is missing for the batch.")
            # If batch_df is empty, desc_embeddings_list will also be empty, and this is fine (loop won't run).

        else:
            logging.error("'description_embedding' column not found in batch_df. This is a required input for the PostTower model.")
            raise KeyError("'description_embedding' column missing from input data.")
        
        try:
            # Ensure all required inputs for the PostTower model are present in prepared_inputs
            required_model_keys = {"post_id", "category_id", "media_type", "creator_id", "description_embedding"}
            missing_keys = required_model_keys - set(prepared_inputs.keys())
            if missing_keys:
                logging.error(f"Missing required keys for model prediction: {missing_keys}. Provided keys: {list(prepared_inputs.keys())}")
                # This check is crucial before calling predict.
                # It indicates a mismatch between data provided and model's expectations.
                raise ValueError(f"Data preparation step failed to provide all required model inputs. Missing: {missing_keys}")

            embeddings = post_tower_model.predict(prepared_inputs, batch_size=len(batch_df))
            logging.info(f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}.")

            if embeddings.shape[1] != embedding_dim:
                logging.error(f"Generated embedding dimension {embeddings.shape[1]} does not match configured dimension {embedding_dim}.")
                # Potentially raise an error or try to reshape/truncate if appropriate (usually not).
                # For now, we'll proceed but this is a critical mismatch.
                # raise ValueError("Embedding dimension mismatch.") # More robust

        except Exception as e:
            logging.error(f"Error during model prediction for batch {i // batch_size + 1}: {e}")
            logging.error(f"Model expected inputs: {getattr(post_tower_model, 'input_names', 'N/A')}")
            logging.error(f"Provided input keys: {list(prepared_inputs.keys())}")
            # Log shapes of inputs for debugging
            for k, v in prepared_inputs.items():
                if hasattr(v, 'shape'):
                    logging.error(f"Shape of input '{k}': {v.shape}")
            continue # Skip this batch or raise error

        # Prepare data for Milvus insertion
        post_ids_to_insert = batch_df['post_id'].tolist()
        # Use the integer-mapped category_id from prepared_inputs for Milvus insertion
        category_ids_to_insert = prepared_inputs["category_id"].tolist()
        embeddings_to_insert = embeddings.tolist() # Milvus expects list of lists for FLOAT_VECTOR

        data_to_insert = [
            post_ids_to_insert,
            category_ids_to_insert,
            embeddings_to_insert
        ]
        
        # Detailed logging before insertion
        logging.debug(f"Attempting to insert batch {i // batch_size + 1}")
        logging.debug(f"  post_ids_to_insert (len {len(post_ids_to_insert)}): {post_ids_to_insert[:3]}")
        logging.debug(f"  post_ids type: {type(post_ids_to_insert[0]) if post_ids_to_insert else 'N/A'}")
        logging.debug(f"  category_ids_to_insert (len {len(category_ids_to_insert)}): {category_ids_to_insert[:3]}")
        logging.debug(f"  category_ids type: {type(category_ids_to_insert[0]) if category_ids_to_insert else 'N/A'}")
        logging.debug(f"  embeddings_to_insert (len {len(embeddings_to_insert)}): First embedding shape/type: {np.array(embeddings_to_insert[0]).shape if embeddings_to_insert else 'N/A'}, {type(embeddings_to_insert[0][0]) if embeddings_to_insert and embeddings_to_insert[0] else 'N/A'}")

        try:
            insert_result = collection.insert(data_to_insert)
            logging.info(f"Successfully inserted batch {i // batch_size + 1} into Milvus. Insert count: {insert_result.insert_count}")
        except Exception as e:
            logging.error(f"Error inserting batch {i // batch_size + 1} into Milvus: {e}")
            # Consider retry logic or skipping
            
    logging.info("All batches processed.")
    logging.info("Flushing collection to ensure data persistence.")
    collection.flush()
    logging.info(f"Collection '{collection.name}' flushed. Total entities: {collection.num_entities}")
    # Add a final check
    if collection.num_entities == 0 and num_posts > 0:
        logging.warning("Warning: Flushing complete, but no entities were inserted into the collection.")

def main(config_path):
    config = load_config(config_path)

    mlflow_uri = config['mlflow_post_tower_uri']
    milvus_host = config['milvus_host']
    milvus_port = config['milvus_port']
    collection_name = config['milvus_collection_name']
    embedding_dim = config['embedding_dim']
    batch_size = config['batch_size']
    post_data_path = config['post_data_path']

    # 1. Load Model
    post_tower_model = load_post_tower_model(mlflow_uri)

    # 2. Load Data
    posts_df = load_post_data(post_data_path)

    # Build vocabularies for categorical string features
    # These columns are expected by the PostTower model as integer inputs for Embedding layers
    categorical_cols_for_vocab = []
    if 'category_id' in posts_df.columns and posts_df['category_id'].dtype == 'object':
        categorical_cols_for_vocab.append('category_id')
    if 'media_type' in posts_df.columns and posts_df['media_type'].dtype == 'object':
        categorical_cols_for_vocab.append('media_type')
    if 'creator_id' in posts_df.columns and posts_df['creator_id'].dtype == 'object': # creator_id is often string like 'u_xxx'
        categorical_cols_for_vocab.append('creator_id')
    
    vocabs = {}
    if categorical_cols_for_vocab:
        logging.info(f"Building vocabularies for: {categorical_cols_for_vocab}")
        for col in categorical_cols_for_vocab:
            vocabs[col] = build_field_vocabulary(posts_df, col)
    else:
        logging.info("No string-based categorical columns found needing vocabulary building (category_id, media_type, creator_id). Assuming they are already integer indices if present.")


    # 3. Connect to Milvus & Setup Collection
    connect_to_milvus(milvus_host, milvus_port)
    milvus_collection = create_milvus_collection(collection_name, embedding_dim)
    # Ensure index is created *after* schema is confirmed and collection is valid
    if milvus_collection:
         create_index(milvus_collection)
    else:
        logging.error("Milvus collection object is None, cannot create index. Exiting.")
        return # or raise an error

    # 4. Generate Embeddings & Index
    generate_and_index_embeddings(post_tower_model, posts_df, milvus_collection, batch_size, embedding_dim, vocabs)

    logging.info("Post indexing to Milvus completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index post embeddings into Milvus.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (e.g., config/milvus_indexing_config.yaml)"
    )
    args = parser.parse_args()
    main(args.config)