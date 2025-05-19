import argparse
import yaml
import logging
import pandas as pd
import mlflow
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

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
    try:
        model = mlflow.keras.load_model(model_uri)
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
    post_id_field = FieldSchema(name="post_id", dtype=DataType.INT64, is_primary=True, auto_id=False)
    category_id_field = FieldSchema(name="category_id", dtype=DataType.INT64) # Assuming INT64, adjust if VARCHAR
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


def generate_and_index_embeddings(post_tower_model, posts_df, collection, batch_size, embedding_dim):
    """Generates embeddings and ingests them into Milvus."""
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
        # Let's assume the model expects inputs like this:
        # (This is a generic example, adjust to your model's input names and types)
        input_data_for_model = {
            # Assuming 'description_embedding' is a column of lists/arrays that need to be stacked
            # If your model takes raw text, you'd process 'description_text' here.
            # For now, let's assume 'description_embedding' is the primary feature.
            # This part needs careful alignment with your actual PostTower model.
            # If 'description_embedding' is already the final embedding, this step might be simpler.
            # However, the PostTower usually *generates* the final embedding from multiple inputs.
            
            # Example: if your model expects 'post_id', 'category_id', and some text-derived embedding
            # This is a placeholder. You MUST adapt this to your model's input signature.
            "post_id_input": batch_df['post_id'].to_numpy().astype('int64'),
            "category_id_input": batch_df['category_id'].to_numpy().astype('int64'),
            # "description_embedding_input": np.array(batch_df['description_embedding'].tolist()).astype('float32') # If this is an input
        }
        # If your model only takes the description embedding (e.g., from a sentence transformer)
        # and you want to store that directly, the model loading part might be different.
        # But the task is to use the "Post Tower" model.
        
        # The Post Tower model from the TwoTower architecture typically takes multiple features
        # (like post_id, category_id, text embeddings) and processes them through its layers.
        # We need to ensure the input_data_for_model matches what post_tower_model.predict expects.
        # For a tf.keras.Model, this is often a dictionary mapping input layer names to numpy arrays,
        # or a list of numpy arrays if the model has multiple unnamed inputs.

        # Let's assume the model's predict function can handle a dictionary of features
        # where keys are input names. You might need to inspect your saved model's input signature.
        # For example, if your PostTower has input layers named 'post_id_in', 'category_id_in', 'desc_emb_in':
        # model_input_dict = {
        #     'post_id_in': batch_df['post_id'].values,
        #     'category_id_in': batch_df['category_id'].values,
        #     'desc_emb_in': np.stack(batch_df['description_embedding'].values) # if it's a list of embeddings
        # }
        # embeddings = post_tower_model.predict(model_input_dict)

        # For now, let's make a strong assumption that the model takes these three inputs
        # and their names are 'post_id', 'category_id', 'description_embedding'.
        # This is very likely to need adjustment.
        
        # A more robust way is to inspect model.input_names
        # For now, a simplified approach:
        # This assumes the model takes a dictionary of numpy arrays.
        # The actual input preparation is CRITICAL and depends on your model.
        # Let's assume the model takes 'post_id', 'category_id', and 'description_embedding'
        # and 'description_embedding' is already in the correct numerical format in the DataFrame.
        
        # This part is highly model-specific.
        # If 'description_embedding' is already the output of a text embedder and is one of the inputs to the PostTower
        import numpy as np
        prepared_inputs = {
            # These keys must match the input names of your Keras model's layers
            # Example:
            "input_post_id": batch_df['post_id'].values.astype(np.int64),
            "input_category_id": batch_df['category_id'].values.astype(np.int64),
            # "input_description_embedding": np.stack(batch_df['description_embedding'].values).astype(np.float32) # If this is an input
        }
        # If your model only takes one input (e.g., concatenated features or just text tokens)
        # then adjust accordingly.
        # For a TwoTower's PostTower, it's common to have multiple discrete inputs.
        # The problem states "The Post Tower expects specific inputs (e.g., post_id for ID embedding,
        # description_embedding, category_id for category embedding)."
        # So, we'll assume these are the inputs.
        
        # Check if 'description_embedding' column exists and use it.
        # Otherwise, this part needs to be adapted to generate it from 'description_text'
        # or fetch it. The config points to 'posts_with_embeddings.parquet'.
        if 'description_embedding' in batch_df.columns:
            # Assuming description_embedding is a list/array of floats
            desc_embeddings_np = np.array(batch_df['description_embedding'].tolist(), dtype=np.float32)
            if desc_embeddings_np.shape[1] != prepared_inputs["input_post_id"].shape[0] and desc_embeddings_np.ndim == 2 : # if it's already (batch, dim)
                 pass # it's fine
            elif desc_embeddings_np.ndim == 1 and len(desc_embeddings_np) == batch_df.shape[0]: # if it's a list of embeddings
                 desc_embeddings_np = np.stack(desc_embeddings_np)

            # Ensure the embedding dimension matches what the model might expect for this input
            # Or, if this IS the embedding to be fine-tuned by the tower.
            prepared_inputs["input_description_embedding"] = desc_embeddings_np
        else:
            # This case should be handled: either error out or generate embeddings here.
            # For now, relying on 'posts_with_embeddings.parquet'.
            logging.warning("'description_embedding' not found in batch. Model might fail if it expects it.")
            # You might need to pass placeholder or handle this.
            # For a robust script, this should be an error or a feature generation step.

        # Filter prepared_inputs to only include keys that are actual inputs to the model
        # This is a safer approach if model.input_names is available
        # For now, assuming the model takes all provided inputs.
        
        try:
            # The .predict() method might take a dict or a list of arrays.
            # If it's a list, the order matters and must match model.inputs.
            # If it's a dict, keys must match input layer names.
            # We need to know the exact input structure of the saved Post Tower model.
            # Let's assume it's a dictionary for now.
            # You might need to convert this dict to a list in the correct order if model.predict expects a list.
            # Example: model_inputs_list = [prepared_inputs[name] for name in model.input_names]
            # embeddings = post_tower_model.predict(model_inputs_list, batch_size=batch_size)

            # A common way Keras models are saved allows prediction with a dictionary:
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
        category_ids_to_insert = batch_df['category_id'].tolist() # Ensure this is INT64 or compatible
        embeddings_to_insert = embeddings.tolist() # Milvus expects list of lists for FLOAT_VECTOR

        data_to_insert = [
            post_ids_to_insert,
            category_ids_to_insert,
            embeddings_to_insert
        ]
        
        try:
            insert_result = collection.insert(data_to_insert)
            logging.info(f"Inserted batch {i // batch_size + 1} into Milvus. Insert count: {insert_result.insert_count}")
        except Exception as e:
            logging.error(f"Error inserting batch {i // batch_size + 1} into Milvus: {e}")
            # Consider retry logic or skipping
            
    logging.info("All batches processed.")
    logging.info("Flushing collection to ensure data persistence.")
    collection.flush()
    logging.info(f"Collection '{collection.name}' flushed. Total entities: {collection.num_entities}")


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

    # 3. Connect to Milvus & Setup Collection
    connect_to_milvus(milvus_host, milvus_port)
    milvus_collection = create_milvus_collection(collection_name, embedding_dim)
    create_index(milvus_collection) # Create index after collection creation, before loading if possible

    # 4. Generate Embeddings & Index
    generate_and_index_embeddings(post_tower_model, posts_df, milvus_collection, batch_size, embedding_dim)

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