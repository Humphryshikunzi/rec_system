import pandas as pd
import tensorflow as tf
from feast import FeatureStore
from datetime import datetime, timedelta
import numpy as np
import os

def get_all_ids_from_interactions(interactions_file_path):
    """Reads interaction data and returns unique user and post IDs."""
    if not os.path.exists(interactions_file_path):
        raise FileNotFoundError(f"Interactions file not found: {interactions_file_path}")
    interactions_df = pd.read_csv(interactions_file_path)
    print(f"DEBUG: Columns in interactions_df: {interactions_df.columns.tolist()}")
    user_ids = interactions_df['user_id'].unique().tolist()
    post_ids = interactions_df['post_id'].unique().tolist()
    return user_ids, post_ids, interactions_df

def get_positive_and_negative_pairs(interactions_df, all_post_ids_list):
    """
    Generates positive and negative pairs for training.
    Positive pairs: 'like', 'bookmark', 'view_full'.
    Negative pairs: For each user, sample posts they haven't interacted with positively.
    """
    positive_interactions = ['like', 'bookmark', 'view_full']
    
    # Positive pairs
    positive_pairs_df = interactions_df[interactions_df['interaction_type'].isin(positive_interactions)][['user_id', 'post_id']].drop_duplicates()
    positive_pairs_df.rename(columns={'user_id': 'uid'}, inplace=True) # Rename to 'uid'
    print(f"DEBUG: Columns in positive_pairs_df after rename: {positive_pairs_df.columns.tolist()}")
    print(f"DEBUG: positive_pairs_df head:\n{positive_pairs_df.head()}")
    positive_pairs_df['label'] = 1

    # Negative pairs
    all_users = interactions_df['user_id'].unique() # From original interactions_df
    
    # positive_pairs_df now uses 'uid'
    positive_interaction_tuples = set(tuple(x) for x in positive_pairs_df[['uid', 'post_id']].values)

    negative_pairs_list = []
    num_negative_samples_per_user = 5

    for user_id in all_users: # user_id here is the value from all_users
        # Filter positive_pairs_df (which uses 'uid') using the current user_id
        user_positive_posts = set(positive_pairs_df[positive_pairs_df['uid'] == user_id]['post_id'])
        
        potential_negatives = [pid for pid in all_post_ids_list if pid not in user_positive_posts]
        
        num_to_sample = min(len(potential_negatives), num_negative_samples_per_user * len(user_positive_posts))
        if num_to_sample == 0 and len(user_positive_posts) > 0:
             num_to_sample = min(len(all_post_ids_list), num_negative_samples_per_user * 1)
             potential_negatives = all_post_ids_list

        if len(potential_negatives) > 0:
            sampled_negatives = np.random.choice(potential_negatives, size=num_to_sample, replace=False)
            for neg_post_id in sampled_negatives:
                # Store with 'uid' for consistency
                negative_pairs_list.append({'uid': user_id, 'post_id': neg_post_id, 'label': 0})

    negative_pairs_df = pd.DataFrame(negative_pairs_list)
    print(f"DEBUG: Columns in negative_pairs_df: {negative_pairs_df.columns.tolist()}")
    print(f"DEBUG: negative_pairs_df head:\n{negative_pairs_df.head()}")
    
    training_df = pd.concat([positive_pairs_df, negative_pairs_df], ignore_index=True)
    training_df = training_df.sample(frac=1).reset_index(drop=True) # Shuffle
    
    return training_df

def fetch_features_for_training(store: FeatureStore, training_df: pd.DataFrame, user_features_list: list, post_features_list: list):
    """
    Fetches features for user and post IDs from Feast.
    Assumes training_df has 'uid', 'post_id', and 'event_timestamp' (or we generate one).
    """
    # Feast expects entity dataframes with 'event_timestamp' and entity keys
    # We'll create a single entity_df for all lookups.
    # For simplicity, using current time as event_timestamp. In a real scenario,
    # this should be the time of the interaction or when features were known.
    
    # Create unique user and post dataframes for feature fetching
    unique_user_df = training_df[['uid']].drop_duplicates().reset_index(drop=True)
    unique_user_df['event_timestamp'] = pd.to_datetime(datetime.now()) # Use current time
    unique_user_df.rename(columns={'uid': 'user_id'}, inplace=True) # Match entity name if different

    unique_post_df = training_df[['post_id']].drop_duplicates().reset_index(drop=True)
    unique_post_df['event_timestamp'] = pd.to_datetime(datetime.now()) # Use current time
    # unique_post_df.rename(columns={'post_id': 'post_id'}, inplace=True) # Match entity name if different

    print(f"Fetching user features for {len(unique_user_df)} users.")
    print(f"User entity DF head:\n{unique_user_df.head()}")
    user_feature_vector = store.get_historical_features(
        entity_df=unique_user_df,
        features=user_features_list
    ).to_df()
    print(f"Fetched user features shape: {user_feature_vector.shape}")
    print(f"User features head:\n{user_feature_vector.head()}")


    print(f"Fetching post features for {len(unique_post_df)} posts.")
    print(f"Post entity DF head:\n{unique_post_df.head()}")
    post_feature_vector = store.get_historical_features(
        entity_df=unique_post_df,
        features=post_features_list
    ).to_df()
    print(f"Fetched post features shape: {post_feature_vector.shape}")
    print(f"Post features head:\n{post_feature_vector.head()}")

    # Merge features back into the training_df
    # Rename columns in feature vectors to avoid clashes and for clarity
    user_feature_vector.rename(columns={'user_id': 'uid'}, inplace=True) # Match training_df key
    
    # Ensure correct merging keys and handle potential missing features (fillna)
    training_data_with_features = pd.merge(training_df, user_feature_vector, on=['uid'], how='left')
    training_data_with_features = pd.merge(training_data_with_features, post_feature_vector, on=['post_id'], how='left')
    
    # Drop rows where features could not be fetched (e.g., new IDs not yet in Feast)
    # Or handle imputation. For now, dropping.
    # Also drop event_timestamp_x, event_timestamp_y if they exist from merge
    cols_to_drop = [col for col in training_data_with_features.columns if 'event_timestamp' in col and col not in ['event_timestamp_x', 'event_timestamp_y']]
    training_data_with_features.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    if 'event_timestamp_x' in training_data_with_features.columns:
        training_data_with_features.drop(columns=['event_timestamp_x'], inplace=True, errors='ignore')
    if 'event_timestamp_y' in training_data_with_features.columns:
        training_data_with_features.drop(columns=['event_timestamp_y'], inplace=True, errors='ignore')


    # Handle NaNs - crucial for embeddings and IDs
    # For ID features that will go into Embedding layers, they must be integers and not NaN.
    # For pre-computed embeddings (like 'about_embedding'), NaNs might mean missing data.
    # A common strategy is to fill with zeros or a special "missing" embedding.
    
    # Example: Fill NaN embeddings with zeros (assuming embeddings are lists/arrays)
    embedding_cols_user = [f.split(':')[1] for f in user_features_list if 'embedding' in f]
    embedding_cols_post = [f.split(':')[1] for f in post_features_list if 'embedding' in f]

    for col in embedding_cols_user + embedding_cols_post:
        if col in training_data_with_features.columns:
            # Assuming embeddings are stored as list/array string representations or actual lists/arrays
            # If they are strings like "[0.1, 0.2, ...]", they need parsing first.
            # For simplicity, assuming they are already in a usable format or Feast handles this.
            # If they are lists/np.arrays, fillna with a list of zeros of the correct dimension.
            # This is a placeholder; actual handling depends on how embeddings are stored and retrieved.
            # Example: training_data_with_features[col] = training_data_with_features[col].apply(lambda x: x if isinstance(x, (list, np.ndarray)) else np.zeros(embedding_dim))
            # A simpler fill for now if they are object types that might contain NaNs:
            is_object_dtype = training_data_with_features[col].dtype == 'object'
            if is_object_dtype:
                 # Attempt to infer embedding dimension from the first valid entry
                first_valid_embedding = training_data_with_features[col].dropna().iloc[0] if not training_data_with_features[col].dropna().empty else None
                if first_valid_embedding is not None and hasattr(first_valid_embedding, '__len__'):
                    emb_dim = len(first_valid_embedding)
                    training_data_with_features[col] = training_data_with_features[col].apply(lambda x: np.zeros(emb_dim) if not isinstance(x, (list, np.ndarray)) else x)
                else: # Fallback if cannot determine dim or not list/array
                    print(f"Warning: Could not determine embedding dimension for {col} to fill NaNs. Filling with scalar 0.")
                    training_data_with_features[col] = training_data_with_features[col].fillna(0) # Or handle more robustly
            else: # For numerical arrays that might have NaN elements (less common for whole array to be NaN)
                # If embeddings are split into multiple columns (e.g., emb_0, emb_1, ...), fillna on those.
                # If it's a single column with array objects, the above 'object' check should handle it.
                pass # Assuming numerical arrays are handled by Feast or are dense.

        if 'description_embedding' in training_data_with_features.columns and not training_data_with_features.empty:
            print(f"DEBUG data_utils: After embedding NaN handling, for 'description_embedding': type {training_data_with_features['description_embedding'].dtype}, example value: {training_data_with_features['description_embedding'].iloc[0] if not training_data_with_features.empty and len(training_data_with_features['description_embedding'].iloc[0]) > 0 else 'N/A or empty list'}")


    # For categorical ID features that need to be integers for Embedding layers:
    # e.g., 'category_id', 'media_type', 'creator_id'
    # These should be mapped to integer indices if they are not already.
    # Feast might return them as original values (strings, numbers).
    # The StringLookup layer in the model handles string inputs, but requires a vocabulary.
    # If they are already integer IDs, ensure they are filled if NaN (e.g., with a 0 for "unknown").
    categorical_id_cols = ['category_id', 'media_type', 'creator_id'] # Add others as needed
    for col in categorical_id_cols:
        if col in training_data_with_features.columns:
            # Ensure column is string type first to handle mixed types or numeric strings
            training_data_with_features[col] = training_data_with_features[col].astype(str)

            # Get unique string categories. Sort for consistent mapping if desired, though not strictly necessary for functionality.
            # NaNs will be handled by fillna(0) at the end.
            unique_categories = sorted(list(training_data_with_features[col].dropna().unique()))
            
            # Create mapping: string category -> integer index (start from 1)
            # 0 will be reserved for NaNs or unseen categories.
            mapping = {category_str: i for i, category_str in enumerate(unique_categories, start=1)}
            
            # Apply mapping.
            # Original NaNs (now strings like 'nan') or strings not in mapping will become NaN after .map().
            training_data_with_features[col] = training_data_with_features[col].map(mapping)
            
            # Fill these NaNs (from original NaNs or unmapped values) with 0.
            # Then, convert the entire column to int. All values should now be integers.
            training_data_with_features[col] = training_data_with_features[col].fillna(0).astype(int)
        
        for cat_col_debug in categorical_id_cols: # Renamed loop variable for clarity
            if cat_col_debug in training_data_with_features.columns and not training_data_with_features.empty:
                print(f"DEBUG data_utils: After categorical processing, for '{cat_col_debug}': type {training_data_with_features[cat_col_debug].dtype}, example value: {training_data_with_features[cat_col_debug].iloc[0]}")
            elif cat_col_debug in training_data_with_features.columns and training_data_with_features.empty:
                 print(f"DEBUG data_utils: After categorical processing, for '{cat_col_debug}': DataFrame is empty.")


    # Ensure user_id and post_id are present and correctly typed
    # Ensure user_id and post_id are present (string IDs are fine here, model handles lookup)
    # training_data_with_features['uid'] = training_data_with_features['uid'].astype(int) # Removed: Model expects string IDs for lookup
    # training_data_with_features['post_id'] = training_data_with_features['post_id'].astype(int) # Removed: Model expects string IDs for lookup


    print(f"Training data with features shape: {training_data_with_features.shape}")
    print(f"Training data with features head:\n{training_data_with_features.head(2)}")
    print(f"Columns: {training_data_with_features.columns}")
    # Check for NaNs after processing
    print(f"NaNs per column:\n{training_data_with_features.isnull().sum()}")
    
    # Drop rows with any remaining NaNs in critical feature columns
    # This depends on which columns are absolutely necessary.
    # For example, if an embedding is missing, that row might be unusable.
    # critical_feature_cols = ['uid', 'post_id'] + [f.split(':')[1] for f in user_features_list + post_features_list]
    # training_data_with_features.dropna(subset=critical_feature_cols, inplace=True)
    # A simpler approach: drop if any feature is NaN after our fillna attempts
    training_data_with_features.dropna(inplace=True)
    print(f"Shape after final dropna: {training_data_with_features.shape}")


    return training_data_with_features


def create_tf_dataset(df: pd.DataFrame, user_feature_names: list, post_feature_names: list, batch_size: int):
    """
    Creates a tf.data.Dataset from the pandas DataFrame.
    Separates features for user tower, post tower, and labels.
    """
    
    # Extract actual feature names (after ':')
    raw_user_feature_keys = [f.split(':')[1] for f in user_feature_names]
    raw_post_feature_keys = [f.split(':')[1] for f in post_feature_names]

    # Add ID features that are directly used by towers but not in feature_store lists
    # (e.g., user_id, post_id if they are inputs to embedding layers)
    # The model expects 'user_id' in user_inputs and 'post_id' in post_inputs
    
    user_inputs = {}
    user_inputs['user_id'] = df['uid'].values # uid from df corresponds to user_id for the model
    for key in raw_user_feature_keys:
        if key in df.columns:
            # Embeddings might be lists/arrays; stack them correctly for TF
            if isinstance(df[key].iloc[0], (list, np.ndarray)):
                user_inputs[key] = np.stack(df[key].values)
            else:
                user_inputs[key] = df[key].values
        else:
            print(f"Warning: User feature '{key}' not found in DataFrame columns: {df.columns}")


    post_inputs = {}
    post_inputs['post_id'] = df['post_id'].values # post_id from df for the model
    # Categorical features for post tower (expected by model.py)
    # These might not be in `post_feature_names` if they are IDs directly used.
    # Ensure they are present in the df.
    categorical_post_features = ['category_id', 'media_type', 'creator_id'] 
    for key in raw_post_feature_keys + categorical_post_features:
        if key in df.columns:
            if key not in post_inputs: # Avoid overwriting post_id if it's also in raw_post_feature_keys
                if isinstance(df[key].iloc[0], (list, np.ndarray)):
                    post_inputs[key] = np.stack(df[key].values)
                else:
                    post_inputs[key] = df[key].values
        elif key in categorical_post_features: # If it's a known categorical but missing, raise error or fill
             raise ValueError(f"Critical post categorical feature '{key}' not found in DataFrame columns: {df.columns}. Check feature fetching and naming.")
        else:
            print(f"Warning: Post feature '{key}' not found in DataFrame columns: {df.columns}")


    # Ensure all expected features by the model are present in user_inputs and post_inputs
    # Model's UserTower expects: "user_id", "about_embedding", "headline_embedding"
    # Model's PostTower expects: "post_id", "description_embedding", "category_id", "media_type", "creator_id"
    
    required_user_model_keys = ["user_id", "about_embedding", "headline_embedding"]
    for r_key in required_user_model_keys:
        if r_key not in user_inputs:
            raise ValueError(f"Missing required user feature for model: {r_key}. Available: {user_inputs.keys()}")

    required_post_model_keys = ["post_id", "description_embedding", "category_id", "media_type", "creator_id"]
    for r_key in required_post_model_keys:
        if r_key not in post_inputs:
            raise ValueError(f"Missing required post feature for model: {r_key}. Available: {post_inputs.keys()}")


    # For tfrs.tasks.Retrieval, the label is implicitly handled by providing (query, positive_candidate) pairs.
    # If using a different loss (e.g., binary crossentropy on scores), you'd need labels.
    # For now, let's assume we are preparing data for tfrs.tasks.Retrieval,
    # which expects a dictionary of features that can be split into query and candidate.
    
    # The model.py TwoTowerModel expects {'user_inputs': ..., 'post_inputs': ...}
    # If using tfrs.Model, the compute_loss will take this dictionary.
    
    dataset_dict = {
        "user_inputs": user_inputs,
        "post_inputs": post_inputs
    }
    
    # If we were to use labels explicitly (e.g. for a pointwise loss):
    # labels = df['label'].values
    # dataset = tf.data.Dataset.from_tensor_slices((dataset_dict, labels))
    
    # For tfrs.tasks.Retrieval, it typically expects all features in one dict per example.
    # The tfrs.Model's compute_loss then extracts query and candidate parts.
    # So, we need to structure it such that each element of the dataset is a single dictionary
    # containing all necessary parts.
    
    # Let's reformat for TFRS: features are (user_id, post_id, other_user_features, other_post_features)
    # The TFRS model's `compute_loss` will then map these to user_tower and post_tower inputs.
    
    # For simplicity with the current TwoTowerModel structure (not the TFRS one yet),
    # the input to model.call is {'user_inputs': user_data_dict, 'post_inputs': post_data_dict}
    # So, tf.data.Dataset should yield elements of this structure.

    # We need to convert the dict of arrays into a dataset of dicts of scalars/vectors per example
    # This means user_inputs should be a dict of tensors, post_inputs a dict of tensors.
    # tf.data.Dataset.from_tensor_slices will handle this if given a dict of dicts where the inner dicts
    # have tensor-like objects (numpy arrays, lists of numbers).

    # Check shapes
    # num_examples = len(df)
    # for k, v in user_inputs.items():
    #     assert v.shape[0] == num_examples, f"Shape mismatch for user_input {k}: {v.shape[0]} vs {num_examples}"
    # for k, v in post_inputs.items():
    #     assert v.shape[0] == num_examples, f"Shape mismatch for post_input {k}: {v.shape[0]} vs {num_examples}"


    # This structure is for the model that takes {'user_inputs': ..., 'post_inputs': ...}
    # If using TFRS, the dataset might be flatter, e.g., {'user_id': ..., 'post_id': ..., ...}
    # and the TFRS model's compute_loss would internally create the tower inputs.
    # For now, aligning with TwoTowerModel's call signature:
    
    # Create a dataset where each element is a tuple: ( (user_features_dict, post_features_dict), label )
    # Or, if using TFRS style, just a dict of all features, and TFRS task handles labels implicitly.
    # Let's assume the model will be wrapped in a TFRS model for training.
    # TFRS Retrieval task expects query embeddings and candidate embeddings.
    # The input to TFRS model's compute_loss is usually a dictionary of all features for a (query, candidate) pair.
    
    # Example:
    # features = {
    #    "user_id": df['uid'].values,
    #    "about_embedding": np.stack(df["about_embedding"].values),
    #    ...
    #    "post_id": df['post_id'].values,
    #    "description_embedding": np.stack(df["description_embedding"].values),
    #    ...
    # }
    # dataset = tf.data.Dataset.from_tensor_slices(features)

    # For the current non-TFRS TwoTowerModel which expects {'user_inputs':..., 'post_inputs':...}
    # and if we were to train it with a standard Keras fit() and a loss function that takes y_true, y_pred,
    # we would need (x, y) where x is the dict {'user_inputs':..., 'post_inputs':...} and y is the label.
    # However, retrieval models are often trained with specialized losses (like in-batch negatives)
    # where the 'label' is implicit in the pairing or handled by the loss function itself.

    # Let's prepare data for the TFRS wrapper approach, as it's common for two-towers.
    # The TFRS model will expect a flat dictionary of all features.
    flat_features_dict = {}
    for key, val_array in user_inputs.items():
        flat_features_dict[key] = val_array
    for key, val_array in post_inputs.items():
        # Ensure no key clashes, e.g. if 'user_id' was accidentally also a post_feature key
        if key in flat_features_dict and key not in ['user_id', 'post_id']: # user_id/post_id are primary keys
             print(f"Warning: Feature key '{key}' exists in both user and post inputs. Post input will overwrite.")
        flat_features_dict[key] = val_array
    
    # Add label if needed for some custom loss, but TFRS Retrieval task handles it.
    # flat_features_dict['label'] = df['label'].values # Only if not using TFRS task or similar

    if 'description_embedding' in flat_features_dict:
        desc_emb_val = flat_features_dict['description_embedding']
        print(f"DEBUG data_utils: In create_tf_dataset, for 'description_embedding': type {type(desc_emb_val)}, shape {desc_emb_val.shape if hasattr(desc_emb_val, 'shape') else 'N/A'}, example value length: {len(desc_emb_val[0]) if hasattr(desc_emb_val, '__len__') and len(desc_emb_val) > 0 and hasattr(desc_emb_val[0], '__len__') else 'N/A or not list/array'}")
    
    for cat_col_tf_debug in ['category_id', 'media_type', 'creator_id']: # Renamed loop variable
        if cat_col_tf_debug in flat_features_dict:
            cat_val = flat_features_dict[cat_col_tf_debug]
            print(f"DEBUG data_utils: In create_tf_dataset, for '{cat_col_tf_debug}': type {type(cat_val)}, dtype of elements {cat_val.dtype if hasattr(cat_val, 'dtype') else 'N/A'}, example value: {cat_val[0] if hasattr(cat_val, '__len__') and len(cat_val) > 0 else 'N/A'}")

    dataset = tf.data.Dataset.from_tensor_slices(flat_features_dict)
    dataset = dataset.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


if __name__ == '__main__':
    # This is an illustrative example.
    # In a real script, you'd get feast_repo_path from config.
    FEAST_REPO_PATH = "../../src/feature_repo" # Adjust path as needed from data_utils.py
    INTERACTIONS_FILE = "../../../artifacts/data/train/interactions.csv" # Adjust path

    if not os.path.exists(os.path.join(FEAST_REPO_PATH, "feature_store.yaml")):
        print(f"Feast repository not found at {FEAST_REPO_PATH}. Skipping example run.")
    elif not os.path.exists(INTERACTIONS_FILE):
        print(f"Interactions file not found at {INTERACTIONS_FILE}. Skipping example run.")
    else:
        store = FeatureStore(repo_path=FEAST_REPO_PATH)
        
        print("Getting all IDs...")
        user_ids, post_ids, interactions_df = get_all_ids_from_interactions(INTERACTIONS_FILE)
        print(f"Found {len(user_ids)} unique users and {len(post_ids)} unique posts.")
        
        print("Generating positive and negative pairs...")
        training_df = get_positive_and_negative_pairs(interactions_df, post_ids)
        print(f"Generated {len(training_df)} training pairs.")
        print(training_df.head())
        print(training_df['label'].value_counts())

        # Example feature lists (must match your Feast definitions)
        user_features = [
            "user_features_view:about_embedding",
            "user_features_view:headline_embedding",
            "user_features_view:avg_rating_as_creator", # Example additional feature
            "user_features_view:total_posts_created"   # Example additional feature
        ]
        post_features = [
            "post_features_view:description_embedding",
            "post_features_view:views", # Example additional feature
            # "post_features_view:category_id", # If category_id is a feature in Feast
            # "post_features_view:media_type",  # If media_type is a feature in Feast
            # "post_features_view:creator_id"   # If creator_id is a feature in Feast
        ]
        
        # Filter out features not actually defined in the dummy feature_views for safety in example
        # This is just for the example to run with the provided feature_views.py
        defined_user_feature_refs = set()
        for fv in store.list_feature_views():
            if fv.name == "user_features_view":
                for feature in fv.features:
                    defined_user_feature_refs.add(f"user_features_view:{feature.name}")
        
        defined_post_feature_refs = set()
        for fv in store.list_feature_views():
            if fv.name == "post_features_view":
                for feature in fv.features:
                    defined_post_feature_refs.add(f"post_features_view:{feature.name}")

        user_features_to_fetch = [f for f in user_features if f in defined_user_feature_refs]
        post_features_to_fetch = [f for f in post_features if f in defined_post_feature_refs]
        
        if not user_features_to_fetch or not post_features_to_fetch:
            print("Warning: Not enough defined features found in Feast for example run. Skipping feature fetching.")
        else:
            print(f"User features to fetch: {user_features_to_fetch}")
            print(f"Post features to fetch: {post_features_to_fetch}")

            print("Fetching features from Feast...")
            # Limit df size for faster example
            training_data_with_features = fetch_features_for_training(store, training_df.head(100), user_features_to_fetch, post_features_to_fetch)
            
            if not training_data_with_features.empty:
                print("Creating TensorFlow dataset...")
                # These names must match what the model expects in its input dictionaries
                # The create_tf_dataset function will map from these to the model's expected keys
                # (e.g. 'uid' to 'user_id')
                
                # The feature names passed here are the full Feast path (view:feature)
                # The create_tf_dataset function extracts the base feature name.
                tf_dataset = create_tf_dataset(training_data_with_features, 
                                               user_feature_names=user_features_to_fetch, 
                                               post_feature_names=post_features_to_fetch, 
                                               batch_size=32)
                
                print("TensorFlow dataset created.")
                for batch in tf_dataset.take(1):
                    print("Batch structure:")
                    for key, value in batch.items():
                        print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                    # Example: Accessing user_inputs and post_inputs if structured for TwoTowerModel directly
                    # print("User inputs in batch:", batch[0]['user_inputs'].keys())
                    # print("Post inputs in batch:", batch[0]['post_inputs'].keys())
            else:
                print("No data after feature fetching, skipping TF dataset creation.")
        print("Data utils example run finished.")