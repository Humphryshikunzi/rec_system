import pandas as pd
import numpy as np
from feast import FeatureStore
from datetime import datetime, timedelta
import yaml
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Loads YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_interaction_data(file_path, positive_interaction_types):
    """Loads interaction data and identifies positive interactions."""
    logger.info(f"Loading interaction data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        # Assuming 'interaction_type' and 'timestamp' columns exist
        # Convert timestamp to datetime if not already
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else: # if no timestamp, use current time for all, or raise error
            logger.warning("Timestamp column not found or not in expected format. Using current time for all events.")
            df['event_timestamp'] = datetime.utcnow()


        df['label'] = df['interaction_type'].isin(positive_interaction_types).astype(int)
        # Keep only positive interactions for initial seed
        positive_df = df[df['label'] == 1][['user_id', 'post_id', 'timestamp', 'label']].copy()
        logger.info(f"Loaded {len(df)} interactions, {len(positive_df)} positive interactions identified.")
        return positive_df, df
    except FileNotFoundError:
        logger.error(f"Interaction data file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading interaction data from {file_path}: {e}")
        raise

def generate_negative_samples(positive_df, all_interactions_df, all_posts_df, ratio=4, random_seed=42):
    """
    Generates negative samples for each user.
    Negative samples are posts the user has not positively interacted with.
    """
    logger.info(f"Generating negative samples with ratio {ratio}")
    np.random.seed(random_seed)
    
    user_positive_interactions = positive_df.groupby('user_id')['post_id'].apply(set).to_dict()
    all_post_ids = set(all_posts_df['post_id'].unique())
    
    negative_samples = []
    for user_id, interacted_posts in user_positive_interactions.items():
        num_positive = len(interacted_posts)
        num_negative_to_sample = num_positive * ratio
        
        possible_negatives = list(all_post_ids - interacted_posts)
        
        if not possible_negatives:
            logger.warning(f"User {user_id} has interacted with all posts. No negative samples can be generated.")
            continue
            
        sampled_negatives = np.random.choice(
            possible_negatives, 
            size=min(num_negative_to_sample, len(possible_negatives)), 
            replace=False # Sample without replacement to avoid duplicate negative samples for a user
        )
        
        # Use the latest timestamp from positive interactions for this user as a proxy for event_timestamp
        # This is a simplification; ideally, impression events would have their own timestamps.
        # If positive_df is empty for a user (should not happen if we iterate user_positive_interactions),
        # or if timestamps are missing, use a default.
        user_timestamps = positive_df[positive_df['user_id'] == user_id]['timestamp']
        event_ts = user_timestamps.max() if not user_timestamps.empty else datetime.utcnow()

        for post_id in sampled_negatives:
            negative_samples.append({'user_id': user_id, 'post_id': post_id, 'label': 0, 'timestamp': event_ts})
            
    negative_df = pd.DataFrame(negative_samples)
    logger.info(f"Generated {len(negative_df)} negative samples.")
    return negative_df

def get_feature_data(store: FeatureStore, entity_df: pd.DataFrame, features: list, config: dict):
    """Fetches features from Feast for the given entities."""
    logger.info(f"Fetching features from Feast: {features}")
    try:
        # Ensure 'event_timestamp' is present and in the correct format
        if 'timestamp' not in entity_df.columns:
            logger.warning("timestamp column not found in entity_df. Adding current UTC time.")
            # This case should ideally not be hit if the calling code prepares entity_df correctly.
            # If it's hit, it means the input DataFrame didn't have 'timestamp' as expected.
            entity_df['timestamp'] = datetime.utcnow()
        elif not pd.api.types.is_datetime64_any_dtype(entity_df['timestamp']):
            entity_df['timestamp'] = pd.to_datetime(entity_df['timestamp'])

        # Ensure timestamps are timezone-aware (UTC) as Feast expects
        if entity_df['timestamp'].dt.tz is None:
            entity_df['timestamp'] = entity_df['timestamp'].dt.tz_localize('UTC')
        else:
            entity_df['timestamp'] = entity_df['timestamp'].dt.tz_convert('UTC')


        # Check for required entity keys
        required_keys = set()
        for feature_ref_str in features:
            # Assuming format "feature_view_name:feature_name"
            # We need to know which entities are involved.
            # This part is tricky without knowing the exact feature view definitions.
            # For now, assume 'user_id' and 'post_id' are the primary keys.
            # A more robust solution would parse feature view definitions.
            if "user_features_view" in feature_ref_str:
                required_keys.add("user_id")
            if "post_features_view" in feature_ref_str:
                required_keys.add("post_id")
            # Add more logic if other entities are involved

        missing_keys = required_keys - set(entity_df.columns)
        if missing_keys:
            raise ValueError(f"Entity DataFrame is missing required keys for Feast: {missing_keys}")


        training_data = store.get_historical_features(
            entity_df=entity_df,
            features=features,
        ).to_df()
        logger.info(f"Successfully fetched {len(training_data)} rows of feature data.")
        return training_data
    except Exception as e:
        logger.error(f"Error fetching features from Feast: {e}")
        # Log more details if possible, e.g., entity_df.head()
        logger.error(f"Entity DF head:\n{entity_df.head()}")
        raise

def compute_on_the_fly_features(feature_df: pd.DataFrame, config: dict):
    """Computes interaction features that are not directly available from Feast."""
    logger.info("Computing on-the-fly features.")
    
    # Example: Cosine similarity between user and post embeddings
    # Ensure embedding columns exist and handle potential NaNs or missing embeddings
    user_emb_col = "user_features_view__about_embedding" # Example, adjust to actual column name
    post_emb_col = "post_features_view__description_embedding" # Example, adjust to actual column name

    if user_emb_col in feature_df.columns and post_emb_col in feature_df.columns:
        # Convert string representations of embeddings (if any) to numpy arrays
        # This is a common issue if embeddings are stored as strings in CSVs/Feast
        def to_numpy_array(emb_str):
            if isinstance(emb_str, (list, np.ndarray)):
                return np.array(emb_str)
            if isinstance(emb_str, str):
                try:
                    return np.array(eval(emb_str)) # Be cautious with eval
                except:
                    return np.nan # Or some default embedding
            return np.nan # Or default

        # Handle cases where embeddings might be missing or not lists/arrays
        # Fill NaNs with a zero vector of appropriate dimension if known, or handle downstream
        # For simplicity, we'll attempt conversion and rely on cosine_similarity to handle NaNs if they propagate
        
        # Check if embeddings are already numpy arrays or lists
        if not feature_df[user_emb_col].empty and isinstance(feature_df[user_emb_col].iloc[0], (np.ndarray, list)):
            user_embeddings = np.array(feature_df[user_emb_col].tolist())
        else: # Assume string representation or other, try to convert
             user_embeddings = np.array(feature_df[user_emb_col].apply(lambda x: to_numpy_array(x) if pd.notnull(x) else np.nan).tolist())


        if not feature_df[post_emb_col].empty and isinstance(feature_df[post_emb_col].iloc[0], (np.ndarray, list)):
            post_embeddings = np.array(feature_df[post_emb_col].tolist())
        else:
            post_embeddings = np.array(feature_df[post_emb_col].apply(lambda x: to_numpy_array(x) if pd.notnull(x) else np.nan).tolist())

        # Handle potential all-NaN slices if embeddings are missing for some rows
        # Cosine similarity will produce NaN if an embedding is all NaN or zero.
        # We need to ensure embeddings are 2D arrays for cosine_similarity.
        # If an embedding is NaN, replace with a zero vector of the correct dimension.
        # This requires knowing the embedding dimension. Let's assume a default or infer.
        
        sims = []
        # Determine embedding dimension (assuming all embeddings have the same dim)
        # This is a simplification. A robust solution would get dim from metadata.
        # Fallback to a default dimension if cannot infer.
        emb_dim = None
        if user_embeddings.ndim > 1 and user_embeddings.shape[1] > 0 :
            emb_dim = user_embeddings.shape[1]
        elif post_embeddings.ndim > 1 and post_embeddings.shape[1] > 0:
            emb_dim = post_embeddings.shape[1]
        
        default_emb_if_nan = np.zeros(emb_dim) if emb_dim else None

        for u_emb, p_emb in zip(user_embeddings, post_embeddings):
            if isinstance(u_emb, float) and np.isnan(u_emb): # Handle scalar NaN from .tolist() if original was NaN
                u_emb_valid = default_emb_if_nan
            else:
                u_emb_valid = u_emb
            
            if isinstance(p_emb, float) and np.isnan(p_emb):
                p_emb_valid = default_emb_if_nan
            else:
                p_emb_valid = p_emb

            if u_emb_valid is not None and p_emb_valid is not None and \
               u_emb_valid.ndim == 1 and p_emb_valid.ndim == 1: # Ensure they are 1D arrays (single embeddings)
                sim = cosine_similarity(u_emb_valid.reshape(1, -1), p_emb_valid.reshape(1, -1))[0, 0]
            else:
                sim = 0.0 # Default similarity if embeddings are invalid or missing
            sims.append(sim)
        feature_df['cosine_similarity_user_post_embedding'] = sims
        logger.info("Computed cosine_similarity_user_post_embedding.")
    else:
        logger.warning(f"Embedding columns ('{user_emb_col}', '{post_emb_col}') not found. Skipping cosine similarity.")
        feature_df['cosine_similarity_user_post_embedding'] = 0.0 # Default value

    # Example: is_post_category_in_onboarding
    # Requires 'user_features_view__onboarding_category_ids' and 'post_features_view__category_id'
    user_onboarding_col = "user_features_view__onboarding_category_ids" # List of ints
    post_category_col = "post_features_view__category_id" # Single int

    if user_onboarding_col in feature_df.columns and post_category_col in feature_df.columns:
        def check_category_in_onboarding(row):
            onboarding_cats = row[user_onboarding_col]
            post_cat = row[post_category_col]
            if isinstance(onboarding_cats, (list, np.ndarray)) and pd.notnull(post_cat):
                return int(post_cat in onboarding_cats)
            # Handle cases where onboarding_cats might be a string representation of a list
            if isinstance(onboarding_cats, str):
                try:
                    onboarding_cats_list = eval(onboarding_cats) # Be cautious with eval
                    if isinstance(onboarding_cats_list, list) and pd.notnull(post_cat):
                        return int(post_cat in onboarding_cats_list)
                except:
                    return 0 # Default if parsing fails
            return 0 # Default if types are not as expected or data is missing

        feature_df['is_post_category_in_onboarding'] = feature_df.apply(check_category_in_onboarding, axis=1)
        logger.info("Computed is_post_category_in_onboarding.")
    else:
        logger.warning(f"Required columns for 'is_post_category_in_onboarding' not found. Skipping.")
        feature_df['is_post_category_in_onboarding'] = 0 # Default value
    
    # Add more on-the-fly feature computations here as needed
    # e.g., is_post_category_in_top_N_interacted, user_interacted_with_creator

    return feature_df

def prepare_data_for_model(df: pd.DataFrame, config: dict):
    """Prepares data for the specified model type (XGBoost, LightGBM, or NN)."""
    logger.info(f"Preparing data for model type: {config.get('model_type', 'xgboost')}")
    
    # Ensure 'label' is present
    if 'label' not in df.columns:
        raise ValueError("Label column is missing from the feature DataFrame.")

    # Identify feature columns (exclude ids, timestamp, label)
    # This needs to be robust. The feature_list from config should guide this.
    # Feast column names are typically like 'feature_view_name__feature_name'
    
    # Construct expected feature columns from config['feature_list']
    # and add on-the-fly computed features
    model_feature_cols = []
    if 'feature_list' in config:
        for f_item in config['feature_list']:
            # Feast format: view_name:feature_name -> view_name__feature_name
            model_feature_cols.append(f_item.replace(":", "__")) 
    
    # Add on-the-fly features if they were computed
    if 'cosine_similarity_user_post_embedding' in df.columns:
        model_feature_cols.append('cosine_similarity_user_post_embedding')
    if 'is_post_category_in_onboarding' in df.columns:
        model_feature_cols.append('is_post_category_in_onboarding')
    # Add other on-the-fly features here

    # Ensure all model_feature_cols are actually in the df
    final_feature_cols = [col for col in model_feature_cols if col in df.columns]
    missing_configured_features = set(model_feature_cols) - set(final_feature_cols)
    if missing_configured_features:
        logger.warning(f"Some configured features were not found in the final DataFrame and will be excluded: {missing_configured_features}")

    X = df[final_feature_cols]
    y = df['label']

    # Handle categorical features for tree-based models (e.g., one-hot or label encoding)
    # For XGBoost/LightGBM, they can often handle categorical features natively or with label encoding.
    # For simplicity, we'll rely on their native handling if possible, or do label encoding.
    # Neural Networks would require embeddings or one-hot encoding.

    # Identify categorical columns (heuristic: object type or specific names like_id)
    # This is a simplification. A more robust approach would use a schema or explicit list.
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Also, ID columns are typically categorical, even if numeric
    id_like_cols = [col for col in X.columns if 'id' in col.lower() and col not in categorical_cols]
    categorical_cols.extend(id_like_cols)
    categorical_cols = list(set(categorical_cols)) # Unique

    # For tree models, convert object/string categorical columns to a numerical representation
    # XGBoost can handle pd.Categorical type directly if enable_categorical=True is used.
    # Or, we can label encode.
    for col in categorical_cols:
        if X[col].dtype == 'object':
            try:
                # Attempt to convert to numeric if they are string numbers
                X[col] = pd.to_numeric(X[col], errors='raise')
            except (ValueError, TypeError):
                # If not numeric, use label encoding or treat as pd.Categorical
                X[col] = X[col].astype('category').cat.codes # Simple label encoding
                # For XGBoost, could also do: X[col] = X[col].astype('category')
        elif pd.api.types.is_list_like(X[col].iloc[0]) and col.endswith("embedding"): # Skip embeddings
            logger.info(f"Column {col} looks like an embedding, will be handled by model or needs flattening.")
            # XGBoost typically expects flat numerical features. Embeddings need to be flattened or aggregated.
            # For now, assume embeddings are lists/arrays of numbers.
            # If they are multi-dimensional, XGBoost will error. They need to be flattened.
            # Example: Flatten if it's a list of numbers
            if X[col].apply(lambda x: isinstance(x, (list, np.ndarray))).all():
                try:
                    # Create new columns for each embedding dimension
                    emb_df = pd.DataFrame(X[col].tolist(), index=X.index).add_prefix(f"{col}_dim_")
                    X = X.drop(columns=[col])
                    X = pd.concat([X, emb_df], axis=1)
                    logger.info(f"Flattened embedding column {col} into {emb_df.shape[1]} dimensions.")
                except Exception as e:
                    logger.error(f"Could not flatten embedding column {col}: {e}. It might be skipped or cause errors.")
            else:
                 logger.warning(f"Column {col} is list-like but not consistently. Skipping flattening.")


    # Handle list-like features that are not embeddings (e.g., onboarding_category_ids)
    # For tree models, MultiLabelBinarizer or similar might be needed if not handled by native cat features.
    # For 'user_features_view__onboarding_category_ids', if it's a list of IDs.
    onboarding_col_feast_name = "user_features_view__onboarding_category_ids"
    if onboarding_col_feast_name in X.columns and X[onboarding_col_feast_name].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
        logger.info(f"Processing list-like column: {onboarding_col_feast_name} using MultiLabelBinarizer-like approach.")
        # This is a simplified approach. A proper MultiLabelBinarizer might be better.
        # Explode the list into multiple binary columns for each category.
        # First, get all unique categories present in this column across the dataset.
        
        # Handle potential string representations of lists
        def parse_list_string(s):
            if isinstance(s, (list, np.ndarray)):
                return s
            if isinstance(s, str):
                try:
                    return eval(s)
                except: return []
            return []

        temp_series = X[onboarding_col_feast_name].apply(parse_list_string)
        mlb = MultiLabelBinarizer()
        
        # Fit MLB only on valid lists
        valid_lists = temp_series[temp_series.apply(lambda x: isinstance(x, list) and len(x) > 0)]
        if not valid_lists.empty:
            mlb.fit(valid_lists)
            binarized_df = pd.DataFrame(mlb.transform(temp_series), columns=[f"{onboarding_col_feast_name}_{cls}" for cls in mlb.classes_], index=X.index)
            X = X.drop(columns=[onboarding_col_feast_name])
            X = pd.concat([X, binarized_df], axis=1)
            logger.info(f"Binarized {onboarding_col_feast_name} into {len(mlb.classes_)} columns.")
        else:
            logger.warning(f"No valid list data found for {onboarding_col_feast_name} to binarize. Dropping column.")
            X = X.drop(columns=[onboarding_col_feast_name], errors='ignore')


    # Ensure all columns are numeric and handle NaNs
    # XGBoost can handle NaNs natively.
    X = X.fillna(np.nan) # Or a specific value like -1 if preferred and NaNs are not handled by model

    # For Neural Network (tf.data.Dataset)
    if config.get('model_type') == "neural_network":
        # This part would be more complex:
        # - Separate numerical, categorical (for embedding), and pre-computed embedding features.
        # - Create a tf.data.Dataset from slices.
        # Example (very simplified):
        # feature_dict = {col: X[col].values for col in X.columns}
        # label_array = y.values
        # dataset = tf.data.Dataset.from_tensor_slices((feature_dict, label_array))
        # dataset = dataset.batch(config.get('batch_size', 256))
        logger.warning("Neural Network data preparation is placeholder. Implement specific preprocessing.")
        pass # Placeholder for NN

    logger.info(f"Final feature shape: {X.shape}, Label shape: {y.shape}")
    logger.info(f"Feature columns for model: {X.columns.tolist()}")
    
    # Check for non-numeric columns remaining (should not happen for XGBoost after processing)
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns
    if not non_numeric_cols.empty:
        logger.error(f"Non-numeric columns remain in X: {non_numeric_cols.tolist()}. This will likely cause errors.")
        for col in non_numeric_cols:
            logger.error(f"Column '{col}' dtype: {X[col].dtype}, example values: {X[col].head().tolist()}")

    return X, y


def get_training_and_validation_data(config_path: str):
    """Main function to load config, fetch data, and prepare for training and validation."""
    
    config = load_config(config_path)
    feast_repo_path = os.path.join(os.getcwd(), config['feast_repo_path']) # Ensure full path
    store = FeatureStore(repo_path=feast_repo_path)
    
    np.random.seed(config.get('random_seed', 42))

    # --- Load All Posts Data (needed for negative sampling) ---
    # This path should ideally be in config or inferred.
    # Assuming posts data is available similar to interactions.
    # This is a simplification; in a real system, you might query a DB or another service.
    try:
        # Try to get post data path from data_config.yaml if it exists, or use a default
        # This is a placeholder for a more robust way to get all_posts_df
        posts_file_path = "artifacts/data/train/posts.csv" # Default, should be configurable
        if os.path.exists("config/data_config.yaml"):
            with open("config/data_config.yaml", 'r') as f:
                data_cfg = yaml.safe_load(f)
                posts_file_path = data_cfg.get('raw_data_paths', {}).get('posts', posts_file_path)
        
        logger.info(f"Loading all posts data from {posts_file_path} for negative sampling.")
        all_posts_df = pd.read_csv(posts_file_path)
        if 'post_id' not in all_posts_df.columns:
            raise ValueError("all_posts_df must contain 'post_id' column.")
    except FileNotFoundError:
        logger.error(f"Posts data file not found: {posts_file_path}. Negative sampling might be impaired.")
        # Create a dummy all_posts_df if file not found, to allow pipeline to proceed with warnings
        # This is not ideal for actual training.
        all_posts_df = pd.DataFrame(columns=['post_id']) # Empty, so negative sampling will be limited
    except Exception as e:
        logger.error(f"Error loading all posts data: {e}")
        all_posts_df = pd.DataFrame(columns=['post_id'])


    # --- Process Training Data ---
    logger.info("--- Processing Training Data ---")
    train_interactions_path = config['interaction_data_paths']['train']
    positive_train_df, all_train_interactions_df = load_interaction_data(train_interactions_path, config['positive_interaction_types'])
    
    # Generate negative samples only if there are positive samples and posts to sample from
    if not positive_train_df.empty and not all_posts_df.empty:
        negative_train_df = generate_negative_samples(
            positive_train_df, all_train_interactions_df, all_posts_df,
            ratio=config.get('negative_sampling_ratio', 4),
            random_seed=config.get('random_seed', 42)
        )
        train_entity_df = pd.concat([
            positive_train_df[['user_id', 'post_id', 'timestamp', 'label']],
            negative_train_df
        ], ignore_index=True)
    elif not positive_train_df.empty and all_posts_df.empty:
        logger.warning("No posts data available for negative sampling. Using only positive samples for training.")
        train_entity_df = positive_train_df[['user_id', 'post_id', 'timestamp', 'label']].copy()
    else: # No positive samples
        logger.warning("No positive training samples found. Training data will be empty.")
        train_entity_df = pd.DataFrame(columns=['user_id', 'post_id', 'timestamp', 'label'])

    if not train_entity_df.empty:
        # Ensure 'timestamp' is datetime and UTC for consistency before fetching and merging
        if not pd.api.types.is_datetime64_any_dtype(train_entity_df['timestamp']):
            train_entity_df['timestamp'] = pd.to_datetime(train_entity_df['timestamp'])
        if train_entity_df['timestamp'].dt.tz is None:
            train_entity_df['timestamp'] = train_entity_df['timestamp'].dt.tz_localize('UTC')
        else:
            train_entity_df['timestamp'] = train_entity_df['timestamp'].dt.tz_convert('UTC')

        train_features_df = get_feature_data(store, train_entity_df[['user_id', 'post_id', 'timestamp']], config['feature_list'], config)
        # Merge labels back after fetching features, as Feast drops non-entity/timestamp columns
        # Now both DataFrames should have 'timestamp' as datetime64[ns, UTC]
        train_full_df = pd.merge(train_features_df, train_entity_df[['user_id', 'post_id', 'timestamp', 'label']], on=['user_id', 'post_id', 'timestamp'])
        train_full_df = compute_on_the_fly_features(train_full_df, config)
        X_train, y_train = prepare_data_for_model(train_full_df, config)
    else:
        X_train, y_train = pd.DataFrame(), pd.Series(dtype='int')


    # --- Process Validation Data ---
    logger.info("--- Processing Validation Data ---")
    val_interactions_path = config['interaction_data_paths']['validation']
    positive_val_df, all_val_interactions_df = load_interaction_data(val_interactions_path, config['positive_interaction_types'])

    if not positive_val_df.empty and not all_posts_df.empty:
        negative_val_df = generate_negative_samples(
            positive_val_df, all_val_interactions_df, all_posts_df,
            ratio=config.get('negative_sampling_ratio', 4), # Use same ratio or could be different
            random_seed=config.get('random_seed', 42) # Use same seed for consistency if desired
        )
        val_entity_df = pd.concat([
            positive_val_df[['user_id', 'post_id', 'timestamp', 'label']],
            negative_val_df
        ], ignore_index=True)
    elif not positive_val_df.empty and all_posts_df.empty:
        logger.warning("No posts data available for negative sampling. Using only positive samples for validation.")
        val_entity_df = positive_val_df[['user_id', 'post_id', 'timestamp', 'label']].copy()
    else: # No positive samples
        logger.warning("No positive validation samples found. Validation data will be empty.")
        val_entity_df = pd.DataFrame(columns=['user_id', 'post_id', 'timestamp', 'label'])


    if not val_entity_df.empty:
        # Ensure 'timestamp' is datetime and UTC for consistency before fetching and merging
        if not pd.api.types.is_datetime64_any_dtype(val_entity_df['timestamp']):
            val_entity_df['timestamp'] = pd.to_datetime(val_entity_df['timestamp'])
        if val_entity_df['timestamp'].dt.tz is None:
            val_entity_df['timestamp'] = val_entity_df['timestamp'].dt.tz_localize('UTC')
        else:
            val_entity_df['timestamp'] = val_entity_df['timestamp'].dt.tz_convert('UTC')

        val_features_df = get_feature_data(store, val_entity_df[['user_id', 'post_id', 'timestamp']], config['feature_list'], config)
        # Now both DataFrames should have 'timestamp' as datetime64[ns, UTC]
        val_full_df = pd.merge(val_features_df, val_entity_df[['user_id', 'post_id', 'timestamp', 'label']], on=['user_id', 'post_id', 'timestamp'])
        val_full_df = compute_on_the_fly_features(val_full_df, config)
        X_val, y_val = prepare_data_for_model(val_full_df, config)
        
        # Align columns between X_train and X_val (important if one-hot encoding or MLB created different columns)
        train_cols = set(X_train.columns)
        val_cols = set(X_val.columns)

        # Add missing columns to X_val (filled with 0 or NaN)
        for col in train_cols - val_cols:
            X_val[col] = 0 # Or np.nan
        # Add missing columns to X_train (filled with 0 or NaN) - less common but possible
        for col in val_cols - train_cols:
            X_train[col] = 0 # Or np.nan
        
        # Ensure same column order
        common_cols = list(X_train.columns) # Use X_train's order as canonical
        X_val = X_val[common_cols]

    else:
        X_val, y_val = pd.DataFrame(columns=X_train.columns if not X_train.empty else []), pd.Series(dtype='int')


    logger.info(f"Training data shape: X_train {X_train.shape}, y_train {y_train.shape}")
    logger.info(f"Validation data shape: X_val {X_val.shape}, y_val {y_val.shape}")
    
    return X_train, y_train, X_val, y_val, config


if __name__ == '__main__':
    # Example usage:
    # This assumes your Feast feature repo is in 'src/feature_repo' relative to where this script is run
    # and your config is 'config/ranking_model_config.yaml'
    
    # Determine project root dynamically to locate config file correctly
    # Assuming this script is in src/model_training/ranking/
    # Project root is ../../.. from here
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    config_file_path = os.path.join(project_root, "config/ranking_model_config.yaml")
    
    if not os.path.exists(config_file_path):
        logger.error(f"Config file not found at {config_file_path}. Please ensure it exists.")
    else:
        logger.info(f"Loading configuration from: {config_file_path}")
        try:
            X_train, y_train, X_val, y_val, loaded_config = get_training_and_validation_data(config_file_path)
            
            logger.info("\n--- Example Data ---")
            if not X_train.empty:
                logger.info("X_train head:\n" + X_train.head().to_string())
                logger.info("\ny_train head:\n" + y_train.head().to_string())
            else:
                logger.info("X_train is empty.")

            if not X_val.empty:
                logger.info("\nX_val head:\n" + X_val.head().to_string())
                logger.info("\ny_val head:\n" + y_val.head().to_string())
            else:
                logger.info("X_val is empty.")
            
            logger.info(f"\nLoaded config: {loaded_config}")

        except Exception as e:
            logger.error(f"An error occurred during data preparation: {e}", exc_info=True)