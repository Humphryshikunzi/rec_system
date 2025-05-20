import argparse
import pandas as pd
import yaml
import os
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from feast import FeatureStore
from datetime import datetime, timedelta
import tempfile
import shutil

from model import UserTower, PostTower # TwoTowerModel (if used directly) or TFRS wrapper
import data_utils # Assuming data_utils.py is in the same directory
import tensorflow_recommenders as tfrs

# For reproducibility
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class TFRSTwoTowerModel(tfrs.Model):
    def __init__(self, user_tower, post_tower, user_id_key, post_id_key, user_feature_keys, post_feature_keys, all_post_ids_for_candidates, post_dataset_for_candidates):
        super().__init__()
        self.user_tower = user_tower
        self.post_tower = post_tower
        
        self.user_id_key = user_id_key
        self.post_id_key = post_id_key
        self.user_feature_keys = user_feature_keys # list of feature names for user tower (excluding id)
        self.post_feature_keys = post_feature_keys # list of feature names for post tower (excluding id)

        # The task is a tfrs.tasks.Retrieval object.
        # It will compute the loss and metrics.
        # For candidates, we can pass all unique post embeddings.
        # This requires computing all post embeddings once.
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=post_dataset_for_candidates.batch(128).map(self.post_tower) # Pass all post features to post_tower
            )
        )

    def _prepare_tower_inputs(self, features, tower_type):
        if tower_type == "user":
            id_key = self.user_id_key
            feature_keys_config = self.user_feature_keys # These are from config (view:feature)
            # Model expects base feature names (e.g. "about_embedding")
            model_input_keys = {f.split(':')[1]: f for f in feature_keys_config}
            model_input_keys[id_key] = id_key # Add user_id key
        elif tower_type == "post":
            id_key = self.post_id_key
            feature_keys_config = self.post_feature_keys
            model_input_keys = {f.split(':')[1]: f for f in feature_keys_config}
            # Add categorical features expected by PostTower explicitly if not in feature_keys_config
            # These are base names the model expects.
            categorical_post_features_model = ['category_id', 'media_type', 'creator_id', 'description_embedding']
            for cat_key in categorical_post_features_model:
                 if cat_key not in model_input_keys: # if not already mapped from config
                    model_input_keys[cat_key] = cat_key # Assume feature name in `features` matches model key
            model_input_keys[id_key] = id_key # Add post_id key
        else:
            raise ValueError("Invalid tower_type")

        tower_inputs = {}
        for model_key, feature_source_key in model_input_keys.items():
            if feature_source_key in features:
                tower_inputs[model_key] = features[feature_source_key]
            elif model_key in features: # If feature_source_key was already the base model key
                tower_inputs[model_key] = features[model_key]
            else:
                # This might happen if a feature specified in config isn't in the dataset batch
                # Or if a model's hardcoded key (like 'category_id') isn't in the batch
                # data_utils should ensure all necessary features are present.
                raise KeyError(f"Feature '{feature_source_key}' (for model key '{model_key}') not found in input features: {list(features.keys())}")
        return tower_inputs

    def compute_loss(self, features, training=False):
        # Features is a batch from the dataset prepared by data_utils.py
        # It should contain all necessary fields for both towers.
        
        user_inputs = self._prepare_tower_inputs(features, "user")
        post_inputs = self._prepare_tower_inputs(features, "post")

        user_embeddings = self.user_tower(user_inputs)
        post_embeddings = self.post_tower(post_inputs) # These are positive post embeddings

        # The task will use these embeddings and the candidates (all post embeddings)
        # to compute the loss (typically in-batch softmax or sampled softmax).
        return self.task(query_embeddings=user_embeddings, candidate_embeddings=post_embeddings, compute_metrics=not training)


def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seeds(config.get('random_seed', 42))
    
    print("Configuration loaded:", config)
    feast_repo_path = os.path.join(config['project_root'], config['feast_repo_path']) if 'project_root' in config else config['feast_repo_path']
    
    # MLflow setup
    mlflow.set_experiment(config['mlflow_experiment_name'])

    with mlflow.start_run() as run:
        mlflow.log_params(config)
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Initialize Feast FeatureStore
        print(f"Initializing Feast store from: {feast_repo_path}")
        store = FeatureStore(repo_path=feast_repo_path)

        # --- 1. Load and Prepare Data ---
        print("Loading and preparing data...")
        interactions_file = os.path.join(config.get('project_root', '.'), 'artifacts/data/train/interactions.csv')
        
        # Get all user and post IDs for vocabularies and negative sampling
        all_user_ids_list, all_post_ids_list, interactions_df = data_utils.get_all_ids_from_interactions(interactions_file)
        
        # Create vocabularies for ID-based features (user_id, post_id, category_id, etc.)
        # This is crucial for Embedding layers. StringLookup can build them.
        # For simplicity, we'll assume max ID values for vocab sizes, or they are pre-calculated.
        # In a robust pipeline, these vocabs would be built from all available IDs.
        user_id_vocab_size = len(all_user_ids_list) + 2 # +1 for OOV, +1 for padding/mask if used
        post_id_vocab_size = len(all_post_ids_list) + 2
        
        # Placeholder vocab sizes for other categorical IDs - these should be derived from data
        # e.g., by finding unique values in posts.csv or features from Feast
        # For now, using arbitrary large enough values or assuming they are handled by StringLookup if inputs are strings
        # If inputs to Embedding layers are already integers, these are max_id + 1
        # Let's assume data_utils.py and Feast provide integer IDs or StringLookup is used internally if needed.
        # The model.py currently expects vocab_size for Embedding layers.
        # If using StringLookup -> Embedding, StringLookup handles vocab creation.
        # For now, we pass estimated vocab sizes.
        
        # Fetch unique values for categorical features from posts.csv to determine vocab sizes
        posts_df_path = os.path.join(config.get('project_root', '.'), 'artifacts/data/train/posts.csv')
        if os.path.exists(posts_df_path):
            posts_df = pd.read_csv(posts_df_path)
            category_id_vocab_size = posts_df['category_id'].nunique() + 2 
            media_type_vocab_size = posts_df['media_type'].nunique() + 2
            creator_id_vocab_size = posts_df['creator_id'].nunique() + 2 # Assuming creator_id is like user_id
        else:
            print(f"Warning: {posts_df_path} not found. Using placeholder vocab sizes for category, media_type, creator_id.")
            category_id_vocab_size = config.get('category_id_vocab_size', 100) # Placeholder
            media_type_vocab_size = config.get('media_type_vocab_size', 20)    # Placeholder
            creator_id_vocab_size = config.get('creator_id_vocab_size', user_id_vocab_size) # Placeholder

        print(f"User ID Vocab Size: {user_id_vocab_size}")
        print(f"Post ID Vocab Size: {post_id_vocab_size}")
        print(f"Category ID Vocab Size: {category_id_vocab_size}")
        print(f"Media Type Vocab Size: {media_type_vocab_size}")
        print(f"Creator ID Vocab Size: {creator_id_vocab_size}")

        training_pairs_df = data_utils.get_positive_and_negative_pairs(interactions_df, all_post_ids_list)
        
        # Fetch features
        user_feature_list_config = config['user_tower_features']
        post_feature_list_config = config['post_tower_features']
        
        training_data_with_features = data_utils.fetch_features_for_training(
            store, 
            training_pairs_df, 
            user_features_list=user_feature_list_config,
            post_features_list=post_feature_list_config
        )

        if training_data_with_features.empty:
            print("No training data after feature fetching. Exiting.")
            mlflow.log_metric("training_samples", 0)
            return

        mlflow.log_metric("training_samples", len(training_data_with_features))
        print(f"Number of training samples with features: {len(training_data_with_features)}")

        # Create tf.data.Dataset
        # The feature names passed to create_tf_dataset are the full "view:feature" strings from config
        train_dataset = data_utils.create_tf_dataset(
            training_data_with_features,
            user_feature_names=user_feature_list_config,
            post_feature_names=post_feature_list_config,
            batch_size=config['batch_size']
        )
        print("TensorFlow training dataset created.")

        # --- 2. Define Model ---
        embedding_dim = config['embedding_dim']
        user_tower = UserTower(
            user_id_vocab_list=all_user_ids_list, # Pass the list of string user IDs
            embedding_dim=embedding_dim
        )
        post_tower = PostTower(
            post_id_vocab_list=all_post_ids_list, # Pass the list of string post IDs
            category_id_vocab_size=category_id_vocab_size,
            media_type_vocab_size=media_type_vocab_size,
            creator_id_vocab_size=creator_id_vocab_size,
            embedding_dim=embedding_dim
        )
    
        # For TFRS, we need a dataset of all candidate items (posts) to compute embeddings for the FactorizedTopK metric.
        # This involves fetching features for ALL posts.
        print("Preparing dataset for all post candidates (for TFRS FactorizedTopK metric)...")
        all_post_ids_df = pd.DataFrame({'post_id': all_post_ids_list})
        all_post_ids_df['event_timestamp'] = pd.to_datetime(datetime.now()) # Use current time
        print(f"DEBUG: Entity DF for all_posts_features lookup:\n{all_post_ids_df.head()}")
        print(f"DEBUG: Requesting features: {post_feature_list_config}")

        # Fetch features for all posts
        all_posts_features_df = store.get_historical_features(
            entity_df=all_post_ids_df,
            features=post_feature_list_config + [ # Add categorical features if not in list but needed by tower
                # "post_features_view:category_id", # Example if these are from Feast
                # "post_features_view:media_type",
                # "post_features_view:creator_id"
            ]
        ).to_df()
        print(f"DEBUG: all_posts_features_df shape after get_historical_features: {all_posts_features_df.shape}")
        print(f"DEBUG: all_posts_features_df head after get_historical_features:\n{all_posts_features_df.head()}")
        
        # Merge with posts_df to get category_id, mediaType, creator_id if not from Feast
        # This assumes these IDs are directly in posts_df and need to be included for the post_tower
        if os.path.exists(posts_df_path):
            base_posts_info_df = pd.read_csv(posts_df_path)[['post_id', 'category_id', 'media_type', 'creator_id']].drop_duplicates(subset=['post_id'])
            # Rename mediaType to media_type to match model expectation
            base_posts_info_df.rename(columns={'mediaType': 'media_type'}, inplace=True)
            all_posts_features_df = pd.merge(all_posts_features_df, base_posts_info_df, on='post_id', how='left')
        
        # Handle potential suffixed column names from Feast joins (e.g., 'category_id_x')
        # These are the base names of categorical features we expect.
        categorical_base_names = ['category_id', 'media_type', 'creator_id']
        rename_map = {}
        for base_name in categorical_base_names:
            suffixed_x = base_name + '_x'
            suffixed_y = base_name + '_y'
            if suffixed_x in all_posts_features_df.columns:
                rename_map[suffixed_x] = base_name
            elif suffixed_y in all_posts_features_df.columns: # Check for _y if _x isn't present
                rename_map[suffixed_y] = base_name
        
        if rename_map:
            print(f"INFO: Renaming suffixed columns in all_posts_features_df: {rename_map}")
            all_posts_features_df.rename(columns=rename_map, inplace=True)

        # Process categorical features (fillna, astype) using the now-standardized base names
        for col in categorical_base_names:
            if col in all_posts_features_df.columns:
                all_posts_features_df[col] = all_posts_features_df[col].fillna(0) # Default for missing categorical
                try:
                    # Ensure it's not all NaNs before astype if fillna(0) wasn't sufficient or if it was already non-numeric
                    if not all_posts_features_df[col].isnull().all():
                         all_posts_features_df[col] = all_posts_features_df[col].astype(int)
                except ValueError:
                    print(f"Warning: Could not convert column '{col}' to int after fillna. It might contain non-numeric strings (e.g. 'cat_01'). Vocabularies should handle this.")
            else:
                print(f"Warning: Expected categorical column '{col}' not found after potential renaming. Skipping fillna/astype.")

        # Process description_embedding
        desc_emb_col = 'description_embedding'
        # Use the actual dimension for description_embedding, not the general embedding_dim
        # This should be consistent with how description_embedding is generated/stored.
        # Defaulting to 384 if not specified in config, as used in the pre-build step.
        actual_desc_emb_dim = config.get('description_embedding_dim', 384)

        if desc_emb_col in all_posts_features_df.columns:
            zero_emb = np.zeros(actual_desc_emb_dim)
            all_posts_features_df[desc_emb_col] = all_posts_features_df[desc_emb_col].apply(
                lambda x: x if isinstance(x, (list, np.ndarray)) and len(x) == actual_desc_emb_dim else zero_emb
            )
        
        # Drop rows if critical features like 'post_id' or 'description_embedding' are still NaN
        # (though description_embedding should be filled with zero_emb if it was NaN).
        # Categorical features have defaults, so they shouldn't cause row drops here if only they were NaN.
        critical_cols_for_dropna = ['post_id']
        if desc_emb_col in all_posts_features_df.columns: # Add if it exists
            critical_cols_for_dropna.append(desc_emb_col)
        all_posts_features_df.dropna(subset=critical_cols_for_dropna, inplace=True)

        if all_posts_features_df.empty:
            raise ValueError("No post features remained after NaN handling for TFRS candidates. Check Feast setup, post data, and embedding dimensions.")

        # Create a tf.data.Dataset for all posts
        post_candidates_dict = {'post_id': all_posts_features_df['post_id'].values}
        
        # Base names from config (e.g., 'description_embedding', 'category_id')
        post_model_feature_keys_from_config = [f.split(':')[1] for f in post_feature_list_config]
        
        # Define all keys expected by the PostTower model, using base names
        # This includes embeddings and the processed categorical features
        expected_post_model_keys = [desc_emb_col] + categorical_base_names
        
        # Combine and deduplicate all keys to iterate over
        all_keys_to_add_to_dict = list(dict.fromkeys(post_model_feature_keys_from_config + expected_post_model_keys))

        for key in all_keys_to_add_to_dict:
            if key == 'post_id': # Already added
                continue
            if key in all_posts_features_df.columns:
                if not all_posts_features_df[key].empty and not all_posts_features_df[key].isnull().all():
                    # Use dropna().iloc[0] to get a valid non-NaN element for type check, robust to NaNs at the start
                    first_valid_item = all_posts_features_df[key].dropna().iloc[0] if not all_posts_features_df[key].dropna().empty else None
                    if first_valid_item is not None and isinstance(first_valid_item, (list, np.ndarray)):
                        # For lists/arrays (embeddings), stack them. Ensure all items are consistent.
                        # This might require padding/truncating if lengths vary and zero_emb fill wasn't exhaustive.
                        try:
                            post_candidates_dict[key] = np.stack(all_posts_features_df[key].values)
                        except Exception as e:
                            print(f"Error stacking column '{key}': {e}. Check embedding consistency.")
                            # Fallback or raise error
                            # For now, let's add it as is and let TF complain if it's ragged.
                            post_candidates_dict[key] = all_posts_features_df[key].values
                    else: # For scalar features
                        post_candidates_dict[key] = all_posts_features_df[key].values
                else:
                     print(f"Warning: Candidate post feature '{key}' is empty or all NaNs in all_posts_features_df. Skipping for TFRS candidates dict.")
            else:
                 print(f"Warning: Candidate post feature '{key}' (expected by model or from config) not found in all_posts_features_df columns. Missing for TFRS candidates dict.")

        # Ensure all keys truly required by the TFRS setup (especially PostTower inputs) are present
        required_tfrs_keys = ['post_id', desc_emb_col] + categorical_base_names
        for req_key in required_tfrs_keys:
            if req_key not in post_candidates_dict:
                 # This check is critical. If a key is missing here, the model will likely fail.
                 raise ValueError(f"CRITICAL: Required key '{req_key}' missing in post_candidates_dict for TFRS. Available keys: {list(post_candidates_dict.keys())}. DataFrame columns: {all_posts_features_df.columns.tolist()}")

        post_candidates_dataset = tf.data.Dataset.from_tensor_slices(post_candidates_dict)
        print(f"Post candidates dataset created with {len(all_posts_features_df)} items. Keys: {list(post_candidates_dict.keys())}")


        # Instantiate TFRS model
        # The keys passed to TFRSTwoTowerModel are the keys as they appear in the `features` dict yielded by `train_dataset`
        # data_utils.create_tf_dataset creates a flat dictionary.
        # User ID key in dataset: 'user_id' (mapped from 'uid')
        # Post ID key in dataset: 'post_id'
        # User feature keys in dataset: base names like 'about_embedding', 'headline_embedding'
        # Post feature keys in dataset: base names like 'description_embedding', 'category_id', etc.
        
        # The TFRSTwoTowerModel's _prepare_tower_inputs needs to map from these dataset keys
        # to the actual input dict for each tower.
        # The user_feature_keys and post_feature_keys in config are "view:feature".
        # The TFRSTwoTowerModel will use these to find the corresponding data in the batch.

        # Removing the explicit PostTower pre-build block as it was causing issues
        # and the FactorizedTopK candidate computation should build it correctly.
        # The main issue seems to be data consistency for the training loop.

        model = TFRSTwoTowerModel(
            user_tower=user_tower,
            post_tower=post_tower,
            user_id_key='user_id',
            post_id_key='post_id',
            user_feature_keys=user_feature_list_config,
            post_feature_keys=post_feature_list_config,
            all_post_ids_for_candidates=all_post_ids_list,
            post_dataset_for_candidates=post_candidates_dataset
        )
        
        # --- 3. Compile and Train Model ---
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']))
        
        print("Starting model training...")
        history = model.fit(train_dataset, epochs=config['epochs'], verbose=1)
        
        # Log metrics
        for metric_name, values in history.history.items():
            for epoch, value in enumerate(values):
                mlflow.log_metric(f"epoch_{metric_name}", value, step=epoch)
        final_loss = history.history['loss'][-1]
        mlflow.log_metric("final_loss", final_loss)
        # Log other TFRS metrics if available, e.g., factorized_top_k/top_100_categorical_accuracy
        for key in history.history.keys():
            if "factorized_top_k" in key or "accuracy" in key : # TFRS metrics
                 mlflow.log_metric(f"final_{key}", history.history[key][-1])


        print("Training complete.")

        # --- 4. Save and Log Model Components ---
        print("Saving model components...")

        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory for model export: {temp_dir}")

            # Define example inputs for MLflow model signatures
            # User Tower Example Input
            # Assuming about_embedding and headline_embedding have the same dimension as config['embedding_dim']
            # If they have different, pre-defined dimensions, those should be used.
            # For this example, let's use config['embedding_dim'] for them.
            user_emb_dim = config['embedding_dim'] # Common dimension for user's own embeddings
            user_inputs_example_for_signature = {
                "user_id": tf.constant(["u_example_001"], dtype=tf.string).numpy(), # Convert to numpy
                "about_embedding": tf.random.normal(shape=(1, user_emb_dim)).numpy(),
                "headline_embedding": tf.random.normal(shape=(1, user_emb_dim)).numpy()
            }

            # Post Tower Example Input
            actual_desc_emb_dim = config.get('description_embedding_dim', 384)
            post_inputs_example_for_signature = {
                "post_id": tf.constant(["p_example_001"], dtype=tf.string).numpy(), # Convert to numpy
                "description_embedding": tf.random.normal(shape=(1, actual_desc_emb_dim)).numpy(),
                "category_id": tf.constant([0], dtype=tf.int64).numpy(),
                "media_type": tf.constant([0], dtype=tf.int64).numpy(),
                "creator_id": tf.constant([0], dtype=tf.int64).numpy()
            }
        
            # Save user tower
            user_tower_path = os.path.join(temp_dir, f"user_tower_{run_id}")
            user_tower.export(user_tower_path) # Use export for SavedModel format
            mlflow.tensorflow.log_model(
                model=user_tower, # Can also pass user_tower_path here if preferred after export
                artifact_path="user_tower",
                input_example=user_inputs_example_for_signature,
                registered_model_name=f"{config['mlflow_experiment_name']}-UserTower"
            )
            print(f"User tower saved to {user_tower_path}, logged to MLflow as 'user_tower' and registered as {config['mlflow_experiment_name']}-UserTower.")

            # Save post tower
            post_tower_path = os.path.join(temp_dir, f"post_tower_{run_id}")
            post_tower.export(post_tower_path)
            mlflow.tensorflow.log_model(
                model=post_tower, # Can also pass post_tower_path here
                artifact_path="post_tower",
                input_example=post_inputs_example_for_signature,
                registered_model_name=f"{config['mlflow_experiment_name']}-PostTower"
            )
            print(f"Post tower saved to {post_tower_path}, logged to MLflow as 'post_tower' and registered as {config['mlflow_experiment_name']}-PostTower.")
            
            # The temporary directory temp_dir and its contents (user_tower_path, post_tower_path)
            # will be automatically cleaned up when the 'with' block exits.
            # No need for manual shutil.rmtree(temp_dir) unless done outside the 'with' block.

        mlflow.log_artifact(config_path, "config")
        print("Training script finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Two-Tower Recommender Model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    # Add project root to config if not present, assuming script is run from project root
    # or config paths are relative to project root.
    # For Feast, it's often easier if feast_repo_path is relative to where `feast apply` runs.
    # The train script might be run from project root: `python src/model_training/two_tower/train.py ...`
    # So, paths in config like "src/feature_repo" should be correct.
    # If config needs project_root for other paths:
    # with open(args.config, 'r') as f:
    #     temp_config = yaml.safe_load(f)
    # if 'project_root' not in temp_config:
    #     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")) # Assuming train.py is in src/model_training/two_tower
    #     temp_config['project_root'] = project_root
    #     with open(args.config, 'w') as f: # This modifies the config, maybe not ideal.
    #         yaml.dump(temp_config, f)

    main(args.config)

# Example command:
# python src/model_training/two_tower/train.py --config config/two_tower_config.yaml