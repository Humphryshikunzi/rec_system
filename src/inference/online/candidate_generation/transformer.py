import numpy as np
from feast import FeatureStore
from pymilvus import connections, Collection
from typing import List, Dict, Any

# Assuming UserTowerPredictor is in the same directory or accessible via PYTHONPATH
from .predictor import UserTowerPredictor


class CandidateGenerationTransformer:
    """
    Transformer for the Candidate Generation service.
    It fetches user features, generates user embeddings (via UserTowerPredictor),
    and queries Milvus for candidate posts.
    Simulates the transformer component in a KServe inference graph.
    """

    def __init__(
        self,
        feast_repo_path: str,
        milvus_host: str,
        milvus_port: str,
        milvus_collection_name: str,
        user_tower_predictor: UserTowerPredictor,
        user_feature_list: List[str],
        embedding_field_name: str = "post_embedding_hf", # Default from Milvus indexing
        output_fields: List[str] = ["post_id"], # Default fields to retrieve from Milvus
        id_field_name: str = "post_id" # Default ID field in Milvus
    ):
        """
        Initializes the CandidateGenerationTransformer.

        Args:
            feast_repo_path (str): Path to the Feast feature repository.
            milvus_host (str): Hostname for the Milvus server.
            milvus_port (str): Port for the Milvus server.
            milvus_collection_name (str): Name of the Milvus collection to query.
            user_tower_predictor (UserTowerPredictor): Instance of UserTowerPredictor.
            user_feature_list (List[str]): List of feature names (view_name:feature_name)
                                           to fetch from Feast for the User Tower model.
            embedding_field_name (str): Name of the embedding vector field in Milvus.
            output_fields (List[str]): List of fields to return from Milvus search results.
            id_field_name (str): Name of the primary ID field in Milvus (e.g., 'post_id').
        """
        try:
            self.fs = FeatureStore(repo_path=feast_repo_path)
        except Exception as e:
            print(f"Error initializing Feast FeatureStore: {e}")
            raise

        try:
            self.milvus_conn_alias = f"cg_transformer_{milvus_collection_name}" # Unique alias
            connections.connect(
                alias=self.milvus_conn_alias,
                host=milvus_host,
                port=milvus_port
            )
            self.milvus_collection = Collection(milvus_collection_name, using=self.milvus_conn_alias)
            self.milvus_collection.load() # Ensure collection is loaded for searching
            print(f"Successfully connected to Milvus and loaded collection: {milvus_collection_name}")
        except Exception as e:
            print(f"Error connecting to Milvus or loading collection '{milvus_collection_name}': {e}")
            # Attempt to disconnect if connection was partially made
            try:
                connections.disconnect(self.milvus_conn_alias)
            except Exception:
                pass # Ignore errors during cleanup disconnect
            raise

        self.user_tower_predictor = user_tower_predictor
        self.user_feature_list = user_feature_list
        self.embedding_field_name = embedding_field_name
        self.output_fields = output_fields
        self.id_field_name = id_field_name


    def preprocess(self, request_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Fetches user features from Feast for the given user_id.

        Args:
            request_data (Dict[str, Any]): Input request data containing "user_id".
                Example: {"user_id": "some_user_id", ...}

        Returns:
            Dict[str, np.ndarray]: A dictionary of user features ready for the
                                   UserTowerPredictor. Feature values are NumPy arrays.
                                   Example: {"feature_name1": np.array(...), ...}
        """
        user_id = request_data.get("user_id")
        if not user_id:
            raise ValueError("user_id not found in request_data")

        entity_rows = [{"user_id": user_id}]

        try:
            # Ensure user_feature_list is in the format "view_name:feature_name"
            # Feast's get_online_features expects a list of "feature_name" if they are unique
            # or "view_name:feature_name" for clarity or if names clash.
            # The user_feature_list from config is expected to be in "view:name" format.
            online_features = self.fs.get_online_features(
                features=self.user_feature_list,
                entity_rows=entity_rows
            ).to_dict()

            # Prepare features for the UserTowerPredictor.
            # The User Tower model might expect features in a specific format (e.g., NumPy arrays, specific shapes).
            # This is a placeholder for any necessary transformations.
            # For example, if a feature is an embedding, it might already be a list/array.
            # If it's a scalar, it might need to be wrapped in a NumPy array.
            predictor_input_features = {}
            for feature_name_with_view in self.user_feature_list:
                # Feast returns keys like "user_features_view__feature_name"
                # We need to map them to what the model expects, or use the raw names.
                # For simplicity, let's assume the model expects keys without the view prefix,
                # or that the user_feature_list contains the exact keys the model expects
                # after Feast processing.
                
                # The key in online_features will be like 'user_features_view__about_embedding'
                # if user_feature_list contained 'user_features_view:about_embedding'.
                # Or it could be just 'about_embedding' if the feature view is implicit.
                # Let's assume the model expects keys as they are after `to_dict()` (e.g. 'user_id', 'feature_name')
                # and that `user_feature_list` contains the base names.
                
                # The `online_features` dict from Feast will have keys like 'user_id',
                # 'user_features_view__about_embedding', etc.
                # The UserTower model likely expects keys like 'user_id', 'about_embedding'.
                # We need to map them.
                
                # Let's assume the model expects keys as the part after ':' from user_feature_list
                # and 'user_id' as 'user_id'.
                
                # Simplification: Assume the model expects keys exactly as they are in user_feature_list (after splitting ':')
                # and that Feast returns them in a way that can be matched.
                # This part is highly dependent on the model's specific input signature.

                # Example: if user_feature_list is ["user_features_view:about_embedding"]
                # online_features might have "user_features_view__about_embedding"
                # The model might expect "about_embedding"
                
                # For now, let's assume the model expects the full Feast key or that
                # the user_feature_list items are directly usable as keys into the model's input.
                # This needs careful alignment with the User Tower model's actual input layer names.

                # A common pattern is that the model expects keys like "user_id", "about_embedding", etc.
                # Feast's `to_dict()` output will have keys like `user_id` (for the join key)
                # and `user_features_view__about_embedding` for features from a view.
                
                # Let's make a simplifying assumption: the model expects keys that are
                # the feature names *without* the view prefix.
                # And `user_id` is passed as `user_id`.

                processed_features = {}
                for key, value_list in online_features.items():
                    # `value_list` is a list because `get_online_features` can fetch for multiple entities.
                    # Here, we have one entity.
                    value = value_list[0]
                    
                    # Convert to NumPy array if not already.
                    # This is crucial for TensorFlow/Keras models.
                    if not isinstance(value, np.ndarray):
                        # Handle embeddings (lists of floats) or scalar features
                        if isinstance(value, list) and all(isinstance(i, (float, int)) for i in value):
                            # Likely an embedding or list feature
                            value = np.array(value, dtype=np.float32)
                            if value.ndim == 1 and key != "user_id": # Ensure embeddings are 2D (batch_size, dim)
                                 value = np.expand_dims(value, axis=0)
                        elif isinstance(value, (str, int, float, bool)):
                            # Scalar feature, wrap in array
                            if isinstance(value, str) and key == "user_id": # User ID often handled as string tensor
                                value = np.array([value], dtype=object) # TF handles string tensors
                            else:
                                value = np.array([value]) # e.g. np.array([123])
                        else:
                            # Fallback or error for unexpected types
                            print(f"Warning: Feature '{key}' has unexpected type {type(value)}. Converting to array.")
                            value = np.array(value)

                    # Map Feast output keys to model input keys if necessary.
                    # Example: 'user_features_view__about_embedding' -> 'about_embedding'
                    model_input_key = key.split('__')[-1] if '__' in key else key
                    processed_features[model_input_key] = value

                # Ensure all features expected by the model (derived from user_feature_list) are present.
                # And add user_id if it's part of the model's features.
                final_predictor_input = {}
                expected_model_keys = [f.split(':')[-1] for f in self.user_feature_list]
                if "user_id" not in expected_model_keys and "user_id" in processed_features:
                     final_predictor_input["user_id"] = processed_features["user_id"]


                for model_key in expected_model_keys:
                    if model_key in processed_features:
                        final_predictor_input[model_key] = processed_features[model_key]
                    else:
                        # This could happen if a feature in user_feature_list wasn't found or mapped correctly.
                        # Or if user_feature_list contains "view:feature" and model expects "feature"
                        # and the mapping above didn't catch it.
                        # For now, we assume `processed_features` has the right keys.
                        # This logic needs to be robust based on actual model input names.
                        print(f"Warning: Expected model key '{model_key}' not found in processed Feast features. This might cause issues.")
                        # Potentially fill with a default or raise an error.
                
                # The `processed_features` should now contain keys like 'user_id', 'about_embedding', etc.
                # with values as numpy arrays.
                return processed_features # Return all processed features from Feast

        except Exception as e:
            print(f"Error during preprocess for user {user_id}: {e}")
            raise

    def postprocess(self, user_embedding: np.ndarray, request_data: Dict[str, Any]) -> List[str]:
        """
        Queries Milvus for candidate posts using the user embedding and filters.

        Args:
            user_embedding (np.ndarray): The user embedding from UserTowerPredictor.
                                         Expected shape: (1, embedding_dim).
            request_data (Dict[str, Any]): Original request data, containing
                                           "preferred_category_ids" and "top_n_candidates".

        Returns:
            List[str]: A list of candidate post_ids.
        """
        preferred_category_ids = request_data.get("preferred_category_ids")
        top_n = request_data.get("top_n_candidates", 100) # Default to 100 if not provided

        if user_embedding.ndim == 1: # Ensure it's (1, dim)
            user_embedding_list = [user_embedding.tolist()]
        elif user_embedding.ndim == 2 and user_embedding.shape[0] == 1:
            user_embedding_list = user_embedding.tolist()
        else:
            raise ValueError(f"User embedding has unexpected shape: {user_embedding.shape}. Expected (1, dim) or (dim,).")

        # Construct Milvus expression filter for categories
        expr = ""
        if preferred_category_ids and isinstance(preferred_category_ids, list) and len(preferred_category_ids) > 0:
            # Assuming 'category_id' is a scalar field in Milvus.
            # If 'category_id' is an array and you need to check for overlap, the expression is more complex.
            # Constructing filter as (category_id == val1 || category_id == val2 || ...)
            # as an alternative to IN [...] to see if it resolves parsing issues.
            if len(preferred_category_ids) == 1:
                expr = f"category_id == {preferred_category_ids[0]}"
            else:
                individual_conditions = [f"category_id == {cid}" for cid in preferred_category_ids]
                expr = f"({ ' || '.join(individual_conditions) })"
            print(f"Milvus search expression: {expr}")
        else:
            print("No preferred_category_ids provided or empty list, searching without category filter.")


        search_params = {
            "metric_type": "L2", # Or "IP" depending on your indexing and distance metric
            "params": {"nprobe": 10}, # Example search parameters, tune as needed
        }

        try:
            print(f"Searching Milvus collection '{self.milvus_collection.name}' with top_n={top_n}, expr='{expr}'")
            results = self.milvus_collection.search(
                data=user_embedding_list,
                anns_field=self.embedding_field_name, # Field name of the embedding vector in Milvus
                param=search_params,
                limit=top_n,
                expr=expr if expr else None, # Pass None if no filter
                output_fields=self.output_fields, # e.g., ["post_id", "category_id"]
                consistency_level="Strong" # Or your desired consistency level
            )
        except Exception as e:
            print(f"Error searching Milvus: {e}")
            raise

        candidate_post_ids = []
        if results:
            for hits in results: # Results is a list of Hits objects (one per query vector)
                for hit in hits:
                    # The 'output_fields' determines what's in hit.entity.
                    # Access field using hit.entity.get(field_name)
                    retrieved_id = hit.entity.get(self.id_field_name)
                    if retrieved_id is not None:
                        candidate_post_ids.append(str(retrieved_id))
                    else:
                        # If the primary key is the ID and it's not in output_fields,
                        # it might be available as hit.id (especially if id_field_name is the primary key name).
                        # However, it's best practice to include the ID field in output_fields.
                        # For now, we'll log if it's not found via .get()
                        all_available_fields = list(hit.entity.fields) # Get all fields present in the entity
                        print(f"Warning: '{self.id_field_name}' not found directly in Milvus hit entity via .get(). Available fields: {all_available_fields}. Trying hit.id as fallback if id_field_name is primary.")
                        # As a fallback, if the id_field_name is indeed the primary key, hit.id might contain it.
                        # This behavior can vary, so relying on output_fields is safer.
                        # Let's assume for now that if .get() fails, the field is truly missing from output.
                        # If hit.id is relevant, the logic would be:
                        # if self.id_field_name == collection_primary_key_name and hasattr(hit, 'id'):
                        #    candidate_post_ids.append(str(hit.id))
                        # else:
                        #    print(f"Field '{self.id_field_name}' also not found as hit.id or not primary key.")
                        pass # Field not found in entity

        print(f"Retrieved {len(candidate_post_ids)} candidate post_ids from Milvus.")
        return candidate_post_ids


    def generate_candidates(self, request_data: Dict[str, Any]) -> List[str]:
        """
        Orchestrates the candidate generation process:
        1. Preprocess request (fetch user features).
        2. Predict user embedding.
        3. Postprocess (query Milvus for candidates).

        Args:
            request_data (Dict[str, Any]): Input request data.
                Example: {"user_id": "...", "preferred_category_ids": [...], "top_n_candidates": ...}

        Returns:
            List[str]: A list of candidate post_ids.
        """
        print(f"Generating candidates for request: {request_data}")
        try:
            # 1. Preprocess: Fetch user features for the predictor
            # This should return a dict suitable for user_tower_predictor.predict()
            predictor_input_features = self.preprocess(request_data)
            print(f"Features for predictor: {list(predictor_input_features.keys())}")

            # 2. Predict: Generate user embedding
            user_embedding = self.user_tower_predictor.predict(predictor_input_features)
            print(f"Generated user embedding shape: {user_embedding.shape}")

            # 3. Postprocess: Query Milvus with the embedding and filters
            candidate_post_ids = self.postprocess(user_embedding, request_data)
            print(f"Final candidate post_ids: {candidate_post_ids[:10]}... (total {len(candidate_post_ids)})")

            return candidate_post_ids
        except Exception as e:
            print(f"Error in generate_candidates pipeline: {e}")
            # Depending on desired error handling, re-raise or return empty/error indicator
            raise

    def __del__(self):
        """Clean up Milvus connection."""
        try:
            if hasattr(self, 'milvus_conn_alias') and self.milvus_conn_alias:
                print(f"Disconnecting from Milvus alias: {self.milvus_conn_alias}")
                connections.disconnect(self.milvus_conn_alias)
        except Exception as e:
            print(f"Error during Milvus disconnection: {e}")

if __name__ == '__main__':
    # This is a placeholder for a very basic test.
    # A full test would require a running Feast, MLflow, and Milvus setup,
    # and a valid configuration.
    print("CandidateGenerationTransformer basic check.")
    print("To run a proper test, ensure all services (Feast, MLflow, Milvus) are running,")
    print("models are registered, features are materialized, and Milvus is populated.")
    print("Use the run_local_test.py script with a valid configuration for a functional test.")
    # Example (conceptual, won't run without setup):
    # try:
    #     # Dummy predictor
    #     class DummyPredictor:
    #         def predict(self, features):
    #             print(f"DummyPredictor received features: {list(features.keys())}")
    #             return np.random.rand(1, 128).astype(np.float32) # Example embedding dim

    #     config_example = {
    #         "feast_repo_path": "src/feature_repo", # Needs to be a valid Feast repo
    #         "milvus_host": "localhost",
    #         "milvus_port": "19530",
    #         "milvus_collection_name": "recsys_poc_posts", # Needs to exist and be populated
    #         "user_feature_list": [
    #             "user_features_view:about_embedding",
    #             "user_features_view:headline_embedding"
    #         ], # Match features in your Feast setup and User Tower model
    #         "mlflow_user_tower_uri": "models:/TwoTowerUserTower/Production" # For the real predictor
    #     }

    #     # predictor_instance = UserTowerPredictor(config_example["mlflow_user_tower_uri"]) # Real
    #     predictor_instance = DummyPredictor() # For this basic check

    #     transformer = CandidateGenerationTransformer(
    #         feast_repo_path=config_example["feast_repo_path"],
    #         milvus_host=config_example["milvus_host"],
    #         milvus_port=config_example["milvus_port"],
    #         milvus_collection_name=config_example["milvus_collection_name"],
    #         user_tower_predictor=predictor_instance,
    #         user_feature_list=config_example["user_feature_list"]
    #     )
    #     print("Transformer initialized (conceptually).")
        
    #     # test_request = {
    #     #     "user_id": "some_synthetic_user_id_0", # Ensure this user exists in Feast or features can be generated
    #     #     "preferred_category_ids": [1, 5],
    #     #     "top_n_candidates": 10
    #     # }
    #     # candidates = transformer.generate_candidates(test_request)
    #     # print(f"Test run generated candidates: {candidates}")

    # except Exception as e:
    #     print(f"Could not run conceptual example: {e}")
    #     print("This is expected if services are not running or paths are incorrect.")
    pass