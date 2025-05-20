import yaml
import pandas as pd
import numpy as np
import os
from typing import Dict, List
import sys

# Add the project root to sys.path
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

# Imports should now work with 'src.' prefix
from src.inference.online.ranking.predictor import RankingModelPredictor
from src.inference.online.ranking.transformer import RankingServiceTransformer


def load_config(config_path: str) -> Dict:
    """Loads YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        raise

# --- Mocking for local testing without live MLflow/Feast ---
# You can enable these mocks if you don't have MLflow/Feast running
# or if the configured model/features are not available.

USE_MOCKS = os.environ.get("USE_RANKING_MOCKS", "false").lower() == "true"

class MockRankingModelPredictor:
    def __init__(self, model_uri: str, model_type: str):
        print(f"MockRankingModelPredictor initialized with URI: {model_uri}, Type: {model_type}")
        self.model_type = model_type

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        print(f"MockRankingModelPredictor: Received {len(features_df)} items to score.")
        if features_df.empty:
            print("MockRankingModelPredictor: Received empty features_df, returning empty scores.")
            return np.array([])

        print("MockRankingModelPredictor: Input features_df (relevant columns):")
        if 'is_post_category_in_onboarding' in features_df.columns:
            print(features_df[['is_post_category_in_onboarding']])
        else:
            print("MockRankingModelPredictor: 'is_post_category_in_onboarding' not in features_df.")

        base_scores = np.full(len(features_df), 0.3) # Default lowish score

        if 'is_post_category_in_onboarding' in features_df.columns:
            base_scores[features_df['is_post_category_in_onboarding'] == 1] = 0.8 # Higher score for match

        # Add some noise
        noise = np.random.uniform(-0.1, 0.1, len(features_df))
        final_scores = np.clip(base_scores + noise, 0, 1) # Clip scores to [0, 1]

        print(f"MockRankingModelPredictor: Generated scores: {final_scores}")
        return final_scores

MOCK_USER_DATA = {
    "user_test_123": {"onboarding_category_ids": [10, 20, 30]}
}

MOCK_POST_DATA = {
    "post_A": {"category_id": 10},  # Category in user's onboarding
    "post_B": {"category_id": 99},  # Category NOT in user's onboarding
    "post_C_new": {"category_id": 20}, # Category in user's onboarding
    "post_D_unseen_cat": {"category_id": 40}, # Category NOT in user's onboarding
    # post_E_random will get a random category
}
EMBEDDING_DIM = 10 # Define embedding dimension

class MockRankingServiceTransformer(RankingServiceTransformer):
    def __init__(self, feast_repo_path: str, ranking_model_predictor: RankingModelPredictor,
                 feature_list: list, user_id_col: str, post_id_col: str):
        super().__init__(feast_repo_path, ranking_model_predictor, feature_list, user_id_col, post_id_col)
        print("MockRankingServiceTransformer initialized.")
        # No Feast store for mock

    def _fetch_features(self, user_id: str, post_ids: List[str]) -> pd.DataFrame:
        print(f"Mock _fetch_features for user: {user_id}, posts: {post_ids}")
        if not post_ids:
            return pd.DataFrame()

        data_rows = []
        user_onboarding_cats = MOCK_USER_DATA.get(user_id, {}).get("onboarding_category_ids", [])

        for post_id_val in post_ids:
            post_data_point = MOCK_POST_DATA.get(post_id_val, {})
            row = {
                self.user_id_col: user_id,
                self.post_id_col: post_id_val,
            }

            # Populate features based on self.model_feature_names
            for feature_name in self.model_feature_names:
                if feature_name == "onboarding_category_ids": # User feature
                    row[feature_name] = user_onboarding_cats
                elif feature_name == "category_id": # Post feature
                    row[feature_name] = post_data_point.get("category_id", np.random.randint(1, 5)) # Default random if post not in MOCK_POST_DATA
                elif "embedding" in feature_name:
                    row[feature_name] = np.random.rand(EMBEDDING_DIM).tolist()
                elif feature_name == "num_posts_created": # User feature
                    row[feature_name] = np.random.randint(0, 100) # Example user-level feature
                elif feature_name == "num_likes_on_post": # Post feature
                    row[feature_name] = np.random.randint(0, 500)
                elif feature_name == "post_age_hours": # Post feature
                    row[feature_name] = np.random.uniform(1, 720)
                elif feature_name == "is_post_category_in_onboarding":
                    # This feature is computed on-the-fly, so it shouldn't be generated here.
                    # It will be added by the base class's preprocess method.
                    # If it's in model_feature_names, it means the model expects it.
                    pass # Do not generate, it's an output of _compute_on_the_fly_features
                else:
                    # Default for other features not explicitly handled
                    row[feature_name] = np.random.rand()
            data_rows.append(row)

        df = pd.DataFrame(data_rows)
        
        # Ensure all model_feature_names are columns, even if not explicitly set above (e.g. if new ones added to config)
        # This is important because _compute_on_the_fly_features might need some of them.
        # And the final selection in preprocess() expects all model_feature_names.
        for feature_name in self.model_feature_names:
            if feature_name not in df.columns:
                # This case should ideally be handled by the loop above.
                # If 'is_post_category_in_onboarding' is here, it's fine, it will be computed.
                if feature_name != "is_post_category_in_onboarding":
                    print(f"Warning: Mock _fetch_features: model feature '{feature_name}' was not explicitly generated. Adding with NaNs or default.")
                    # Add with NaNs or a default. For simplicity, let's use random for now if it's not an on-the-fly one.
                    df[feature_name] = np.random.rand(len(df))


        print(f"Mock _fetch_features generated DataFrame with columns: {df.columns.tolist()}")
        if not df.empty:
             print("Sample of generated data by _fetch_features (first 5 rows):")
             print(df[[self.post_id_col, "category_id", "onboarding_category_ids"]].head())
        return df

def run_test():
    """
    Runs a local test of the Ranking Service components.
    """
    # Determine the correct path to the config file
    # This assumes the script is in src/inference/online/ranking/
    # So config is ../../../config/
    # project_root_path is defined globally at the top of the script
    config_path = os.path.join(project_root_path, "config", "ranking_service_config.yaml")
    
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    feast_repo_path_relative = config["feast_repo_path"]
    # Convert relative feast_repo_path from config (relative to project root)
    # to an absolute path or path relative to this script's execution context if needed.
    # For Feast, it's often best to use a path relative to the project root.
    # Use the globally defined project_root_path
    feast_repo_full_path = os.path.join(project_root_path, feast_repo_path_relative)
    print(f"Feast repository path (resolved): {feast_repo_full_path}")


    print(f"\n--- Initializing Ranking Service Components (Mocks: {USE_MOCKS}) ---")
    try:
        if USE_MOCKS:
            ranking_predictor = MockRankingModelPredictor(
                model_uri=config["mlflow_ranking_model_uri"],
                model_type=config["ranking_model_type"]
            )
            ranking_transformer = MockRankingServiceTransformer(
                feast_repo_path=feast_repo_full_path, # Mock might not use it, but pass for consistency
                ranking_model_predictor=ranking_predictor,
                feature_list=config["ranking_feature_list"],
                user_id_col=config["user_id_col_name"],
                post_id_col=config["post_id_col_name"],
            )
        else:
            # Attempt to use actual implementations
            print("Attempting to initialize REAL RankingModelPredictor...")
            print("Note: This requires MLflow setup and the model URI to be valid.")
            ranking_predictor = RankingModelPredictor(
                model_uri=config["mlflow_ranking_model_uri"],
                model_type=config["ranking_model_type"]
            )
            print("REAL RankingModelPredictor initialized.")

            print("\nAttempting to initialize REAL RankingServiceTransformer...")
            print(f"Note: This requires Feast setup at {feast_repo_full_path} and features to be available.")
            ranking_transformer = RankingServiceTransformer(
                feast_repo_path=feast_repo_full_path,
                ranking_model_predictor=ranking_predictor,
                feature_list=config["ranking_feature_list"],
                user_id_col=config["user_id_col_name"],
                post_id_col=config["post_id_col_name"],
            )
            print("REAL RankingServiceTransformer initialized.")

    except Exception as e:
        print(f"\nError during component initialization: {e}")
        print("This might be due to missing MLflow/Feast setup or invalid configuration.")
        print("Consider setting USE_RANKING_MOCKS=true environment variable to use mocks for testing basic flow.")
        import traceback
        traceback.print_exc()
        return

    print("\n--- Simulating a Ranking Request ---")
    # Simulate a request payload. These IDs should ideally exist if not using mocks.
    # The keys here ('user_id', 'post_id') must match user_id_col_name and post_id_col_name from config
    # if the transformer's `rank_candidates` or `preprocess` methods expect them directly from request_data.
    # The current transformer.py expects keys named by config["user_id_col_name"] and config["post_id_col_name"].
    user_id_key = config["user_id_col_name"]
    post_id_key = config["post_id_col_name"]

    simulated_request = {
        user_id_key: "user_test_123",
        post_id_key: ["post_A", "post_B", "post_C_new", "post_D_unseen_cat", "post_E_random"]
    }
    print(f"Request data: {simulated_request}")

    try:
        print("\nCalling transformer.rank_candidates()...")
        ranked_candidates = ranking_transformer.rank_candidates(simulated_request)

        print("\n--- Ranking Results ---")
        if ranked_candidates:
            for post_id, score in ranked_candidates:
                print(f"Post ID: {post_id}, Score: {score:.4f}")
        else:
            print("No candidates were ranked (or an issue occurred).")

    except Exception as e:
        print(f"\nError during ranking: {e}")
        print("If not using mocks, ensure Feast data is available for the test IDs and features.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("======================================")
    print("  Ranking Service Local Test Script   ")
    print("======================================")
    print(f"To use mocks (no live MLflow/Feast needed), run with env var: USE_RANKING_MOCKS=true")
    print(f"Currently, USE_MOCKS is set to: {USE_MOCKS}")
    run_test()
    print("\nTest script finished.")