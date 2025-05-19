import yaml
import pandas as pd
import numpy as np
import os
from typing import Dict, List

# Adjust import paths if your project structure is different
# This assumes run_local_test.py is in src/inference/online/ranking/
# and config is in ../../../config/
# and other modules are in the same directory (ranking/)
try:
    from predictor import RankingModelPredictor
    from transformer import RankingServiceTransformer
except ImportError:
    # Fallback for running directly from project root or other locations
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
            return np.array([])
        # Simulate scores (e.g., random probabilities)
        scores = np.random.rand(len(features_df))
        print(f"MockRankingModelPredictor: Generated scores: {scores}")
        return scores

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
        
        # Create a dummy DataFrame with the expected feature columns
        # self.model_feature_names was extracted in __init__ from feature_list
        data = {}
        for feature_name in self.model_feature_names:
            # Simplistic: generate random data or zeros
            # Handle specific known features for more realistic mock
            if "embedding" in feature_name:
                 # Create a list of lists for embedding-like features
                data[feature_name] = [np.random.rand(10).tolist() for _ in range(len(post_ids))]
            elif "onboarding_category_ids" == feature_name:
                data[feature_name] = [[1,2,3] for _ in range(len(post_ids))] # Example list
            elif "category_id" == feature_name:
                 data[feature_name] = [np.random.randint(1,5) for _ in range(len(post_ids))]
            else:
                data[feature_name] = np.random.rand(len(post_ids))
        
        # Add id columns needed for internal processing if any (e.g. by _compute_on_the_fly_features)
        # The base class _fetch_features returns these, so mock should too.
        df = pd.DataFrame(data)
        df[self.user_id_col] = user_id
        df[self.post_id_col] = post_ids
        # df["event_timestamp"] = datetime.now() # Not strictly needed if not used downstream by mock

        print(f"Mock _fetch_features returning DataFrame with columns: {df.columns.tolist()}")
        return df
    
    # _compute_on_the_fly_features will be called from the parent class's preprocess method.
    # If it relies on specific data from _fetch_features, ensure the mock provides it.
    # The current _compute_on_the_fly_features for 'is_post_category_in_onboarding'
    # should work if 'onboarding_category_ids' and 'category_id' are in the mock df.

def run_test():
    """
    Runs a local test of the Ranking Service components.
    """
    # Determine the correct path to the config file
    # This assumes the script is in src/inference/online/ranking/
    # So config is ../../../config/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "..", "config", "ranking_service_config.yaml")
    
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    feast_repo_path_relative = config["feast_repo_path"]
    # Convert relative feast_repo_path from config (relative to project root)
    # to an absolute path or path relative to this script's execution context if needed.
    # For Feast, it's often best to use a path relative to the project root.
    # Assuming the script is run from project root or Feast can handle the relative path:
    project_root = os.path.join(script_dir, "..", "..", "..")
    feast_repo_full_path = os.path.join(project_root, feast_repo_path_relative)
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
        post_id_key: ["post_A", "post_B", "post_C_new"] # Mix of potentially known/unknown posts
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