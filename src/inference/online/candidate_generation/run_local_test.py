import yaml
import argparse
import os
import sys
import numpy as np

# Add the src directory to PYTHONPATH to allow direct imports
# This is useful for running scripts directly from the 'online/candidate_generation' directory
# or when the package structure isn't fully installed/recognized.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../")) # Adjust based on actual depth
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Add the directory containing the 'model.py' for UserTower/PostTower to allow MLflow to find it
MODEL_TRAINING_DIR = os.path.abspath(os.path.join(SRC_DIR, "model_training", "two_tower"))
if MODEL_TRAINING_DIR not in sys.path:
    sys.path.insert(0, MODEL_TRAINING_DIR) # Insert at the beginning

try:
    from inference.online.candidate_generation.predictor import UserTowerPredictor
    from inference.online.candidate_generation.transformer import CandidateGenerationTransformer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Ensure PYTHONPATH is set correctly or run from a location where 'inference' module is accessible.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

def load_config(config_path: str) -> dict:
    """Loads YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {e}")
        sys.exit(1)

def main(config_path: str, user_id: str, preferred_categories: list[int], top_n: int):
    """
    Main function to run a local test of the Candidate Generation service.
    """
    config = load_config(config_path)

    # --- Configuration Values ---
    PROJECT_ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
    feast_repo_path = os.path.join(PROJECT_ROOT_DIR, config["feast_repo_path"]) # Path relative to project root
    mlflow_model_uri = config["mlflow_user_tower_uri"]
    milvus_host = config["milvus_host"]
    milvus_port = config["milvus_port"]
    milvus_collection_name = config["milvus_collection_name"]
    user_features_for_tower = config["user_features_for_tower"]
    default_top_n = config.get("default_top_n_candidates", 100)
    
    # Transformer-specific Milvus params from config (if they exist, otherwise use defaults in Transformer)
    milvus_embedding_field = config.get("milvus_embedding_field_name", "post_embedding_hf")
    milvus_output_fields = config.get("milvus_output_fields", ["post_id"])
    milvus_id_field = config.get("milvus_id_field_name", "post_id")


    print("\n--- Initializing Components ---")
    # 1. Instantiate UserTowerPredictor
    # Note: This will attempt to load the model from MLflow.
    # Ensure MLflow tracking server is accessible or model is available locally if URI points there.
    try:
        print(f"Loading User Tower model from URI: {mlflow_model_uri}...")
        # For local testing, especially if MLflow server isn't running,
        # you might need to mock this or use a local model path.
        # For now, we proceed assuming MLflow is accessible.
        user_tower_predictor = UserTowerPredictor(model_uri=mlflow_model_uri)
        print("UserTowerPredictor initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize UserTowerPredictor: {e}")
        print("Ensure MLflow is configured, server is running, and model URI is correct.")
        print("For a true local test without full MLflow, you might need to mock UserTowerPredictor")
        print("or point to a locally saved model version that mlflow.pyfunc.load_model can access directly.")
        sys.exit(1)

    # 2. Instantiate CandidateGenerationTransformer
    try:
        print(f"Initializing CandidateGenerationTransformer with Feast repo: {feast_repo_path}")
        candidate_transformer = CandidateGenerationTransformer(
            feast_repo_path=feast_repo_path,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            milvus_collection_name=milvus_collection_name,
            user_tower_predictor=user_tower_predictor,
            user_feature_list=user_features_for_tower,
            embedding_field_name=milvus_embedding_field,
            output_fields=milvus_output_fields,
            id_field_name=milvus_id_field
        )
        print("CandidateGenerationTransformer initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize CandidateGenerationTransformer: {e}")
        print("Ensure Feast repository path is correct and Milvus server is accessible.")
        sys.exit(1)

    # 3. Simulate a request
    # Use provided arguments or defaults from config
    request_user_id = user_id
    request_preferred_categories = preferred_categories
    request_top_n = top_n if top_n is not None else default_top_n

    request_data = {
        "user_id": request_user_id,
        "preferred_category_ids": request_preferred_categories,
        "top_n_candidates": request_top_n
    }

    print("\n--- Simulating Candidate Generation Request ---")
    print(f"Request Data: {request_data}")

    # 4. Call generate_candidates and print results
    try:
        print("Calling transformer.generate_candidates()...")
        candidate_post_ids = candidate_transformer.generate_candidates(request_data)
        print("\n--- Candidate Generation Results ---")
        if candidate_post_ids:
            print(f"Successfully retrieved {len(candidate_post_ids)} candidate post_ids:")
            for i, post_id in enumerate(candidate_post_ids):
                print(f"  {i+1}. {post_id}")
                if i >= 19: # Print max 20 candidates for brevity
                    print(f"  ... and {len(candidate_post_ids) - (i+1)} more.")
                    break
        else:
            print("No candidate post_ids were retrieved.")
            print("This could be due to: ")
            print("  - User features not found in Feast.")
            print("  - Issues with the User Tower model prediction.")
            print("  - No matching posts in Milvus for the given embedding and category filters.")
            print("  - Milvus collection being empty or not containing relevant data.")

    except Exception as e:
        print(f"\nAn error occurred during candidate generation: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up Milvus connection if the transformer object goes out of scope
        # or if explicitly deleting. Python's GC will call __del__.
        # For explicit cleanup:
        # del candidate_transformer
        print("\nLocal test script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a local test for the Candidate Generation service.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(SCRIPT_DIR, "../../../../config/candidate_generation_service_config.yaml"), # Relative to this script
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--user_id",
        type=str,
        default="synthetic_user_test_001", # Example user ID
        help="User ID for which to generate candidates."
    )
    parser.add_argument(
        "--categories",
        type=int,
        nargs='*', # 0 or more category IDs
        default=[1, 5], # Example preferred category IDs
        help="List of preferred category IDs (space-separated integers)."
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=None, # Will use default_top_n_candidates from config if None
        help="Number of top candidates to retrieve."
    )

    args = parser.parse_args()

    # Adjust config path to be absolute if it's relative to the script's original location
    # This helps if the script is run from a different working directory.
    if not os.path.isabs(args.config):
        abs_config_path = os.path.abspath(os.path.join(SCRIPT_DIR, args.config))
    else:
        abs_config_path = args.config
    
    # Ensure the config path is correct relative to the project root if needed
    # The default path is already calculated relative to SCRIPT_DIR,
    # so it should point to `config/candidate_generation_service_config.yaml`
    # in the project root.

    print(f"Using configuration file: {abs_config_path}")

    main(
        config_path=abs_config_path,
        user_id=args.user_id,
        preferred_categories=args.categories,
        top_n=args.top_n
    )