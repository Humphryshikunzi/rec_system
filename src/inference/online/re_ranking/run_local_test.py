import yaml
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

# Adjust the path to import ReRankService from its location
# This assumes run_local_test.py is in src/inference/online/re_ranking/
# and re_rank_service.py is in the same directory.
from re_rank_service import ReRankService

def load_config(config_path):
    """Loads YAML configuration."""
    # Construct an absolute path to the config file
    # Assuming this script is in src/inference/online/re_ranking
    # and config is in ../../../config/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    abs_config_path = os.path.join(base_dir, "..", "..", "..", config_path)
    
    if not os.path.exists(abs_config_path):
        print(f"Error: Config file not found at {abs_config_path}")
        # Try a path relative to workspace root as a fallback for different execution contexts
        workspace_root_config_path = os.path.join(os.getcwd(), config_path)
        if os.path.exists(workspace_root_config_path):
            abs_config_path = workspace_root_config_path
            print(f"Found config at workspace relative path: {abs_config_path}")
        else:
            print(f"Also not found at {workspace_root_config_path}. Please check the path.")
            return None

    with open(abs_config_path, 'r') as f:
        return yaml.safe_load(f)

def get_mock_post_metadata(post_ids: list) -> dict:
    """
    Mocks the _get_post_metadata method for local testing.
    Provides sample category_id and creation_timestamp for given post_ids.
    """
    mock_data = {}
    now = datetime.now(timezone.utc)
    for i, pid in enumerate(post_ids):
        # Simulate different categories and creation times
        category = f"cat_{(i % 3) + 1}" # cat_1, cat_2, cat_3
        if i < len(post_ids) // 4 : # ~First quarter are "new"
            creation_time = now - timedelta(hours=(i * 2) + 1)
        elif i < len(post_ids) // 2: # ~Second quarter are moderately old
            creation_time = now - timedelta(days=(i - len(post_ids) // 4) + 1)
        else: # Rest are older
            creation_time = now - timedelta(days=30 + (i - len(post_ids) // 2))
        
        mock_data[pid] = {
            "category_id": category,
            "creation_timestamp": creation_time # Feast typically returns datetime objects
        }
    return mock_data

def run_test():
    """
    Demonstrates instantiating and using the ReRankService with mocked metadata.
    """
    # 1. Load Configuration
    config_file_path = "config/re_ranking_service_config.yaml"
    config = load_config(config_file_path)
    if not config:
        print("Failed to load configuration. Exiting test.")
        return

    print(f"Configuration loaded: {config}")

    # 2. Instantiate ReRankService
    # We don't need a real feast_repo_path if we are mocking _get_post_metadata
    # However, the class constructor expects it.
    # For a more robust mock, you might mock the FeatureStore object itself.
    try:
        re_rank_service = ReRankService(
            feast_repo_path=config.get("feast_repo_path", "src/feature_repo"), # Path from config
            category_diversity_factor=config.get("category_diversity_factor", 0.3),
            new_post_boost_hours=config.get("new_post_boost_hours", 48),
            new_post_boost_factor=config.get("new_post_boost_factor", 1.2)
        )
    except Exception as e:
        print(f"Error instantiating ReRankService: {e}")
        print("This might be due to Feast trying to connect. Ensure Feast is installed, or mock FeatureStore.")
        return


    # Ensure some posts can be identified as "new" and have varying categories.
    # This is now a template, the actual metadata (categories, timestamps) comes from true_mock_metadata
    simulated_scored_posts_template = [
        ("post_A", 0.95),
        ("post_B", 0.92),
        ("post_C", 0.90),
        ("post_D", 0.88),
        ("post_E", 0.85),
        ("post_F", 0.82),
        ("post_G", 0.80),
        ("post_H", 0.78),
        ("post_I", 0.75),
        ("post_J", 0.72),
        ("post_K", 0.70),
        ("post_L", 0.68)
    ]
    # Add more posts to test diversity and truncation
    for i in range(10): # This will create post_M0-M9, N0-N9, O0-O9
        simulated_scored_posts_template.append((f"post_M{i}", 0.65 - i*0.01))
        simulated_scored_posts_template.append((f"post_N{i}", 0.64 - i*0.01))
        simulated_scored_posts_template.append((f"post_O{i}", 0.63 - i*0.01))

    # Extract post_ids to pass to the mocked metadata function
    # The mock function will use these to generate diverse metadata
    all_post_ids_for_mock = [pid for pid, _ in simulated_scored_posts_template] # Use template here
    
    # Generate the "true" mock metadata that the service will use
    true_mock_metadata = get_mock_post_metadata(all_post_ids_for_mock)

    # 3. Mock the _get_post_metadata method
    # This is crucial for local testing without a live Feast setup.
    def mock_side_effect_func(post_ids_input_to_service):
        # The service will call this with a list of post_ids.
        # We return the relevant subset from our pre-generated true_mock_metadata.
        return {pid: true_mock_metadata[pid] for pid in post_ids_input_to_service if pid in true_mock_metadata}

    re_rank_service._get_post_metadata = MagicMock(side_effect=mock_side_effect_func)
    print("ReRankService._get_post_metadata has been mocked to use pre-generated consistent metadata.")

    # 4. Simulate a list of scored posts (using the template)
    # (post_id, score)
    # The actual list of posts passed to the service
    simulated_scored_posts = simulated_scored_posts_template[:]


    print(f"\nSimulated scored posts (first 5): {simulated_scored_posts[:5]}")
    print(f"Total simulated posts: {len(simulated_scored_posts)}")

    # 5. Call re_rank_and_present()
    num_recommendations = config.get("default_num_final_recommendations", 10)
    print(f"\nRequesting {num_recommendations} final recommendations...")

    try:
        final_recommendations = re_rank_service.re_rank_and_present(
            scored_posts=simulated_scored_posts,
            num_final_recommendations=num_recommendations
        )
    except Exception as e:
        print(f"Error during re_rank_and_present: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Print Results
    print("\n--- Final Re-ranked Recommendations ---")
    if final_recommendations:
        for i, post_id in enumerate(final_recommendations):
            # Use the true_mock_metadata for display to ensure consistency
            meta_for_display = true_mock_metadata.get(post_id, {})
            category = meta_for_display.get("category_id", "N/A")
            creation_ts = meta_for_display.get("creation_timestamp")
            
            is_new_str = ""
            if creation_ts and (datetime.now(timezone.utc) - creation_ts) <= timedelta(hours=config.get("new_post_boost_hours", 48)):
                 is_new_str = "(Boosted)"
            
            print(f"{i+1}. Post ID: {post_id} (Category: {category}) {is_new_str}")
    else:
        print("No recommendations were generated.")

    print(f"\nTotal recommendations returned: {len(final_recommendations)}")

    # Verify the mock was called
    re_rank_service._get_post_metadata.assert_called()
    # print(f"Mock _get_post_metadata call args: {re_rank_service._get_post_metadata.call_args}")


if __name__ == "__main__":
    print("Running Re-ranking Service Local Test Script...")
    run_test()
    print("\nTest script finished.")