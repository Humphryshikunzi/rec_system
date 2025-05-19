import argparse
import csv
import logging
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml
from faker import Faker
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_config(config_path):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        raise


def generate_users(num_users, num_categories, fake, start_datetime, end_datetime):
    """Generates synthetic user data."""
    users = []
    for i in range(num_users):
        created_at = fake.date_time_between(
            start_date=start_datetime, end_date=end_datetime
        )
        users.append(
            {
                "user_id": f"u_{i:04d}",
                "about_text": fake.paragraph(nb_sentences=3),
                "headline_text": fake.sentence(nb_words=6),
                "onboarding_category_ids": sorted(
                    random.sample(
                        [f"cat_{j:02d}" for j in range(num_categories)],
                        k=random.randint(1, min(5, num_categories)),
                    )
                ),
                "creation_timestamp": created_at.isoformat(),
            }
        )
    logging.info(f"Generated {len(users)} users.")
    return pd.DataFrame(users)


def generate_posts(
    num_posts, users_df, num_categories, fake, start_datetime, end_datetime
):
    """Generates synthetic post data."""
    posts = []
    user_ids = users_df["user_id"].tolist()
    user_creation_map = pd.to_datetime(
        users_df.set_index("user_id")["creation_timestamp"]
    )

    for i in range(num_posts):
        creator_id = random.choice(user_ids)
        # Ensure post is created after user
        min_post_time = user_creation_map[creator_id]
        if min_post_time >= end_datetime: # If user created at the very end, shift post time slightly
            post_created_at = end_datetime
        else:
            post_created_at = fake.date_time_between(
                start_date=min_post_time, end_date=end_datetime
            )

        posts.append(
            {
                "post_id": f"p_{i:05d}",
                "creator_id": creator_id,
                "description_text": fake.paragraph(
                    nb_sentences=random.randint(1, 5)
                ),
                "category_id": f"cat_{random.randint(0, num_categories - 1):02d}",
                "media_type": random.choice(["text", "image", "video"]),
                "creation_timestamp": post_created_at.isoformat(),
            }
        )
    logging.info(f"Generated {len(posts)} posts.")
    return pd.DataFrame(posts)


def generate_interactions(
    users_df, posts_df, avg_interactions_per_user, fake, end_datetime
):
    """Generates synthetic interaction data."""
    interactions = []
    user_ids = users_df["user_id"].tolist()
    post_ids = posts_df["post_id"].tolist()
    num_interactions = len(user_ids) * avg_interactions_per_user

    # Create maps for faster lookups of creation timestamps
    user_creation_map = pd.to_datetime(
        users_df.set_index("user_id")["creation_timestamp"]
    )
    post_creation_map = pd.to_datetime(
        posts_df.set_index("post_id")["creation_timestamp"]
    )

    interaction_types = [
        "like",
        "view_full",
        "comment",
        "share",
        "bookmark",
        "dislike",
        "report",
    ]
    interaction_id_counter = 0

    for _ in range(num_interactions):
        user_id = random.choice(user_ids)
        post_id = random.choice(post_ids)

        user_created_at = user_creation_map[user_id]
        post_created_at = post_creation_map[post_id]

        # Interaction must happen after both user and post are created
        min_interaction_time = max(user_created_at, post_created_at)

        if min_interaction_time >= end_datetime:
            # If min interaction time is at or after end_date, skip or adjust
            # For simplicity, we'll set it to end_datetime if it's very close
            # or skip if it's impossible. Here, let's try to make it happen at end_datetime.
            interaction_timestamp = end_datetime
        else:
            interaction_timestamp = fake.date_time_between(
                start_date=min_interaction_time, end_date=end_datetime
            )
        
        # Ensure interaction_timestamp is not before min_interaction_time
        if interaction_timestamp < min_interaction_time:
            interaction_timestamp = min_interaction_time + timedelta(seconds=random.randint(1, 3600))
            if interaction_timestamp > end_datetime:
                 interaction_timestamp = end_datetime


        interactions.append(
            {
                "interaction_id": f"i_{interaction_id_counter:07d}",
                "user_id": user_id,
                "post_id": post_id,
                "interaction_type": random.choice(interaction_types),
                "timestamp": interaction_timestamp.isoformat(),
            }
        )
        interaction_id_counter += 1
    logging.info(f"Generated {len(interactions)} interactions.")
    return pd.DataFrame(interactions)


def save_data(df, output_dir, entity_name, split_name):
    """Saves a DataFrame to a CSV file."""
    if df is None or df.empty:
        logging.warning(f"DataFrame for {entity_name} ({split_name}) is empty. Skipping save.")
        return
    os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)
    file_path = os.path.join(output_dir, split_name, f"{entity_name}.csv")
    try:
        df.to_csv(file_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        logging.info(f"Saved {entity_name} data to {file_path}")
    except Exception as e:
        logging.error(f"Error saving {entity_name} to {file_path}: {e}")
        raise

def split_data(interactions_df, users_df, posts_df, split_ratios, output_dir):
    """
    Splits interaction data by time and ensures corresponding users/posts
    are correctly assigned or duplicated if necessary.
    For simplicity in this initial version, we'll split interactions and then
    filter users and posts based on the interactions in each split.
    A more robust approach might involve ensuring all user/post history prior
    to an interaction's split point is included.
    """
    # Sort interactions by timestamp
    interactions_df = interactions_df.sort_values(by="timestamp").reset_index(drop=True)

    n_total = len(interactions_df)
    n_train = int(n_total * split_ratios["train"])
    n_val = int(n_total * split_ratios["validation"])

    train_interactions = interactions_df.iloc[:n_train]
    val_interactions = interactions_df.iloc[n_train : n_train + n_val]
    test_interactions = interactions_df.iloc[n_train + n_val :]

    logging.info(f"Interaction splits: Train={len(train_interactions)}, Val={len(val_interactions)}, Test={len(test_interactions)}")

    # Save interaction splits
    save_data(train_interactions, output_dir, "interactions", "train")
    save_data(val_interactions, output_dir, "interactions", "validation")
    save_data(test_interactions, output_dir, "interactions", "test")

    # Filter users and posts for each split
    # This ensures that only users/posts relevant to the interactions in a split are included.
    # Note: A user or post might appear in multiple splits if they have interactions across time boundaries.
    # This is a common way to handle it, or one might choose to assign users/posts to a single split.

    for split_name, int_df in [("train", train_interactions), ("validation", val_interactions), ("test", test_interactions)]:
        if int_df is not None and not int_df.empty:
            split_user_ids = int_df["user_id"].unique()
            split_post_ids = int_df["post_id"].unique()

            split_users = users_df[users_df["user_id"].isin(split_user_ids)]
            split_posts = posts_df[posts_df["post_id"].isin(split_post_ids)]
            
            # Also include creators of posts present in the interactions, even if they didn't interact
            post_creator_ids = posts_df[posts_df["post_id"].isin(split_post_ids)]["creator_id"].unique()
            additional_users = users_df[users_df["user_id"].isin(post_creator_ids)]
            split_users = pd.concat([split_users, additional_users]).drop_duplicates(subset=["user_id"]).reset_index(drop=True)


            save_data(split_users, output_dir, "users", split_name)
            save_data(split_posts, output_dir, "posts", split_name)
        else:
            logging.info(f"No interactions for {split_name} split, so no users/posts files will be generated for it.")


def main(config_path):
    """Main function to generate synthetic data."""
    try:
        config = load_config(config_path)
    except Exception:
        logging.error("Failed to load configuration. Exiting.")
        return

    # Apply random seed
    seed = config.get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)
    fake = Faker()

    logging.info(f"Using random seed: {seed}")

    num_users = config.get("num_users", 100)
    num_posts = config.get("num_posts", 1000)
    num_categories = config.get("num_categories", 10)
    avg_interactions_per_user = config.get("avg_interactions_per_user", 20)
    output_dir = config.get("output_dir", "artifacts/data")
    split_ratios = config.get("split_ratios", {"train": 0.7, "validation": 0.15, "test": 0.15})

    try:
        start_date_str = config.get("start_date", "2023-01-01")
        end_date_str = config.get("end_date", "2023-12-31")
        start_datetime = datetime.fromisoformat(start_date_str + "T00:00:00")
        end_datetime = datetime.fromisoformat(end_date_str + "T23:59:59")
    except ValueError as e:
        logging.error(f"Invalid date format in configuration: {e}. Dates should be YYYY-MM-DD.")
        return

    if start_datetime >= end_datetime:
        logging.error("Start date must be before end date.")
        return

    # Create base output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate data
    users_df = generate_users(num_users, num_categories, fake, start_datetime, end_datetime)
    posts_df = generate_posts(num_posts, users_df, num_categories, fake, start_datetime, end_datetime)
    interactions_df = generate_interactions(users_df, posts_df, avg_interactions_per_user, fake, end_datetime)

    if interactions_df.empty:
        logging.error("No interactions were generated. Cannot proceed with splitting and saving.")
        return

    # Split and save data
    split_data(interactions_df, users_df, posts_df, split_ratios, output_dir)

    logging.info("Synthetic data generation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for a social recommender system."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/data_config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    main(args.config)