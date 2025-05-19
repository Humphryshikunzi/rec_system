import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
DATA_DIR = "artifacts/data/train/"
OUTPUT_DIR = "artifacts/data/train/"
USER_DATA_FILE = os.path.join(DATA_DIR, "users.csv")
POST_DATA_FILE = os.path.join(DATA_DIR, "posts.csv")
INTERACTION_DATA_FILE = os.path.join(DATA_DIR, "interactions.csv")

USER_AGG_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "user_aggregated_features.parquet")
POST_AGG_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "post_aggregated_derived_features.parquet")

def generate_aggregated_features():
    """
    Loads user, post, and interaction data, computes aggregated and derived features,
    and saves them to Parquet files.
    """
    try:
        # Load data
        logging.info(f"Loading data from {USER_DATA_FILE}, {POST_DATA_FILE}, and {INTERACTION_DATA_FILE}")
        users_df = pd.read_csv(USER_DATA_FILE)
        posts_df = pd.read_csv(POST_DATA_FILE)
        interactions_df = pd.read_csv(INTERACTION_DATA_FILE)
        logging.info("Data loaded successfully.")

        # Convert timestamp columns to datetime
        posts_df['creation_timestamp'] = pd.to_datetime(posts_df['creation_timestamp'])
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])

        # --- User-Level Aggregations ---
        logging.info("Computing user-level aggregations...")
        user_aggregated_features = pd.DataFrame(users_df[['user_id']].copy())
        user_aggregated_features.set_index('user_id', inplace=True)


        # num_posts_created
        num_posts = posts_df.groupby('creator_id').size().rename('num_posts_created')
        user_aggregated_features = user_aggregated_features.join(num_posts, how='left').fillna(0)

        # count_likes_given
        likes_given = interactions_df[interactions_df['interaction_type'] == 'like'].groupby('user_id').size().rename('count_likes_given')
        user_aggregated_features = user_aggregated_features.join(likes_given, how='left').fillna(0)

        # count_comments_given
        comments_given = interactions_df[interactions_df['interaction_type'] == 'comment'].groupby('user_id').size().rename('count_comments_given')
        user_aggregated_features = user_aggregated_features.join(comments_given, how='left').fillna(0)

        # last_interaction_timestamp
        last_interaction = interactions_df.groupby('user_id')['timestamp'].max().rename('last_interaction_timestamp')
        user_aggregated_features = user_aggregated_features.join(last_interaction, how='left')
        # Add a default timestamp for users with no interactions for Feast compatibility
        user_aggregated_features['last_interaction_timestamp'] = user_aggregated_features['last_interaction_timestamp'].fillna(pd.Timestamp('1970-01-01 00:00:00+0000', tz='UTC'))


        # distinct_categories_interacted
        # Merge interactions with posts to get category_id
        interactions_with_posts = pd.merge(interactions_df, posts_df[['post_id', 'category_id']], on='post_id', how='left')
        distinct_categories = interactions_with_posts.groupby('user_id')['category_id'].nunique().rename('distinct_categories_interacted')
        user_aggregated_features = user_aggregated_features.join(distinct_categories, how='left').fillna(0)


        # num_likes_received_on_posts
        # First, count likes per post
        likes_per_post = interactions_df[interactions_df['interaction_type'] == 'like'].groupby('post_id').size().rename('num_likes_on_post_temp')
        # Merge with posts to get creator_id
        posts_with_likes = pd.merge(posts_df[['post_id', 'creator_id']], likes_per_post, on='post_id', how='left').fillna(0)
        # Aggregate likes by creator_id
        likes_received = posts_with_likes.groupby('creator_id')['num_likes_on_post_temp'].sum().rename('num_likes_received_on_posts')
        user_aggregated_features = user_aggregated_features.join(likes_received, how='left').fillna(0)

        user_aggregated_features.reset_index(inplace=True)
        # Add a timestamp column for Feast - using last_interaction_timestamp or a fixed one if not available
        user_aggregated_features['event_timestamp'] = user_aggregated_features['last_interaction_timestamp']


        logging.info("User-level aggregations computed.")

        # --- Post-Level Aggregations & Derived Features ---
        logging.info("Computing post-level aggregations and derived features...")
        post_aggregated_features = pd.DataFrame(posts_df[['post_id', 'creation_timestamp']].copy())
        post_aggregated_features.set_index('post_id', inplace=True)

        # Determine a reference time for post_age_hours (e.g., latest interaction or current time)
        # Using the latest timestamp from interactions as the reference for consistency
        reference_time = interactions_df['timestamp'].max()
        if pd.isna(reference_time): # Handle case with no interactions
            reference_time = datetime.now(pd.Timestamp(0).tz)


        # post_age_hours
        post_aggregated_features['post_age_hours'] = (reference_time - posts_df.set_index('post_id')['creation_timestamp']) / pd.Timedelta(hours=1)
        post_aggregated_features['post_age_hours'] = post_aggregated_features['post_age_hours'].fillna(0).astype(float)


        # num_likes_on_post
        likes_on_post = interactions_df[interactions_df['interaction_type'] == 'like'].groupby('post_id').size().rename('num_likes_on_post')
        post_aggregated_features = post_aggregated_features.join(likes_on_post, how='left').fillna(0)

        # num_views_on_post
        views_on_post = interactions_df[interactions_df['interaction_type'] == 'view_full'].groupby('post_id').size().rename('num_views_on_post')
        post_aggregated_features = post_aggregated_features.join(views_on_post, how='left').fillna(0)

        # num_comments_on_post
        comments_on_post = interactions_df[interactions_df['interaction_type'] == 'comment'].groupby('post_id').size().rename('num_comments_on_post')
        post_aggregated_features = post_aggregated_features.join(comments_on_post, how='left').fillna(0)

        post_aggregated_features.reset_index(inplace=True)
        # Add a timestamp column for Feast - using creation_timestamp for posts
        post_aggregated_features['event_timestamp'] = post_aggregated_features['creation_timestamp']
        # Drop the original creation_timestamp if it's redundant after setting event_timestamp
        # post_aggregated_features.drop(columns=['creation_timestamp'], inplace=True)


        logging.info("Post-level aggregations and derived features computed.")

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Save aggregated dataframes to Parquet
        logging.info(f"Saving user aggregated features to {USER_AGG_OUTPUT_FILE}")
        user_aggregated_features.to_parquet(USER_AGG_OUTPUT_FILE, index=False)

        logging.info(f"Saving post aggregated and derived features to {POST_AGG_OUTPUT_FILE}")
        post_aggregated_features.to_parquet(POST_AGG_OUTPUT_FILE, index=False)

        logging.info("Aggregated features generated and files saved successfully.")

    except FileNotFoundError as e:
        logging.error(f"Error: Input file not found. {e}")
    except Exception as e:
        logging.error(f"An error occurred during aggregated feature generation: {e}", exc_info=True)

if __name__ == "__main__":
    generate_aggregated_features()