from datetime import timedelta
from feast import FeatureView, Field
from feast.types import String, Int64, UnixTimestamp, Float64, Float32, Array # Added Float64, Float32, Array

# Import entities and data sources
from entities import user, post
from data_sources import (
    users_source, # Kept for now if any feature view still uses it, but new ones use parquet
    posts_source, # Kept for now
    # interactions_source, # Likely superseded by aggregated sources
    users_with_embeddings_source,
    posts_with_embeddings_source,
    user_aggregated_features_source,
    post_aggregated_derived_features_source,
)

# User Features from original CSV (e.g., for non-embedded, non-aggregated raw features if any)
# This view can be adjusted or removed if all user features are now in users_with_embeddings_source
# or user_aggregated_features_source.
# For now, let's assume some basic user features might still be sourced directly if not in embeddings file.
# However, the instruction implies users_with_embeddings_source is comprehensive.
# Let's redefine user_features_view to use the new embedding source.

user_profile_features_view = FeatureView(
    name="user_profile_features", # Renamed for clarity
    entities=[user],
    source=users_with_embeddings_source, # Source from parquet with embeddings
    schema=[
        # Assuming 'onboarding_category_ids' is in users_with_embeddings.parquet
        # If it's a string list in parquet, Feast might infer it or it might need specific handling.
        # For Parquet, Feast often infers types like lists of strings/floats correctly.
        Field(name="onboarding_category_ids", dtype=String), # Or appropriate list type if applicable
        Field(name="about_embedding", dtype=Array(Float32)),
        Field(name="headline_embedding", dtype=Array(Float32)),
        # Other original user columns from users.csv if they are in users_with_embeddings.parquet
        # and needed as features directly can be listed here.
        # Example: Field(name="some_original_user_column", dtype=String),
    ],
    # ttl=timedelta(days=30), # Temporarily commented out for debugging
    description="User profile features including text embeddings."
)

# Post Features from original CSV (similar to user_profile_features_view)
# Redefining post_features_view to use the new embedding source.
post_details_features_view = FeatureView(
    name="post_details_features", # Renamed for clarity
    entities=[post],
    source=posts_with_embeddings_source, # Source from parquet with embeddings
    schema=[
        Field(name="category_id", dtype=String),
        Field(name="media_type", dtype=String),
        Field(name="creator_id", dtype=String), # Links to user entity
        # 'creation_timestamp' is the timestamp_field of posts_with_embeddings_source
        Field(name="creation_timestamp", dtype=UnixTimestamp),
        Field(name="description_embedding", dtype=Array(Float32)),
        # Other original post columns from posts.csv if they are in posts_with_embeddings.parquet
        # and needed as features directly can be listed here.
    ],
    # ttl=timedelta(days=30), # Temporarily commented out for debugging
    description="Post detail features including text embeddings."
)

# New FeatureView for User Aggregated Features
user_aggregated_stats_view = FeatureView(
    name="user_aggregated_stats",
    entities=[user],
    source=user_aggregated_features_source,
    schema=[
        Field(name="num_posts_created", dtype=Int64),
        Field(name="count_likes_given", dtype=Int64),
        Field(name="count_comments_given", dtype=Int64),
        Field(name="distinct_categories_interacted", dtype=Int64),
        Field(name="last_interaction_timestamp", dtype=UnixTimestamp),
        Field(name="num_likes_received_on_posts", dtype=Int64),
        # event_timestamp is the source's timestamp_field
    ],
    # ttl=timedelta(days=7), # Temporarily commented out for debugging
    description="Aggregated statistical features for users."
)

# New FeatureView for Post Aggregated and Derived Features
post_aggregated_derived_stats_view = FeatureView(
    name="post_aggregated_derived_stats",
    entities=[post],
    source=post_aggregated_derived_features_source,
    schema=[
        Field(name="post_age_hours", dtype=Float64), # Derived, can be float
        Field(name="num_likes_on_post", dtype=Int64),
        Field(name="num_views_on_post", dtype=Int64),
        Field(name="num_comments_on_post", dtype=Int64),
        # event_timestamp is the source's timestamp_field
        # creation_timestamp from the source parquet is also available if needed as a feature
        Field(name="creation_timestamp", dtype=UnixTimestamp), # If needed as a feature itself
    ],
    # ttl=timedelta(days=7), # Temporarily commented out for debugging
    description="Aggregated and derived statistical features for posts."
)


# The old user_interaction_features_view is now effectively replaced by
# user_aggregated_stats_view which uses a precomputed source.
# Commenting out the old conceptual view:
#
# user_interaction_features_view = FeatureView(
#     name="user_interaction_aggregates",
#     entities=[user],
#     source=interactions_source, # Base source for aggregations
#     schema=[
#         Field(name="count_likes_given", dtype=Int64),
#         Field(name="count_comments_given", dtype=Int64),
#         Field(name="distinct_categories_interacted", dtype=Int64),
#         Field(name="last_interaction_timestamp", dtype=UnixTimestamp),
#     ],
#     ttl=timedelta(days=7),
#     description="Aggregated features based on user interactions. (Superseded)"
# )

# Developer Notes:
# 1. Embeddings are now sourced from *_with_embeddings.parquet files.
#    Using ValueType.FLOAT_LIST as specified.
# 2. Aggregated Features are now sourced from precomputed parquet files
#    (user_aggregated_features.parquet, post_aggregated_derived_features.parquet).
# 3. Derived Features like 'post_age_hours' are also in precomputed files.
# 4. Original sources (users_source, posts_source) might be phased out if the
#    new parquet files are comprehensive and used by all feature views.
# 5. Ensure 'onboarding_category_ids' in users_with_embeddings.parquet has a type
#    that Feast can map to String or a list type if it's structured. If it's a simple
#    string (e.g. "cat1,cat2") and needs to be a list, a transformation might still be needed
#    or the generation script should prepare it as a list if Parquet supports that directly.
#    For now, assuming String is acceptable or Parquet handles list-like data well.