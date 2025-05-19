from feast import FileSource

# Define a FileSource for the original users data
users_source = FileSource(
    name="users_source",
    path="../../artifacts/data/train/users.csv",  # Path relative to feature_repo directory
    timestamp_field="creation_timestamp",
    description="Original user features data source",
    field_mapping={
        "user_id": "user_id",
        "about_text": "about_text", # Will be used by embedding FV if not directly by this source
        "headline_text": "headline_text", # Will be used by embedding FV
        "onboarding_category_ids": "onboarding_category_ids",
        "creation_timestamp": "creation_timestamp"
    },
)

# Define a FileSource for the original posts data
posts_source = FileSource(
    name="posts_source",
    path="../../artifacts/data/train/posts.csv", # Path relative to feature_repo directory
    timestamp_field="creation_timestamp",
    description="Original post features data source",
    field_mapping={
        "post_id": "post_id",
        "creator_id": "creator_id",
        "description_text": "description_text", # Will be used by embedding FV
        "category_id": "category_id",
        "media_type": "media_type",
        "creation_timestamp": "creation_timestamp"
    },
)

# Define a FileSource for the original interactions data
interactions_source = FileSource(
    name="interactions_source",
    path="../../artifacts/data/train/interactions.csv", # Path relative to feature_repo directory
    timestamp_field="timestamp",
    description="User-post interaction data source",
    field_mapping={
        "interaction_id": "interaction_id",
        "user_id": "user_id",
        "post_id": "post_id",
        "interaction_type": "interaction_type",
        "timestamp": "timestamp"
    },
)

# --- New Data Sources for Transformed Features ---

# Source for users with embeddings
users_with_embeddings_source = FileSource(
    name="users_with_embeddings_source",
    path="../../artifacts/data/train/users_with_embeddings.parquet",
    timestamp_field="creation_timestamp", # Assuming original timestamp is preserved
    description="User features with text embeddings",
    # field_mapping will be inferred from Parquet, or specify if needed
    # Example:
    # field_mapping={
    #     "user_id": "user_id",
    #     "about_embedding": "about_embedding",
    #     "headline_embedding": "headline_embedding",
    #     "creation_timestamp": "creation_timestamp"
    # }
    # For Parquet, Feast can often infer schema well.
)

# Source for posts with embeddings
posts_with_embeddings_source = FileSource(
    name="posts_with_embeddings_source",
    path="../../artifacts/data/train/posts_with_embeddings.parquet",
    timestamp_field="creation_timestamp", # Assuming original timestamp is preserved
    description="Post features with text embeddings",
    # field_mapping inferred from Parquet
)

# Source for user aggregated features
user_aggregated_features_source = FileSource(
    name="user_aggregated_features_source",
    path="../../artifacts/data/train/user_aggregated_features.parquet",
    timestamp_field="event_timestamp", # As defined in the generation script
    description="Aggregated features for users",
    # field_mapping inferred from Parquet
)

# Source for post aggregated and derived features
post_aggregated_derived_features_source = FileSource(
    name="post_aggregated_derived_features_source",
    path="../../artifacts/data/train/post_aggregated_derived_features.parquet",
    timestamp_field="event_timestamp", # As defined in the generation script
    description="Aggregated and derived features for posts",
    # field_mapping inferred from Parquet
)

# Note: For Parquet files, Feast can infer the schema including column names and types.
# Explicit field_mapping and schema definitions are less critical than for CSVs,
# but can be provided for clarity or to override inference.
# The actual dtypes for features like embeddings (e.g., FloatList) will be specified
# in the FeatureView definitions.