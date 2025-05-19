# This file serves as the entry point for Feast to discover your feature definitions.
# Import all your defined entities, data sources, and feature views here.

from .data_sources import (
    users_source, # Original source, may or may not be used by a FV
    posts_source, # Original source, may or may not be used by a FV
    interactions_source, # Original source, likely superseded for FVs by aggregated sources
    users_with_embeddings_source,
    posts_with_embeddings_source,
    user_aggregated_features_source,
    post_aggregated_derived_features_source,
)

from .entities import (
    user,
    post
)

from .feature_views import (
    user_profile_features_view, # Updated view for user embeddings
    post_details_features_view, # Updated view for post embeddings
    user_aggregated_stats_view, # New view for user aggregated features
    post_aggregated_derived_stats_view # New view for post aggregated/derived features
    # The old user_features_view, post_features_view, and user_interaction_features_view
    # have been replaced or updated by the views above.
)

# Feast will automatically discover these imported objects when you run `feast apply`.
# You can optionally define a __all__ list, but it's not strictly necessary for Feast's discovery mechanism.
# __all__ = [
#     "users_source", "posts_source", "interactions_source",
#     "users_with_embeddings_source", "posts_with_embeddings_source",
#     "user_aggregated_features_source", "post_aggregated_derived_features_source",
#     "user", "post",
#     "user_profile_features_view", "post_details_features_view",
#     "user_aggregated_stats_view", "post_aggregated_derived_stats_view"
# ]