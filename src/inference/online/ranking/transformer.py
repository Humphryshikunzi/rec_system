import pandas as pd
import numpy as np
from feast import FeatureStore
from datetime import datetime
from typing import List, Dict, Tuple

from src.inference.online.ranking.predictor import RankingModelPredictor # Assuming predictor.py is in the same directory

class RankingServiceTransformer:
    """
    Transformer class for the Ranking Service.
    Fetches features from Feast, prepares them for the model,
    and processes model output.
    Simulates KServe's Transformer component.
    """
    def __init__(
        self,
        feast_repo_path: str,
        ranking_model_predictor: RankingModelPredictor,
        feature_list: List[str],
        user_id_col: str,
        post_id_col: str,
    ):
        """
        Initializes the RankingServiceTransformer.

        Args:
            feast_repo_path (str): Path to the Feast feature repository.
            ranking_model_predictor (RankingModelPredictor): Instance of the RankingModelPredictor.
            feature_list (List[str]): List of all features required by the ranking model
                                      (format: "view_name:feature_name").
            user_id_col (str): Name of the user ID column.
            post_id_col (str): Name of the post ID column.
        """
        self.store = FeatureStore(repo_path=feast_repo_path)
        self.ranking_model_predictor = ranking_model_predictor
        self.feature_list = feature_list
        self.user_id_col = user_id_col
        self.post_id_col = post_id_col
        self.model_feature_names = self._extract_model_feature_names(feature_list)


    def _extract_model_feature_names(self, feature_list: List[str]) -> List[str]:
        """Helper to get just the feature names for DataFrame column ordering."""
        return [f.split(":")[1] if ":" in f else f for f in feature_list]

    def _fetch_features(self, user_id: str, post_ids: List[str]) -> pd.DataFrame:
        """
        Fetches historical features from Feast for a given user and list of post_ids.

        Args:
            user_id (str): The ID of the user.
            post_ids (List[str]): A list of post IDs.

        Returns:
            pd.DataFrame: A DataFrame containing the fetched features.
        """
        if not post_ids:
            return pd.DataFrame()

        entity_df = pd.DataFrame(
            {
                self.user_id_col: [user_id] * len(post_ids),
                self.post_id_col: post_ids,
                "event_timestamp": [datetime.now()] * len(post_ids), # Revert to tz-naive
            }
        )
        print(f"DEBUG: entity_df created in _fetch_features (tz-naive):\n{entity_df}")

        # Ensure feature_list contains fully qualified feature names (view:feature)
        # If some features are just names, they might need qualification or handling
        # For now, assume feature_list is correctly formatted for Feast
        
        # Filter feature_list to only include features to be fetched from Feast (those with ':')
        feast_feature_list = [f for f in self.feature_list if ":" in f]
        
        if not feast_feature_list: # If no features are to be fetched from Feast
            # Return a DataFrame with just the entity keys, so subsequent processing can add on-the-fly features
            # or handle cases where only on-the-fly features are needed.
            # However, the current _compute_on_the_fly_features expects some base columns.
            # For now, if no feast_feature_list, we might return an empty df or one with minimal entity info.
            # Let's return an empty df and let preprocess handle it, or adjust if needed.
            # A more robust solution might involve returning entity_df if no feast features are requested.
            # For this specific case, if all features were on-the-fly, this path would be taken.
            # But usually, we fetch some base features.
            # If feast_feature_list is empty, it implies an issue or a very specific use case.
            # For now, let's assume it's not empty if we expect to fetch.
            # If it IS empty and we proceed, get_historical_features might error or return empty.
            # Let's proceed with the filtered list. If it's empty, Feast will handle it (likely return empty features).
            pass

        historical_features_df = self.store.get_historical_features(
            entity_df=entity_df,
            features=feast_feature_list, # Use the filtered list
        ).to_df()

        return historical_features_df

    def _compute_on_the_fly_features(self, features_df: pd.DataFrame, user_id: str) -> pd.DataFrame:
        """
        Computes on-the-fly features.
        Placeholder for complex interaction features. For this PoC, it can compute
        simpler ones or act as a passthrough if all features are from Feast.

        Example: 'is_post_category_in_onboarding'
        Requires 'user_features_view:onboarding_category_ids' and 'post_features_view:category_id'
        to be fetched in _fetch_features.

        Args:
            features_df (pd.DataFrame): DataFrame with features fetched from Feast.
            user_id (str): The ID of the user.

        Returns:
            pd.DataFrame: DataFrame with added/modified on-the-fly features.
        """
        if features_df.empty:
            return features_df

        # Example: is_post_category_in_onboarding
        # Ensure the required columns exist from the feature fetching step
        user_onboarding_col = "onboarding_category_ids" # from user_features_view:onboarding_category_ids
        post_category_col = "category_id" # from post_features_view:category_id

        if user_onboarding_col in features_df.columns and post_category_col in features_df.columns:
            def check_category(row):
                user_categories = row[user_onboarding_col] # This might be a list or array
                post_category = row[post_category_col]
                if isinstance(user_categories, (list, np.ndarray)):
                    return 1 if post_category in user_categories else 0
                return 0 # Or handle other types/missing data appropriately

            features_df["is_post_category_in_onboarding"] = features_df.apply(check_category, axis=1)
            # Add this new feature to model_feature_names if it's used by the model
            # and not already part of the initial feature_list (as it's computed)
            if "is_post_category_in_onboarding" not in self.model_feature_names:
                 # This assumes the model was trained with this feature name.
                 # If the model expects a different name, adjust accordingly.
                 pass # Or self.model_feature_names.append("is_post_category_in_onboarding")
                      # if it's dynamically added and model expects it.

        # Add other on-the-fly feature computations here.
        # For example, if you had 'user_embedding' and 'post_embedding'
        # you could compute cosine similarity here.

        return features_df

    def preprocess(self, request_data: Dict) -> pd.DataFrame:
        """
        Preprocesses the input request data.
        Fetches features from Feast, computes on-the-fly features,
        and ensures the DataFrame matches the model's expected input.

        Args:
            request_data (Dict): Input data, e.g., {"user_id": "u1", "post_ids": ["p1", "p2"]}.

        Returns:
            pd.DataFrame: Prepared DataFrame for the RankingModelPredictor.
        """
        user_id = request_data.get(self.user_id_col)
        post_ids = request_data.get(self.post_id_col) # Assuming request_data uses post_id_col for the key

        if not user_id or not post_ids:
            raise ValueError("user_id and post_ids must be provided in request_data")

        # 1. Fetch base features from Feast
        features_df = self._fetch_features(user_id=user_id, post_ids=post_ids)

        if features_df.empty:
            # Handle case where no features could be fetched (e.g., all new posts/users)
            # Return an empty DataFrame with expected columns, or raise error, or return default scores
            # For now, return empty df, predictor should handle it or we should return empty scores.
            return pd.DataFrame(columns=self.model_feature_names)


        # 2. Compute on-the-fly features
        features_df = self._compute_on_the_fly_features(features_df, user_id)

        # 3. Ensure columns match the order and names expected by the Ranking Model
        # Select only the features the model needs and in the correct order.
        # The `self.model_feature_names` should be derived from `ranking_feature_list`
        # in the config, representing the exact feature names and order the model was trained on.

        # Check for missing columns that the model expects
        missing_cols = [col for col in self.model_feature_names if col not in features_df.columns]
        if missing_cols:
            # Handle missing features: fill with default values, error, or other strategy
            # For now, let's print a warning and they will be NaN, which model might handle or error on.
            print(f"Warning: Missing expected model features: {missing_cols}. These will be NaN.")
            for col in missing_cols:
                features_df[col] = np.nan # Or a suitable default like 0 or mean

        # Reorder and select columns
        # If 'is_post_category_in_onboarding' was computed and is part of model_feature_names,
        # it will be included here.
        final_features_df = features_df[self.model_feature_names]

        return final_features_df

    def postprocess(self, scores: np.ndarray, original_post_ids: List[str]) -> List[Tuple[str, float]]:
        """
        Postprocesses the scores from the RankingModelPredictor.

        Args:
            scores (np.ndarray): Array of scores from RankingModelPredictor.
            original_post_ids (List[str]): The list of post_ids that were scored,
                                           in the same order as the input to preprocess.

        Returns:
            List[Tuple[str, float]]: A list of ("post_id", score) tuples.
        """
        if len(scores) != len(original_post_ids):
            # This case should ideally not happen if preprocess and predict are aligned
            raise ValueError(
                f"Mismatch between number of scores ({len(scores)}) and "
                f"original post_ids ({len(original_post_ids)})."
            )
        
        # Handle cases where scores might be empty (e.g. if preprocess returned empty df)
        if scores.size == 0:
            return []

        return list(zip(original_post_ids, scores.tolist())) # Convert scores to list of floats

    def rank_candidates(self, request_data: Dict) -> List[Tuple[str, float]]:
        """
        Orchestrates the ranking process: preprocess, predict, postprocess.

        Args:
            request_data (Dict): Input data containing user_id and post_ids.
                                 e.g., {"user_id": "u1", "post_ids": ["p1", "p2"]}

        Returns:
            List[Tuple[str, float]]: A list of ("post_id", score) tuples,
                                     potentially sorted by score in descending order.
        """
        user_id = request_data.get(self.user_id_col)
        post_ids_to_rank = request_data.get(self.post_id_col) # Assuming key is post_id_col

        if not user_id or not post_ids_to_rank:
            # Or return empty list, or raise error, depending on desired behavior
            print("Warning: user_id or post_ids not provided in request_data for ranking.")
            return []
        
        if not post_ids_to_rank: # No candidates to rank
            return []

        # 1. Preprocess data to get features
        # The `preprocess` method uses `self.post_id_col` internally for the key from `request_data`
        # if it's designed that way, or it expects `post_ids` directly.
        # Let's ensure `request_data` for `preprocess` is structured as it expects.
        # The current `preprocess` expects `request_data` to have keys matching `self.user_id_col` and `self.post_id_col`.
        
        features_df = self.preprocess(request_data)

        if features_df.empty:
            print(f"No features generated for user {user_id} and posts {post_ids_to_rank}. Returning empty ranking.")
            return []

        # 2. Get predictions from the model
        scores = self.ranking_model_predictor.predict(features_df)

        # 3. Postprocess scores
        # `postprocess` needs the original post_ids in the order they were processed.
        # `features_df` rows correspond to `post_ids_to_rank` if `_fetch_features` preserves order
        # and no posts are dropped. Feast's `get_historical_features` preserves the order
        # of entities in the input `entity_df`.
        ranked_results = self.postprocess(scores, post_ids_to_rank)

        # Optionally, sort results by score
        ranked_results.sort(key=lambda x: x[1], reverse=True)

        return ranked_results

if __name__ == '__main__':
    # This is a placeholder for a very basic test.
    # A full test would require a running Feast service, MLflow, and a registered model.
    print("RankingServiceTransformer basic structure defined.")
    # Example (conceptual, would need more setup):
    # try:
    #     # 1. Mock or setup RankingModelPredictor
    #     class MockRankingModelPredictor:
    #         def predict(self, features_df: pd.DataFrame) -> np.ndarray:
    #             print(f"MockPredictor received features for {len(features_df)} items.")
    #             # Return dummy scores, one for each row in features_df
    #             return np.random.rand(len(features_df))

    #     mock_predictor = MockRankingModelPredictor()

    #     # 2. Define parameters for Transformer
    #     #    Replace with your actual Feast repo path and feature details
    #     feast_repo_path = "src/feature_repo" # Path to your feature_store.yaml's parent
    #     # This feature list should match what your model expects and what's in your Feast/on-the-fly logic
    #     # Example, ensure these features are defined in your Feast feature views
    #     # and that the _compute_on_the_fly_features handles any extras.
    #     # The names here should be the final names the model sees.
    #     # For features from Feast, it's "view_name:feature_name".
    #     # For computed features, it's just "feature_name".
    #     # The _extract_model_feature_names will get the base names.
    #     # The preprocess method will ensure the DataFrame has these columns.
    #     ranking_features = [
    #         "user_features_view:some_user_feature1",
    #         "post_features_view:some_post_feature1",
    #         "user_features_view:onboarding_category_ids", # For on-the-fly example
    #         "post_features_view:category_id",           # For on-the-fly example
    #         "is_post_category_in_onboarding" # If this is a feature name the model expects
    #     ]
    #     user_col = "user_id"
    #     post_col = "post_id"


    #     transformer = RankingServiceTransformer(
    #         feast_repo_path=feast_repo_path,
    #         ranking_model_predictor=mock_predictor,
    #         feature_list=ranking_features, # This list is used by Feast AND to define model's expected columns
    #         user_id_col=user_col,
    #         post_id_col=post_col
    #     )

    #     # 3. Simulate a request
    #     # Ensure your Feast setup has data for these entities if not mocking _fetch_features
    #     # For a real test, you'd use actual user/post IDs from your system.
    #     # The `post_id_col` here is the key in the request_data.
    #     sample_request = {
    #         user_col: "user_123", # An existing user in your Feast data
    #         post_col: ["post_abc", "post_def", "post_xyz"] # Existing posts
    #     }
    #     print(f"Simulating rank_candidates for: {sample_request}")

    #     # 4. Call rank_candidates
    #     # This will call _fetch_features, which requires Feast to be set up correctly
    #     # and have the feature views mentioned in `ranking_features`.
    #     # If Feast is not fully up, this will fail.
    #     # To test without full Feast, you'd need to mock `_fetch_features`.
    #     results = transformer.rank_candidates(sample_request)
    #     print(f"Ranked results: {results}")

    # except ImportError as ie:
    #     print(f"Import error: {ie}. Make sure Feast and other dependencies are installed.")
    # except FileNotFoundError as fe:
    #     print(f"File not found: {fe}. Is `feast_repo_path` correct and `feature_store.yaml` present?")
    # except Exception as e:
    #     print(f"An error occurred during transformer example: {e}")
    #     import traceback
    #     traceback.print_exc()