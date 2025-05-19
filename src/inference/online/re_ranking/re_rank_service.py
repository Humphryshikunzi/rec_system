import yaml
from feast import FeatureStore
from datetime import datetime, timedelta
from collections import Counter

class ReRankService:
    def __init__(self, feast_repo_path: str, category_diversity_factor: float = 0.2, new_post_boost_hours: int = 24, new_post_boost_factor: float = 1.1):
        self.store = FeatureStore(repo_path=feast_repo_path)
        self.category_diversity_factor = category_diversity_factor
        self.new_post_boost_hours = new_post_boost_hours
        self.new_post_boost_factor = new_post_boost_factor

    def _get_post_metadata(self, post_ids: list) -> dict:
        """
        Fetches category_id and creation_timestamp for the given post_ids from Feast.
        Returns a dictionary: {post_id: {"category_id": cat_id, "creation_timestamp": ts}}.
        """
        if not post_ids:
            return {}

        # Ensure post_ids are strings, as Feast expects entity keys to be strings
        entity_rows = [{"post_id": str(pid)} for pid in post_ids]
        
        # Define features to retrieve
        # These should match the feature view and feature names in your Feast definition
        features_to_retrieve = [
            "post_features_view:category_id",
            "post_features_view:post_creation_timestamp" 
        ]

        try:
            feature_vector = self.store.get_online_features(
                features=features_to_retrieve,
                entity_rows=entity_rows
            ).to_dict()
        except Exception as e:
            print(f"Error fetching features from Feast: {e}")
            # Fallback or error handling: return empty metadata or raise
            return {pid: {"category_id": None, "creation_timestamp": None} for pid in post_ids}

        post_metadata = {}
        for i, post_id_str in enumerate(feature_vector["post_id"]):
            # Feast might return post_id as string, ensure consistency if original post_ids were int
            original_post_id = post_ids[i] # Assuming order is maintained and matches input post_ids
            post_metadata[original_post_id] = {
                "category_id": feature_vector["category_id"][i],
                "creation_timestamp": feature_vector["post_creation_timestamp"][i] # Ensure this is a datetime object or parseable
            }
        return post_metadata

    def apply_new_post_boost(self, scored_posts: list, post_metadata: dict) -> list:
        """
        Iterates through scored_posts [("post_id", score), ...].
        If a post's creation_timestamp (from post_metadata) is within new_post_boost_hours from now,
        multiply its score by new_post_boost_factor.
        Returns the list with potentially updated scores.
        """
        boosted_posts = []
        now = datetime.utcnow() # Use UTC if timestamps are in UTC

        for post_id, score in scored_posts:
            metadata = post_metadata.get(post_id)
            if metadata and metadata.get("creation_timestamp"):
                # Ensure creation_timestamp is datetime object
                creation_ts = metadata["creation_timestamp"]
                if isinstance(creation_ts, (int, float)): # Assuming timestamp might be Unix epoch
                    creation_ts = datetime.utcfromtimestamp(creation_ts)
                elif isinstance(creation_ts, str): # Or a string that needs parsing
                    try:
                        creation_ts = datetime.fromisoformat(creation_ts.replace('Z', '+00:00')) # Handle 'Z' for UTC
                    except ValueError:
                        try:
                            creation_ts = datetime.strptime(creation_ts, "%Y-%m-%d %H:%M:%S.%f%z") # Example format
                        except ValueError:
                             print(f"Warning: Could not parse creation_timestamp '{creation_ts}' for post {post_id}")
                             creation_ts = None # Or handle as an old post

                if creation_ts and (now - creation_ts) <= timedelta(hours=self.new_post_boost_hours):
                    score *= self.new_post_boost_factor
            boosted_posts.append((post_id, score))
        return boosted_posts

    def apply_category_diversity(self, scored_posts: list, post_metadata: dict, num_final_recommendations: int) -> list:
        """
        Sorts scored_posts by score in descending order.
        Implements a simple category diversification algorithm.
        Returns the diversified and ordered list of post_ids.
        """
        # Sort by score descending
        sorted_posts_with_scores = sorted(scored_posts, key=lambda x: x[1], reverse=True)

        final_recommendations = []
        category_counts = Counter()
        # Window size for checking category dominance, can be related to num_final_recommendations
        # or a fixed small number. For simplicity, let's consider a dynamic window or overall proportion.
        
        # Max allowed items from a single category in the final list (heuristic)
        # This factor means no single category should make up more than (factor * 100)% of the list.
        # For example, if factor is 0.3 and num_final is 20, max 6 items from one category.
        # This is a simple interpretation. A more robust way is to penalize runs of same category.
        
        # Let's try a simpler greedy approach with a check for too many from one category.
        # We want to avoid too many *consecutive* items from the same category.
        # A more direct interpretation of category_diversity_factor could be:
        # if category_diversity_factor = 0.3, it means we try to ensure that
        # in any small window, no more than (1 - 0.3) = 70% are from the same category,
        # or we try to ensure at least 30% are from different categories.

        # Simpler approach: Greedily select, but try to avoid too many from one category overall.
        # And penalize/skip if too many *recent* items are from the same category.
        
        # Let's use a window for recent category check
        recent_category_window_size = 5 # Check last 5 items for diversity
        
        temp_skipped_posts = [] # Posts skipped due to diversity, to reconsider later

        for post_id, score in sorted_posts_with_scores:
            if len(final_recommendations) >= num_final_recommendations:
                break

            category_id = post_metadata.get(post_id, {}).get("category_id", "unknown_category")

            # Check 1: Overall category balance (simple version)
            # No more than X% of total recommendations from one category
            max_per_category = max(1, int(num_final_recommendations * (self.category_diversity_factor + 0.2))) # e.g. 0.3 -> 50% max
            
            # Check 2: Recent category dominance (more important for perceived diversity)
            # Avoid too many consecutive items from the same category.
            # If the last 'k' items have too many from this category, skip.
            is_dominant_in_recent = False
            if len(final_recommendations) >= recent_category_window_size:
                recent_categories = [post_metadata.get(pid, {}).get("category_id", "unknown_category") 
                                     for pid in final_recommendations[-recent_category_window_size:]]
                recent_cat_counts = Counter(recent_categories)
                # If this category makes up more than (e.g. 60%) of the recent window, consider skipping
                if recent_cat_counts[category_id] >= recent_category_window_size * (1 - self.category_diversity_factor): # e.g. 0.3 -> 70%
                    is_dominant_in_recent = True
            
            if category_counts[category_id] < max_per_category and not is_dominant_in_recent:
                final_recommendations.append(post_id)
                category_counts[category_id] += 1
            else:
                # If skipped due to overall balance or recent dominance, save for later
                temp_skipped_posts.append((post_id, score, category_id))
        
        # Fill remaining slots with skipped posts if any, less strictly
        # This ensures we try to reach num_final_recommendations
        idx = 0
        while len(final_recommendations) < num_final_recommendations and idx < len(temp_skipped_posts):
            post_id, _, _ = temp_skipped_posts[idx]
            if post_id not in final_recommendations: # Avoid duplicates if logic allows
                 final_recommendations.append(post_id)
            idx += 1
            
        return final_recommendations[:num_final_recommendations]


    def re_rank_and_present(self, scored_posts: list, num_final_recommendations: int = 20) -> list:
        """
        Input scored_posts: [("post_id", score), ...].
        Return the final ordered list of post_ids, truncated to num_final_recommendations.
        """
        if not scored_posts:
            return []

        post_ids = [pid for pid, _ in scored_posts]
        
        # 1. Get post metadata
        post_metadata = self._get_post_metadata(post_ids)
        if not post_metadata: # If metadata fetch fails for all, proceed without it or handle error
            print("Warning: Could not fetch any post metadata. Proceeding without boosts or diversity based on metadata.")
            # Fallback: simple sort by score
            sorted_by_score = sorted(scored_posts, key=lambda x: x[1], reverse=True)
            return [pid for pid, _ in sorted_by_score][:num_final_recommendations]

        # 2. Apply new post boost
        boosted_posts = self.apply_new_post_boost(scored_posts, post_metadata)
        
        # 3. Apply category diversity
        # apply_category_diversity expects scored_posts and post_metadata
        final_ordered_post_ids = self.apply_category_diversity(
            boosted_posts, 
            post_metadata, 
            num_final_recommendations
        )
        
        return final_ordered_post_ids

if __name__ == '__main__':
    # This is a placeholder for where run_local_test.py would be.
    # The actual test script will be in a separate file.
    print("ReRankService class defined. Run run_local_test.py for an example.")