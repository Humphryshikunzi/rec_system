import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Concatenate, Input
from tensorflow.keras.models import Model

class UserTower(Model):
    def __init__(self, user_id_vocab_size, embedding_dim, name="user_tower", **kwargs):
        super().__init__(name=name, **kwargs)
        self.user_id_embedding = Embedding(input_dim=user_id_vocab_size,
                                           output_dim=embedding_dim,
                                           name="user_id_embedding")
        # Assuming onboarding_category_ids_embedding is an embedding layer for category IDs
        # If pre-computed, it will be passed directly to call method
        # For simplicity, let's assume it's handled like user_id or passed as pre-embedded
        # self.category_embedding = Embedding(...) 
        self.dense_1 = Dense(128, activation="relu")
        self.dense_2 = Dense(embedding_dim, activation="relu") # Output embedding

    def call(self, inputs):
        user_id = inputs["user_id"]
        about_embedding = inputs["about_embedding"] # Expected shape: (batch_size, embedding_size)
        headline_embedding = inputs["headline_embedding"] # Expected shape: (batch_size, embedding_size)
        # onboarding_category_ids_embedding = inputs["onboarding_category_ids_embedding"] # (batch_size, num_categories, cat_embedding_dim) or (batch_size, cat_embedding_dim) if aggregated

        user_id_embedded = self.user_id_embedding(user_id) # (batch_size, embedding_dim)

        # Concatenate all features
        # Ensure all embeddings are 2D (batch_size, feature_dim) before concatenation
        # If onboarding_category_ids_embedding is (batch_size, num_categories, cat_embedding_dim),
        # it needs to be flattened or aggregated first, e.g., tf.reduce_mean(onboarding_category_ids_embedding, axis=1)
        
        # For now, assuming pre-computed embeddings are already in the correct shape (batch_size, feature_dim)
        # and onboarding_category_ids_embedding is also handled similarly or passed as a single aggregated embedding.
        # If onboarding_category_ids_embedding is a sequence of embeddings, it needs pooling.
        # For simplicity, let's assume it's a single embedding vector for now.
        # Example: onboarding_category_ids_embedding = inputs.get("onboarding_category_ids_embedding", tf.zeros_like(about_embedding)) # Placeholder if not provided

        concatenated_features = Concatenate()([
            user_id_embedded,
            about_embedding,
            headline_embedding,
            # onboarding_category_ids_embedding # Add this once its shape is clear
        ])
        
        x = self.dense_1(concatenated_features)
        return self.dense_2(x)

class PostTower(Model):
    def __init__(self, post_id_vocab_size, category_id_vocab_size, media_type_vocab_size, creator_id_vocab_size, embedding_dim, name="post_tower", **kwargs):
        super().__init__(name=name, **kwargs)
        self.post_id_embedding = Embedding(input_dim=post_id_vocab_size, output_dim=embedding_dim, name="post_id_embedding")
        self.category_id_embedding = Embedding(input_dim=category_id_vocab_size, output_dim=embedding_dim, name="category_id_embedding")
        self.media_type_embedding = Embedding(input_dim=media_type_vocab_size, output_dim=embedding_dim, name="media_type_embedding")
        self.creator_id_embedding = Embedding(input_dim=creator_id_vocab_size, output_dim=embedding_dim, name="creator_id_embedding")
        
        self.dense_1 = Dense(128, activation="relu")
        self.dense_2 = Dense(embedding_dim, activation="relu") # Output embedding

    def call(self, inputs):
        post_id = inputs["post_id"]
        description_embedding = inputs["description_embedding"] # (batch_size, embedding_size)
        category_id = inputs["category_id"] # (batch_size, 1) or (batch_size, num_categories) if multi-hot
        media_type = inputs["media_type"]   # (batch_size, 1)
        creator_id = inputs["creator_id"]   # (batch_size, 1)

        post_id_embedded = self.post_id_embedding(post_id)
        category_id_embedded = self.category_id_embedding(category_id) # (batch_size, embedding_dim)
        media_type_embedded = self.media_type_embedding(media_type)   # (batch_size, embedding_dim)
        creator_id_embedded = self.creator_id_embedding(creator_id)   # (batch_size, embedding_dim)

        # If category_id, media_type, creator_id are single IDs, their embeddings will be (batch_size, embedding_dim)
        # If they are multi-hot or sequences, they might need pooling/aggregation.
        # Assuming single IDs for now.

        concatenated_features = Concatenate()([
            post_id_embedded,
            description_embedding,
            category_id_embedded,
            media_type_embedded,
            creator_id_embedded
        ])
        
        x = self.dense_1(concatenated_features)
        return self.dense_2(x)

class TwoTowerModel(Model):
    def __init__(self, user_tower, post_tower, name="two_tower_model", **kwargs):
        super().__init__(name=name, **kwargs)
        self.user_tower = user_tower
        self.post_tower = post_tower

    def call(self, inputs):
        """
        Inputs should be a dictionary with keys for user tower and post tower.
        Example: {'user_inputs': user_data_dict, 'post_inputs': post_data_dict}
        """
        user_embedding = self.user_tower(inputs['user_inputs'])
        post_embedding = self.post_tower(inputs['post_inputs'])
        
        # Cosine similarity
        # user_embedding_norm = tf.linalg.l2_normalize(user_embedding, axis=1)
        # post_embedding_norm = tf.linalg.l2_normalize(post_embedding, axis=1)
        # score = tf.reduce_sum(user_embedding_norm * post_embedding_norm, axis=1, keepdims=True)
        
        # Dot product for retrieval tasks is common
        score = tf.matmul(user_embedding, post_embedding, transpose_b=True) # For batch-wise dot products if needed for loss
        # If used for single pair scoring:
        # score = tf.reduce_sum(user_embedding * post_embedding, axis=1, keepdims=True)
        return score # Shape: (batch_size, 1) for single pair or (batch_size, batch_size) for in-batch negatives

if __name__ == '__main__':
    # Example Usage (Illustrative - vocab sizes and embedding dim are placeholders)
    USER_VOCAB_SIZE = 10000
    POST_VOCAB_SIZE = 50000
    CATEGORY_VOCAB_SIZE = 100
    MEDIA_TYPE_VOCAB_SIZE = 10
    CREATOR_VOCAB_SIZE = 5000
    EMBEDDING_DIM = 64

    # User Tower
    user_tower = UserTower(USER_VOCAB_SIZE, EMBEDDING_DIM)
    
    # Dummy user inputs
    dummy_user_ids = tf.constant([[1], [2]])
    dummy_about_embeddings = tf.random.normal((2, 128)) # Assuming precomputed embedding size 128
    dummy_headline_embeddings = tf.random.normal((2, 128))
    # dummy_onboarding_categories = tf.random.normal((2, EMBEDDING_DIM)) # Assuming aggregated

    user_inputs_example = {
        "user_id": dummy_user_ids,
        "about_embedding": dummy_about_embeddings,
        "headline_embedding": dummy_headline_embeddings,
        # "onboarding_category_ids_embedding": dummy_onboarding_categories
    }
    user_embedding_example = user_tower(user_inputs_example)
    print("User Embedding Shape:", user_embedding_example.shape)

    # Post Tower
    post_tower = PostTower(POST_VOCAB_SIZE, CATEGORY_VOCAB_SIZE, MEDIA_TYPE_VOCAB_SIZE, CREATOR_VOCAB_SIZE, EMBEDDING_DIM)

    # Dummy post inputs
    dummy_post_ids = tf.constant([[101], [102]])
    dummy_description_embeddings = tf.random.normal((2, 128)) # Assuming precomputed embedding size 128
    dummy_category_ids = tf.constant([[5], [10]])
    dummy_media_types = tf.constant([[1], [2]])
    dummy_creator_ids = tf.constant([[201], [202]])

    post_inputs_example = {
        "post_id": dummy_post_ids,
        "description_embedding": dummy_description_embeddings,
        "category_id": dummy_category_ids,
        "media_type": dummy_media_types,
        "creator_id": dummy_creator_ids,
    }
    post_embedding_example = post_tower(post_inputs_example)
    print("Post Embedding Shape:", post_embedding_example.shape)

    # Two-Tower Model
    two_tower_retrieval_model = TwoTowerModel(user_tower, post_tower)
    
    # For scoring a batch of user-item pairs
    # Assuming user_inputs_example and post_inputs_example are for corresponding pairs
    # To make them compatible for pair-wise scoring, ensure user_tower and post_tower output (batch_size, embedding_dim)
    # Then the dot product will be (batch_size, 1)
    # For in-batch negatives, the score matrix would be (batch_size_users, batch_size_posts)
    
    # Example for pair-wise scoring (adjusting the call if needed)
    # If user_tower and post_tower are called with batch_size N, and we want N scores for N pairs:
    # We need to ensure the matmul or reduce_sum in TwoTowerModel.call results in (N,1) or (N,)
    # The current matmul is more for (batch_user_embeddings, batch_post_embeddings_transposed) -> (batch_users, batch_posts)
    # For simple dot product of corresponding pairs:
    
    class TwoTowerScoringModel(Model): # A slight modification for direct scoring
        def __init__(self, user_tower, post_tower, name="two_tower_scoring_model", **kwargs):
            super().__init__(name=name, **kwargs)
            self.user_tower = user_tower
            self.post_tower = post_tower

        def call(self, inputs):
            user_embedding = self.user_tower(inputs['user_inputs']) # (batch_size, embedding_dim)
            post_embedding = self.post_tower(inputs['post_inputs']) # (batch_size, embedding_dim)
            score = tf.reduce_sum(user_embedding * post_embedding, axis=1, keepdims=True) # (batch_size, 1)
            return score

    two_tower_scoring_model_example = TwoTowerScoringModel(user_tower, post_tower)
    
    model_inputs_example = {
        'user_inputs': user_inputs_example,
        'post_inputs': post_inputs_example
    }
    
    scores_example = two_tower_scoring_model_example(model_inputs_example)
    print("Scores Shape (Pairwise):", scores_example.shape)
    print("Scores (Pairwise):", scores_example)

    # For use with tfrs.tasks.Retrieval, the TwoTowerModel's call method returning separate embeddings is often preferred
    # Or a model that returns query_embedding, candidate_embedding
    # The tfrs.Model would then compute loss using these.
    # The `TwoTowerModel` as defined initially (returning a score matrix) is suitable for in-batch negative sampling loss.
    # If using tfrs.tasks.Retrieval, it expects (query_embeddings, candidate_embeddings) from the model's call method
    # when computing loss.

    # Example of how it might be structured for TFRS
    # class TFRSTwoTowerModel(tfrs.Model):
    #     def __init__(self, user_tower, post_tower, task):
    #         super().__init__()
    #         self.user_tower = user_tower
    #         self.post_tower = post_tower
    #         self.task = task # e.g., tfrs.tasks.Retrieval
    #
    #     def compute_loss(self, features, training=False):
    #         # Assuming features contains user_inputs and post_inputs (positive pair)
    #         user_embeddings = self.user_tower(features["user_inputs"])
    #         positive_post_embeddings = self.post_tower(features["post_inputs"])
    #
    #         # The task computes the loss and metrics.
    #         return self.task(
    #             query_embeddings=user_embeddings,
    #             candidate_embeddings=positive_post_embeddings,
    #             compute_metrics=not training # Only compute metrics during evaluation
    #         )
    #
    # # If using TFRS, the `call` method might not be directly used for loss computation in the same way.
    # # The `compute_loss` method is key.
    # # The `task` itself might handle candidate sampling (e.g., from all posts in the batch).
    #
    # # For inference/serving, you'd typically use the user_tower and post_tower separately
    # # or a model that just computes scores.
    # user_tower.save("user_tower_model")
    # post_tower.save("post_tower_model")

    print("Model definitions created.")