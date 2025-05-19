import mlflow
import numpy as np
import tensorflow as tf # Assuming User Tower model is TensorFlow based

class UserTowerPredictor:
    """
    Predictor for generating user embeddings using the User Tower model.
    Simulates the predictor component in a KServe inference graph.
    """
    def __init__(self, model_uri: str):
        """
        Initializes the UserTowerPredictor.

        Args:
            model_uri (str): The URI of the User Tower model in MLflow
                             (e.g., "models:/TwoTowerUserTower/Production").
        """
        self.model = mlflow.pyfunc.load_model(model_uri)
        # If the MLflow model is a Keras model and you need direct access:
        # self.keras_model = mlflow.tensorflow.load_model(model_uri)
        # Or if it's a signature-based model, you might need to inspect its signature.

    def predict(self, user_features: dict) -> np.ndarray:
        """
        Generates a user embedding from user features.

        Args:
            user_features (dict): A dictionary of user features, prepared to match
                                  the User Tower model's input signature.
                                  Example:
                                  {
                                      "user_id": np.array(["user_123"]),
                                      "feature_name_1": np.array([[0.1, 0.2, ...]]),
                                      "feature_name_2": np.array([[0.3, 0.4, ...]])
                                  }
                                  The exact structure depends on how the User Tower
                                  model was saved and its input signature.

        Returns:
            np.ndarray: The user embedding as a NumPy array.
        """
        # The MLflow pyfunc model's predict method typically expects a pandas DataFrame
        # or a dictionary of numpy arrays.
        # We need to ensure the input format matches what the loaded model expects.
        # This might involve converting lists to numpy arrays, ensuring correct shapes, etc.

        # Example: If the model expects a dictionary of tf.Tensor
        # This is a placeholder. The actual preprocessing depends heavily on the
        # specific User Tower model's input requirements.
        # You might need to convert features to tensors, reshape them, etc.
        
        # For a typical Keras model loaded via mlflow.pyfunc,
        # it often expects a dictionary of numpy arrays or a pandas DataFrame.
        # Let's assume the user_features dict is already in the correct format
        # (e.g., values are numpy arrays of the correct shape and type).

        # If your model has a specific signature, you might need to adapt.
        # For instance, if it's a TensorFlow model with a serving signature:
        # predictions = self.model.predict(user_features) # This is for pyfunc
        # If using self.keras_model directly:
        # predictions = self.keras_model.predict(user_features)

        # Assuming the pyfunc model's predict method handles the conversion
        # or the input `user_features` is already correctly formatted.
        try:
            # MLflow pyfunc models usually expect a DataFrame or Dict[str, np.ndarray]
            # The output format also depends on how the model was saved.
            # It might be a dict, a list, or a numpy array.
            # We assume it returns the embedding directly or within a structure.
            raw_prediction = self.model.predict(user_features)

            # Post-process the prediction to get the embedding as a NumPy array.
            # This depends on the output structure of your User Tower model.
            # If it's a dictionary with a key like 'user_embedding':
            if isinstance(raw_prediction, dict) and 'user_embedding' in raw_prediction:
                user_embedding = raw_prediction['user_embedding']
            # If it's a list of embeddings (e.g., for batch prediction, though here it's single):
            elif isinstance(raw_prediction, list):
                user_embedding = raw_prediction[0] # Assuming first element is the embedding
            # If it's directly a NumPy array:
            elif isinstance(raw_prediction, np.ndarray):
                user_embedding = raw_prediction
            else:
                raise ValueError(f"Unexpected prediction output format: {type(raw_prediction)}")

            # Ensure it's a 2D array (e.g., (1, embedding_dim)) if not already
            if user_embedding.ndim == 1:
                user_embedding = np.expand_dims(user_embedding, axis=0)
            
            return user_embedding.astype(np.float32)

        except Exception as e:
            print(f"Error during UserTowerPredictor predict: {e}")
            # Potentially re-raise or return a specific error indicator
            raise

if __name__ == '__main__':
    # This is a placeholder for a very basic test.
    # In a real scenario, you'd need a running MLflow server or a local model path.
    print("UserTowerPredictor basic check (requires model to be available)")
    # Example:
    # model_uri_example = "models:/TwoTowerUserTower/Production" # or "runs:/<run_id>/user_tower_model_path"
    # try:
    #     # predictor = UserTowerPredictor(model_uri=model_uri_example)
    #     # dummy_features = {
    #     #     "user_id": np.array(["test_user"]),
    #     #     # Add other features your user tower expects, matching the training format
    #     #     # "about_embedding": np.random.rand(1, 128).astype(np.float32), # Example
    #     # }
    #     # embedding = predictor.predict(dummy_features)
    #     # print(f"Generated embedding shape: {embedding.shape}")
    #     print("To run a proper test, ensure MLflow is set up and the model URI is correct.")
    # except Exception as e:
    #     print(f"Could not run example: {e}")
    #     print("This might be due to MLflow setup or model URI not being accessible.")
    pass