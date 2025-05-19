import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.tensorflow
import mlflow.pyfunc # For generic model logging

import yaml
import os
import pandas as pd
import numpy as np
import joblib # For saving scikit-learn compatible models

# Model specific imports
import xgboost as xgb
import lightgbm as lgb
# import tensorflow as tf # Uncomment if using Neural Network

from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score, accuracy_score

# Custom data utility
from data_utils import get_training_and_validation_data, load_config

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_random_seeds(seed):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    # tf.random.set_seed(seed) # Uncomment if using TensorFlow
    # random.seed(seed) # For general python random
    logger.info(f"Set random seed to {seed}")

def train_xgboost_model(X_train, y_train, X_val, y_val, config):
    """Trains an XGBoost model."""
    logger.info("Training XGBoost model...")
    model_params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'learning_rate': config.get('learning_rate', 0.01),
        'max_depth': config.get('max_depth', 7),
        'n_estimators': config.get('n_estimators', 100),
        'random_state': config.get('random_seed', 42),
        'tree_method': 'hist', # Faster for large datasets
        'enable_categorical': True # Enable native categorical support if X_train dtypes are set correctly
    }
    
    # XGBoost DMatrix can handle pandas DataFrames with categorical types if enable_categorical=True
    # Ensure X_train and X_val columns are in the same order and have correct dtypes.
    # data_utils.prepare_data_for_model should handle this.
    
    # Convert boolean columns to int for XGBoost if any exist
    for df in [X_train, X_val]:
        if df is not None and not df.empty:
            bool_cols = df.select_dtypes(include='bool').columns
            if not bool_cols.empty:
                logger.info(f"Converting boolean columns to int for XGBoost: {bool_cols.tolist()}")
                for col in bool_cols:
                    df[col] = df[col].astype(int)

    model = xgb.XGBClassifier(**model_params)
    
    if X_val is not None and not X_val.empty and y_val is not None and not y_val.empty:
        eval_set = [(X_train, y_train), (X_val, y_val)]
        eval_names = ['train', 'validation']
    else:
        eval_set = [(X_train, y_train)]
        eval_names = ['train']
        logger.warning("Validation set is empty or not provided. Training without early stopping based on validation.")

    model.fit(X_train, y_train,
              eval_set=eval_set,
              # early_stopping_rounds=config.get('early_stopping_rounds', 10) if X_val is not None else None, # Requires eval_set
              verbose=config.get('xgboost_verbose', 100)) # Log every 100 rounds
              
    logger.info("XGBoost model training complete.")
    return model

def train_lightgbm_model(X_train, y_train, X_val, y_val, config):
    """Trains a LightGBM model."""
    logger.info("Training LightGBM model...")
    model_params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'learning_rate': config.get('learning_rate', 0.01),
        'max_depth': config.get('max_depth', 7), # LightGBM uses num_leaves, max_depth is an approximation
        'num_leaves': config.get('num_leaves', 31), # Default is 31
        'n_estimators': config.get('n_estimators', 100),
        'random_state': config.get('random_seed', 42),
        # LightGBM can handle categorical features directly if specified
        # 'categorical_feature': 'auto' # or pass list of column names/indices
    }

    # Identify categorical features for LightGBM
    # data_utils.prepare_data_for_model should have handled encoding if not using native support
    # If using native support, ensure categorical columns are of 'category' dtype in pandas
    categorical_features_names = X_train.select_dtypes(include='category').columns.tolist()
    if categorical_features_names:
        logger.info(f"Using categorical features for LightGBM: {categorical_features_names}")
        model_params['categorical_feature'] = categorical_features_names
    
    model = lgb.LGBMClassifier(**model_params)

    if X_val is not None and not X_val.empty and y_val is not None and not y_val.empty:
        eval_set = [(X_val, y_val)]
        eval_metric = ['binary_logloss', 'auc']
        callbacks = [lgb.early_stopping(config.get('early_stopping_rounds', 10), verbose=1)]
    else:
        eval_set = None
        eval_metric = None
        callbacks = None
        logger.warning("Validation set is empty or not provided. Training without early stopping.")

    model.fit(X_train, y_train,
              eval_set=eval_set,
              eval_metric=eval_metric,
              callbacks=callbacks)
              
    logger.info("LightGBM model training complete.")
    return model

def train_nn_model(X_train, y_train, X_val, y_val, config, model_definition_path=None):
    """Trains a Neural Network model (TensorFlow/Keras)."""
    logger.info("Training Neural Network model (Placeholder)...")
    # This requires X_train, y_train to be tf.data.Dataset or compatible
    # And a model definition (e.g., from src/model_training/ranking/model.py)
    
    # Example:
    # if model_definition_path and os.path.exists(model_definition_path):
    #     import importlib.util
    #     spec = importlib.util.spec_from_file_location("ranking_model_def", model_definition_path)
    #     model_module = importlib.util.module_from_spec(spec)
    #     spec.loader.exec_module(model_module)
    #     # Assuming a function like create_model(config, num_features, categorical_features_info)
    #     # model = model_module.create_model(config, X_train.element_spec[0]) # This needs careful handling of feature specs
    # else:
    #     raise FileNotFoundError(f"Neural network model definition not found at {model_definition_path}")

    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate')),
    #               loss='binary_crossentropy',
    #               metrics=['AUC', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    # history = model.fit(X_train, # This should be the tf.data.Dataset prepared by data_utils
    #                     epochs=config.get('epochs', 10),
    #                     validation_data=X_val, # This should be the validation tf.data.Dataset
    #                     callbacks=[tf.keras.callbacks.EarlyStopping(patience=config.get('early_stopping_patience', 3))])
    
    # logger.info("Neural Network training complete.")
    # return model, history
    raise NotImplementedError("Neural Network training is not fully implemented yet.")


def evaluate_model(model, X_data, y_data, model_type):
    """Evaluates the model and returns a dictionary of metrics."""
    logger.info(f"Evaluating model on dataset of shape {X_data.shape}")
    if X_data.empty or y_data.empty:
        logger.warning("Evaluation data is empty. Skipping evaluation.")
        return {}

    if model_type == "neural_network":
        # For NN, evaluation might be part of model.evaluate() or need predictions first
        # y_pred_proba = model.predict(X_data)[:, 0] # Assuming output is (batch, 1)
        logger.warning("NN evaluation placeholder.")
        return {"auc": 0.0, "logloss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0} # Placeholder
    else: # XGBoost, LightGBM
        y_pred_proba = model.predict_proba(X_data)[:, 1]

    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(y_data, y_pred_proba),
        "logloss": log_loss(y_data, y_pred_proba),
        "accuracy": accuracy_score(y_data, y_pred_binary),
        "precision": precision_score(y_data, y_pred_binary, zero_division=0),
        "recall": recall_score(y_data, y_pred_binary, zero_division=0)
    }
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics

def save_model_local(model, model_type, save_path="model_output"):
    """Saves the model locally."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    model_filename = ""
    if model_type == "xgboost":
        model_filename = os.path.join(save_path, "model.xgb")
        model.save_model(model_filename)
    elif model_type == "lightgbm":
        model_filename = os.path.join(save_path, "model.txt")
        model.booster_.save_model(model_filename) # Saves the core booster
        # Or use joblib for the scikit-learn wrapper:
        # model_filename = os.path.join(save_path, "model.joblib")
        # joblib.dump(model, model_filename)
    elif model_type == "neural_network":
        model_filename = os.path.join(save_path, "model_nn") # Keras saves as a directory
        model.save(model_filename)
    else:
        logger.error(f"Unsupported model type for saving: {model_type}")
        return None
        
    logger.info(f"Model saved locally to {model_filename}")
    return model_filename


def main(config_path):
    """Main training pipeline."""
    logger.info(f"Starting ranking model training pipeline with config: {config_path}")
    config = load_config(config_path)

    set_random_seeds(config.get('random_seed', 42))

    mlflow.set_experiment(config['mlflow_experiment_name'])

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        mlflow.log_params(config) # Log all config parameters

        # 1. Load and prepare data
        logger.info("Loading and preparing data...")
        X_train, y_train, X_val, y_val, data_config = get_training_and_validation_data(config_path)
        
        if X_train.empty or y_train.empty:
            logger.error("Training data is empty. Aborting training.")
            mlflow.log_metric("training_status", 0) # 0 for failure
            return

        mlflow.log_metric("num_train_samples", len(X_train))
        mlflow.log_metric("num_val_samples", len(X_val) if X_val is not None else 0)
        mlflow.log_param("training_feature_names", X_train.columns.tolist())


        # 2. Train model
        model_type = config.get('model_type', 'xgboost')
        model = None

        if model_type == "xgboost":
            model = train_xgboost_model(X_train, y_train, X_val, y_val, config)
        elif model_type == "lightgbm":
            model = train_lightgbm_model(X_train, y_train, X_val, y_val, config)
        elif model_type == "neural_network":
            # model_def_path = os.path.join(os.path.dirname(__file__), "model.py") # If NN model is in model.py
            # model, history = train_nn_model(X_train, y_train, X_val, y_val, config, model_def_path)
            # for key, value in history.history.items(): # Log NN training history
            #     mlflow.log_metric(f"nn_train_{key}", value[-1]) # Log last epoch metric
            logger.warning("Neural Network training is a placeholder.")
            # For now, let's skip actual NN training to avoid unimplemented errors
            mlflow.log_metric("training_status", 0) # Mark as failed if NN selected for now
            logger.error("Neural Network model type selected but not fully implemented.")
            return # Exit if NN is chosen and not implemented
        else:
            logger.error(f"Unsupported model type: {model_type}")
            mlflow.log_metric("training_status", 0)
            return

        if model is None:
            logger.error("Model training failed.")
            mlflow.log_metric("training_status", 0)
            return

        # 3. Evaluate model
        logger.info("Evaluating model on training set...")
        train_metrics = evaluate_model(model, X_train, y_train, model_type)
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)

        if X_val is not None and not X_val.empty and y_val is not None and not y_val.empty:
            logger.info("Evaluating model on validation set...")
            val_metrics = evaluate_model(model, X_val, y_val, model_type)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
        else:
            logger.info("No validation set for evaluation.")

        # 4. Save and Log model
        logger.info("Saving and logging model to MLflow...")
        model_save_dir = f"models/ranking_model/{run_id}" # Save locally with run_id
        local_model_path = save_model_local(model, model_type, save_path=model_save_dir)

        if local_model_path:
            if model_type == "xgboost":
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="model", # This is a sub-directory within the MLflow run's artifact store
                    # registered_model_name=f"{config['mlflow_experiment_name']}-XGB" # Optional: register model
                )
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(
                    lgb_model=model,
                    artifact_path="model",
                    # registered_model_name=f"{config['mlflow_experiment_name']}-LGBM"
                )
            elif model_type == "neural_network":
                # mlflow.tensorflow.log_model(
                #     model=model,
                #     artifact_path="model",
                #     # registered_model_name=f"{config['mlflow_experiment_name']}-NN"
                # )
                pass # Placeholder for NN
            
            # Example of registering the model (can be done after logging)
            # client = mlflow.tracking.MlflowClient()
            # model_uri = f"runs:/{run_id}/model"
            # try:
            #     client.create_registered_model(f"{config['mlflow_experiment_name']}-{model_type.upper()}")
            # except mlflow.exceptions.MlflowException as e:
            #     if "already exists" in str(e).lower(): # Handle if model name already registered
            #         logger.info(f"Registered model '{config['mlflow_experiment_name']}-{model_type.upper()}' already exists.")
            #     else:
            #         raise e
            #
            # client.create_model_version(
            #     name=f"{config['mlflow_experiment_name']}-{model_type.upper()}",
            #     source=model_uri,
            #     run_id=run_id
            # )
            # logger.info(f"Model registered in MLflow Model Registry as {config['mlflow_experiment_name']}-{model_type.upper()}")

        else:
            logger.error("Failed to save model locally, cannot log to MLflow.")

        mlflow.log_metric("training_status", 1) # 1 for success
        logger.info("Training pipeline finished successfully.")


if __name__ == "__main__":
    # Assuming this script is in src/model_training/ranking/
    # Project root is ../../.. from here
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    default_config_path = os.path.join(project_root, "config/ranking_model_config.yaml")

    # Allow overriding config path via command line argument for flexibility
    import argparse
    parser = argparse.ArgumentParser(description="Train a ranking model.")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path,
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
    else:
        main(args.config)