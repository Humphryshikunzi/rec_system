# Personalized Social Recommender System - Synthetic Data Generation

## Project Overview

This project focuses on building a personalized social recommender system. This initial phase involves setting up the project structure and generating synthetic data that mimics user behavior on a social media platform. The generated data includes users, posts, and interactions, which will serve as the foundation for subsequent feature engineering and model training.

## Assumptions

*   **Category Definition:** A predefined list of 10-20 categories is assumed for user onboarding preferences and post categorization. The exact nature of these categories is abstracted to `cat_ID`.
*   **Interaction Logic:** Interactions are generated with timestamps that are logically consistent (i.e., an interaction occurs after the involved user and post have been created).
*   **Text Content:** User `about_text`, `headline_text`, and post `description_text` are generated as random placeholder text using the Faker library. The semantic content is not critical for this phase but the structure is.
*   **Media Types:** A simple list of media types (`text`, `image`, `video`) is used for posts.
*   **Data Splitting:** Interactions are split chronologically into train, validation, and test sets. Users and posts are then filtered/assigned to these splits based on the interactions they are involved in. This means a user or post might appear in multiple splits if their interactions span across the time boundaries of these splits.

## Environment Setup

1.  **Clone the repository (if applicable).**
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Synthetic Data Generation

The synthetic data generation script uses a configuration file to control various parameters.

**Configuration:**

The main configuration file is located at `config/data_config.yaml`. You can modify parameters such as the number of users, posts, categories, interaction density, date ranges, and output directories.

```yaml
# Example: config/data_config.yaml
num_users: 1000
num_posts: 10000
num_categories: 15
avg_interactions_per_user: 50
start_date: "2024-01-01"
end_date: "2024-12-31"
random_seed: 42
output_dir: "artifacts/data"
split_ratios: # train, validation, test
  train: 0.7
  validation: 0.15
  test: 0.15
```

**Running the Script:**

To generate the synthetic data, run the following command from the root of the project:

```bash
python src/data_generation/generate_synthetic_data.py --config config/data_config.yaml
```

You can also run it without specifying the config path if `config/data_config.yaml` exists in the default location:
```bash
python src/data_generation/generate_synthetic_data.py
```

## Feature Transformation Scripts

After generating the synthetic raw data, feature transformation scripts are used to compute advanced features like text embeddings and aggregations. These scripts are located in `src/feature_processing/`.

*   **`src/feature_processing/generate_embeddings.py`**:
    *   Loads `users.csv` and `posts.csv`.
    *   Generates text embeddings for `about_text`, `headline_text` (users) and `description_text` (posts) using `sentence-transformers`.
    *   Outputs `users_with_embeddings.parquet` and `posts_with_embeddings.parquet` to `artifacts/data/train/`.
*   **`src/feature_processing/generate_aggregated_features.py`**:
    *   Loads `users.csv`, `posts.csv`, and `interactions.csv`.
    *   Computes user-level aggregations (e.g., `num_posts_created`, `count_likes_given`) and post-level aggregations/derived features (e.g., `post_age_hours`, `num_likes_on_post`).
    *   Outputs `user_aggregated_features.parquet` and `post_aggregated_derived_features.parquet` to `artifacts/data/train/`.

## Feature Engineering with Feast

This section describes how to set up and use Feast for feature engineering in this project. The Feast repository is located in `src/feature_repo/`.

**Order of Operations (Full Pipeline):**

1.  **Generate Synthetic Data (if not already done or needs refresh):**
    ```bash
    python src/data_generation/generate_synthetic_data.py
    ```
2.  **Run Feature Transformation Scripts:**
    ```bash
    python src/feature_processing/generate_embeddings.py
    python src/feature_processing/generate_aggregated_features.py
    ```
3.  **Apply Feast Definitions:** Register feature definitions with Feast.
    ```bash
    ./scripts/feast_apply.sh
    ```
4.  **Materialize Features:** Load transformed features into the Feast offline store.
    ```bash
    ./scripts/materialize_features.sh
    # Or for a specific range, e.g., initial load:
    # ./scripts/materialize_features.sh "2023-01-01T00:00:00" "$(date -u +'%Y-%m-%dT%H:%M:%S')"
    ```

**Initialization and Applying Definitions:**

1.  **Ensure Feast is installed:** If you haven't already, make sure `feast` is listed in your `requirements.txt` and installed in your virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
2.  **Apply Feast Definitions:** To register your feature definitions (entities, data sources, feature views) with Feast, run the following script from the project root:
    ```bash
    ./scripts/feast_apply.sh
    ```
    This script navigates to the `src/feature_repo/` directory, activates the virtual environment (if not already active), and runs `feast apply`.

**Materializing Features:**

Features need to be materialized from your offline data sources (now Parquet files generated by transformation scripts) into the offline store.

1.  **Run the Materialization Script:** To materialize features, use the following script from the project root:
    ```bash
    ./scripts/materialize_features.sh
    ```
    By default, this script attempts to materialize features from "1970-01-01T00:00:00" to the current time, effectively performing an initial full load.
2.  **Custom Date Range (Optional):** You can provide start and end dates as arguments to the script if needed:
    ```bash
    ./scripts/materialize_features.sh "YYYY-MM-DDTHH:MM:SS" "YYYY-MM-DDTHH:MM:SS"
    ```

**Key Transformed Features Available:**

*   **User Embeddings:** `about_embedding`, `headline_embedding`
*   **Post Embeddings:** `description_embedding`
*   **User Aggregations:** `num_posts_created`, `count_likes_given`, `count_comments_given`, `distinct_categories_interacted`, `last_interaction_timestamp`, `num_likes_received_on_posts`
*   **Post Aggregations & Derived:** `post_age_hours`, `num_likes_on_post`, `num_views_on_post`, `num_comments_on_post`

**Defined Feature Views (Updated):**

The Feast repository defines the following key feature views using the transformed data:

*   **`user_profile_features_view`**:
    *   **Source:** `users_with_embeddings.parquet`
    *   **Entity:** `user`
    *   **Purpose:** Provides user profile features including text embeddings.
    *   **Key Features:** `onboarding_category_ids`, `about_embedding`, `headline_embedding`.

*   **`post_details_features_view`**:
    *   **Source:** `posts_with_embeddings.parquet`
    *   **Entity:** `post`
    *   **Purpose:** Provides post detail features including text embeddings.
    *   **Key Features:** `category_id`, `media_type`, `creator_id`, `creation_timestamp`, `description_embedding`.

*   **`user_aggregated_stats_view`**:
    *   **Source:** `user_aggregated_features.parquet`
    *   **Entity:** `user`
    *   **Purpose:** Provides aggregated statistical features for users.
    *   **Key Features:** `num_posts_created`, `count_likes_given`, `count_comments_given`, `distinct_categories_interacted`, `last_interaction_timestamp`, `num_likes_received_on_posts`.

*   **`post_aggregated_derived_stats_view`**:
    *   **Source:** `post_aggregated_derived_features.parquet`
    *   **Entity:** `post`
    *   **Purpose:** Provides aggregated and derived statistical features for posts.
    *   **Key Features:** `post_age_hours`, `num_likes_on_post`, `num_views_on_post`, `num_comments_on_post`, `creation_timestamp`.

## Expected Output (Data Artifacts)

After running the data generation and feature transformation scripts, the `artifacts/data/train/` directory will contain:

*   `users.csv`, `posts.csv`, `interactions.csv` (original synthetic data)
*   `users_with_embeddings.parquet` (users with text embeddings)
*   `posts_with_embeddings.parquet` (posts with text embeddings)
*   `user_aggregated_features.parquet` (aggregated features for users)
*   `post_aggregated_derived_features.parquet` (aggregated and derived features for posts)

**Directory Structure (Post-Transformations):**

```
.
├── artifacts/
│   └── data/
│       ├── test/
│       │   ├── interactions.csv
│       │   ├── posts.csv
│       │   └── users.csv
│       ├── train/
│       │   ├── interactions.csv
│       │   ├── posts.csv
│       │   ├── users.csv
│       │   ├── users_with_embeddings.parquet
│       │   ├── posts_with_embeddings.parquet
│       │   ├── user_aggregated_features.parquet
│       │   └── post_aggregated_derived_features.parquet
│       └── validation/
│           ├── interactions.csv
│           ├── posts.csv
│           └── users.csv
├── config/
│   └── data_config.yaml
├── notebooks/
├── requirements.txt
├── README.md
├── scripts/
│   ├── feast_apply.sh
│   └── materialize_features.sh
└── src/
    ├── data_generation/
    │   └── generate_synthetic_data.py
    ├── feature_processing/
    │   ├── generate_aggregated_features.py
    │   └── generate_embeddings.py
    └── feature_repo/
        ├── __init__.py
        ├── data_sources.py
        ├── definitions.py
        ├── entities.py
        ├── feature_store.yaml
        └── feature_views.py

## Two-Tower Model Training

This section outlines the process for training the Two-Tower model for candidate generation. The model consists of a User Tower and a Post Tower, which learn to generate compatible embeddings for users and posts.

**Model Components:**

*   **User Tower:** Takes user features (e.g., `user_id`, `about_embedding`, `headline_embedding`, `onboarding_category_ids_embedding`) and produces a user embedding.
*   **Post Tower:** Takes post features (e.g., `post_id`, `description_embedding`, `category_id_embedding`, `media_type_embedding`, `creator_id_embedding`) and produces a post embedding.
*   **Training:** The model is trained using positive (user, post) pairs from strong interactions (likes, bookmarks) and negative pairs. It uses TensorFlow Recommenders (`tfrs`) for the retrieval task, which typically employs an in-batch softmax loss.

**Configuration:**

Model training parameters are configured in `config/two_tower_config.yaml`. This includes:
*   `embedding_dim`: The dimensionality of the output embeddings for user and post towers.
*   `learning_rate`: Optimizer learning rate.
*   `batch_size`: Training batch size.
*   `epochs`: Number of training epochs.
*   `feast_repo_path`: Path to the Feast feature repository.
*   `mlflow_experiment_name`: Name of the MLflow experiment for tracking.
*   `user_tower_features`: List of Feast feature references for the User Tower.
*   `post_tower_features`: List of Feast feature references for the Post Tower.

**Running the Training Script:**

To train the Two-Tower model, execute the following command from the project root:

```bash
python src/model_training/two_tower/train.py --config config/two_tower_config.yaml
```

**MLflow Integration:**

The training script is integrated with MLflow for:
*   **Experiment Tracking:** Logging parameters, metrics (loss, retrieval accuracy), and artifacts.
*   **Model Registry:** Saving the trained User Tower and Post Tower separately and registering them in the MLflow Model Registry. This allows for versioning and easier deployment of the model components.

The trained model components (User Tower and Post Tower) can then be used for candidate generation in a recommendation pipeline. The User Tower generates a user embedding, and the Post Tower (or its embeddings stored in an index like FAISS/ScaNN) is used to find the top-K most relevant post embeddings.
## Ranking Model Training

This section describes the process for training the Ranking Model, which predicts the probability of a positive user-post interaction (e.g., 'like', 'bookmark'). This model is typically used as a second stage in a recommendation system, after a candidate generation step (like the Two-Tower model), to re-rank a smaller set of candidate items.

**Model Overview:**

*   **Objective:** Predict the probability of positive user-post interaction.
*   **Model Options:** The training script supports XGBoost, LightGBM. A Neural Network architecture is planned but currently a placeholder.
*   **Input Features:** The model uses a rich set of features fetched from Feast, including:
    *   User features: `about_embedding`, `headline_embedding`, `onboarding_category_ids`, aggregated interaction counts.
    *   Post features: `description_embedding`, `category_id`, `media_type`, `creator_id`, `post_age_hours`, engagement features.
    *   User-Post Interaction features: Some are pre-computed and fetched from Feast, while others like `cosine_similarity_user_post_embedding` and `is_post_category_in_onboarding` are computed on-the-fly during data preparation.
*   **Labels:** Positive (1) for strong positive interactions (e.g., 'like', 'bookmark'), Negative (0) for impressions without positive interaction (negative sampling).
*   **Frameworks:** Scikit-learn compatible (XGBoost/LightGBM). TensorFlow/Keras for the planned Neural Network.
*   **Tools:** Feast (feature retrieval), MLflow (model logging/registry).

**Configuration:**

Model training parameters, feature lists, and paths are configured in `config/ranking_model_config.yaml`. Key parameters include:
*   `model_type`: Specifies the model to train (e.g., "xgboost", "lightgbm").
*   `learning_rate`, `max_depth`, `n_estimators`: Model-specific hyperparameters.
*   `feast_repo_path`: Path to the Feast feature repository.
*   `mlflow_experiment_name`: Name for the MLflow experiment.
*   `feature_list`: List of Feast feature references to be used for training.
*   `positive_interaction_types`: Defines which interactions are considered positive.
*   `negative_sampling_ratio`: Controls how many negative samples are generated per positive sample.

**Running the Training Script:**

To train the Ranking Model, execute the following command from the project root:

```bash
python src/model_training/ranking/train.py --config config/ranking_model_config.yaml
```
Or, if your configuration file is at the default location (`config/ranking_model_config.yaml`):
```bash
python src/model_training/ranking/train.py
```

**MLflow Integration:**

The training script (`src/model_training/ranking/train.py`) is integrated with MLflow for:
*   **Experiment Tracking:** Logging configuration parameters, training/validation metrics (e.g., AUC, LogLoss, Precision, Recall), and other artifacts.
*   **Model Logging:** Saving the trained model (e.g., XGBoost, LightGBM) and logging it to MLflow.
*   **Model Registry (Manual Step Recommended):** While the script logs the model, explicitly registering the model version in the MLflow Model Registry is a recommended subsequent step for production workflows.

The trained ranking model can then be used to score and re-rank candidates provided by a candidate generation model, leading to a more refined list of recommendations for the user.
## Offline Inference: Post Indexing to Milvus

This section describes how to generate embeddings for all posts using the trained Post Tower model and index them into a local Milvus instance for efficient similarity search.

**1. Setup Local Milvus Instance:**

You need a running Milvus instance. The easiest way to get one up for local development is using Docker Compose.
Refer to the official Milvus documentation for the latest instructions. A common way is:

```bash
# Create a docker-compose.yml file (e.g., in a temporary directory or project root)
# with content from: https://milvus.io/docs/install_standalone-docker.md
# Example content (check official docs for updates for v2.3.x or later):
# version: '3.5'
# services:
#   milvus-standalone:
#     container_name: milvus-standalone
#     image: milvusdb/milvus:v2.3.10-standalone # Use a specific v2.3.x version
#     ports:
#       - "19530:19530" # Milvus SDK port
#       - "9091:9091"   # Milvus gRPC port (used by Attu/API)
#     volumes:
#       - ./milvus_data:/var/lib/milvus # Persist Milvus data (optional for PoC)
#     environment:
#       ETCD_ENDPOINTS: etcd:2379 # Not needed for standalone
#       MINIO_ADDRESS: minio:9000 # Not needed for standalone

# Then run:
docker compose up -d # Or: docker run -d --name milvus_standalone ... (see official docs)
```

Ensure Milvus is running and accessible on `localhost:19530` (or as configured).

**2. Run the Indexing Script:**

The script `src/inference/offline/index_posts_to_milvus.py` loads the trained Post Tower model from MLflow, generates embeddings for posts specified in the configuration, and ingests them into Milvus.

*   **Configuration:** The script uses `config/milvus_indexing_config.yaml` to get parameters like the MLflow model URI, Milvus connection details, collection name, and post data path.
    ```yaml
    # Example: config/milvus_indexing_config.yaml
    mlflow_post_tower_uri: "models:/TwoTowerPostTower/Production"
    milvus_host: "localhost"
    milvus_port: "19530"
    milvus_collection_name: "recsys_poc_posts"
    embedding_dim: 64 # Should match Post Tower output
    batch_size: 256
    post_data_path: "artifacts/data/train/posts_with_embeddings.parquet"
    ```

*   **Execution:**
    Run the script from the project root:
    ```bash
    python src/inference/offline/index_posts_to_milvus.py --config config/milvus_indexing_config.yaml
    ```

**3. Milvus Collection Schema:**

The script will create a Milvus collection (if it doesn't exist) with the following schema:

*   `post_id`: `INT64` (Primary Key) - The unique identifier for the post.
*   `category_id`: `INT64` - The category ID of the post (can be used for filtered search).
*   `embedding`: `FLOAT_VECTOR` (Dimension as per `embedding_dim` in config, e.g., 64) - The embedding generated by the Post Tower model.

An index (e.g., HNSW) will be created on the `embedding` field to enable fast similarity searches.
## Online Inference - Stage 1: Candidate Generation

This stage focuses on retrieving a list of candidate `post_id`s given a `user_id` and preferred `category_id`s. It simulates a KServe Predictor and Transformer pattern for online inference.

**Components:**

*   **`UserTowerPredictor` (`src/inference/online/candidate_generation/predictor.py`):**
    *   Loads a pre-trained User Tower model from MLflow.
    *   The `predict` method takes user features (prepared by the Transformer) and generates a user embedding.
*   **`CandidateGenerationTransformer` (`src/inference/online/candidate_generation/transformer.py`):**
    *   Initializes connections to Feast (for user feature retrieval) and Milvus (for Approximate Nearest Neighbor search).
    *   Uses an instance of `UserTowerPredictor`.
    *   `preprocess(request_data)`: Fetches online user features from Feast based on `user_id` from the input request.
    *   `postprocess(user_embedding, request_data)`: Queries Milvus using the generated `user_embedding` to find relevant `post_id`s. It filters candidates by `preferred_category_ids` and limits results to `top_n_candidates` specified in the request.
    *   `generate_candidates(request_data)`: Orchestrates the flow: `preprocess` -> `UserTowerPredictor.predict` -> `postprocess`.

**Configuration:**

The service is configured via `config/candidate_generation_service_config.yaml`. This file specifies:
*   Paths and URIs for Feast repository, MLflow model, and Milvus connection details.
*   The list of user features (`user_features_for_tower`) required by the User Tower model.
*   Default number of candidates (`default_top_n_candidates`).
*   Milvus specific parameters like embedding field name, output fields, and ID field name.

**Running the Local Test Script:**

A local test script `src/inference/online/candidate_generation/run_local_test.py` is provided to demonstrate the functionality of the candidate generation service.

1.  **Ensure Prerequisites:**
    *   A running MLflow server with the specified User Tower model registered (e.g., `models:/TwoTowerUserTower/Production`).
*   To start a local MLflow server, navigate to your project root directory in the terminal and run:
            ```bash
            mlflow ui --backend-store-uri ./mlruns --default-artifact-root ./mlruns -p 5000
            ```
            This will typically make the MLflow UI accessible at `http://localhost:5000`. Ensure your `MLFLOW_TRACKING_URI` environment variable is set (e.g., to `http://localhost:5000`) if you are running MLflow on a different host or port, or if your scripts need to explicitly know where the server is.
    *   A running Feast instance with materialized user features.
    *   A running Milvus instance with an indexed collection of post embeddings (e.g., `recsys_poc_posts`, populated by the offline indexing script).
    *   The configuration file `config/candidate_generation_service_config.yaml` is correctly set up.

2.  **Execute the script:**
    From the project root directory:
    ```bash
    python src/inference/online/candidate_generation/run_local_test.py --config config/candidate_generation_service_config.yaml --user_id "u_0000" --categories 0 5 --top_n 10
    ```
    *   `--config`: Path to the service configuration YAML file. Defaults to `config/candidate_generation_service_config.yaml` relative to the project root.
    *   `--user_id`: The ID of the user for whom to generate candidates.
    *   `--categories`: A list of preferred category IDs (space-separated).
    *   `--top_n`: The number of candidates to retrieve.

    The script will:
    *   Load the configuration.
    *   Instantiate `UserTowerPredictor` and `CandidateGenerationTransformer`.
    *   Simulate a request with the provided (or default) `user_id`, `preferred_category_ids`, and `top_n_candidates`.
    *   Call `transformer.generate_candidates()` and print the resulting `post_id`s.

    **Note:** For the test script to run successfully without errors, all external services (MLflow, Feast, Milvus) must be operational and populated with the necessary data and models as per your configuration. The script includes basic error handling if components fail to initialize.
## Online Inference - Stage 2: Filtering Service

This stage implements a `FilterService` that filters a list of candidate `post_id`s for a given `user_id` by removing posts the user has previously interacted with (seen, disliked, or reported). Interaction history is queried from a local ScyllaDB instance.

**Components:**

*   **`FilterService` (`src/inference/online/filtering/filter_service.py`):**
    *   `__init__(self, scylla_contact_points: list, scylla_keyspace: str, scylla_port: int)`: Initializes a connection to ScyllaDB.
    *   `_get_user_interactions(self, user_id: str, post_ids: list, interaction_types: list) -> set`: Queries ScyllaDB for `post_id`s that the `user_id` has interacted with, matching the given `post_ids` and `interaction_types`.
    *   `filter_candidates(self, user_id: str, candidate_post_ids: list) -> list`: Fetches posts the user has 'seen', 'disliked', or 'reported' and returns a new list of candidates excluding these interacted posts.
    *   `shutdown(self)`: Closes the ScyllaDB connection.

**Configuration:**

The service is configured via `config/filtering_service_config.yaml`. This file specifies:
*   `scylla_contact_points`: List of ScyllaDB host IPs/names.
*   `scylla_port`: Port for ScyllaDB (default 9042).
*   `scylla_keyspace`: The keyspace to use (e.g., `recsys_poc_interactions`).
*   `sample_test_data`: A list of sample interactions used by the local test script to populate ScyllaDB for testing purposes.

**ScyllaDB Setup (Local PoC):**

For the filtering service to work, you need a local ScyllaDB instance running.

1.  **Run ScyllaDB (e.g., using Docker):**
    ```bash
    docker run -p 9042:9042 --name scylla_filter_poc -d scylladb/scylla
    ```
    Wait a few moments for the ScyllaDB node to initialize.

2.  **Create Keyspace and Table:**
    You can use `cqlsh` to connect to your ScyllaDB instance and create the necessary schema. The `run_local_test.py` script also attempts to create these if they don't exist.
    ```bash
    docker exec -it scylla_filter_poc cqlsh
    ```
    Inside `cqlsh`:
    ```cql
    CREATE KEYSPACE IF NOT EXISTS recsys_poc_interactions WITH REPLICATION = {'class': 'SimpleStrategy', 'replication_factor': 1};
    USE recsys_poc_interactions;
    CREATE TABLE IF NOT EXISTS user_post_interactions (
        user_id TEXT,
        post_id TEXT, // Assuming post_id is text, adjust if it's int/uuid
        interaction_type TEXT, // 'seen', 'disliked', 'reported'
        interaction_timestamp TIMESTAMP,
        PRIMARY KEY ((user_id), post_id, interaction_type)
    );
    EXIT;
    ```

**Running the Local Test Script:**

A local test script `src/inference/online/filtering/run_local_test.py` demonstrates the `FilterService`.

1.  **Ensure Prerequisites:**
    *   A running ScyllaDB instance (as described above).
    *   The `cassandra-driver` is installed (it should be in `requirements.txt`).
    *   The configuration file `config/filtering_service_config.yaml` is present and correctly set up.

2.  **Execute the script:**
    From the project root directory:
    ```bash
    python src/inference/online/filtering/run_local_test.py
    ```
    The script will:
    *   Load configuration from `config/filtering_service_config.yaml`.
    *   Attempt to create the keyspace and table in ScyllaDB if they don't exist.
    *   Insert/update sample interaction data into ScyllaDB based on `sample_test_data` in the config. It clears previous data for test users defined in the sample data to ensure a clean test state.
    *   Instantiate the `FilterService`.
    *   Simulate a list of candidate `post_id`s.
    *   Call `filter_service.filter_candidates()` for different test users.
    *   Print the original and filtered lists of posts.
    *   Properly shut down the `FilterService` connection.

    The script includes logging, so you can observe its operations and the results of the filtering.

## Online Inference - Stage 3: Ranking Service

This stage implements the Ranking Service, which scores a list of filtered candidate `post_id`s for a given `user_id` using a trained Ranking Model. It simulates KServe's Predictor and Transformer pattern.

**Components:**

*   **`RankingModelPredictor` (`src/inference/online/ranking/predictor.py`):**
    *   Loads a trained Ranking Model (e.g., XGBoost, LightGBM, TensorFlow) from MLflow using a specified model URI and type.
    *   The `predict` method takes features prepared by the Transformer and returns model scores.
*   **`RankingServiceTransformer` (`src/inference/online/ranking/transformer.py`):**
    *   Initializes a connection to Feast for feature retrieval.
    *   Uses an instance of `RankingModelPredictor`.
    *   `preprocess(request_data)`: Fetches online features for the given `user_id` and each `post_id` from Feast. It constructs a feature DataFrame suitable for the Ranking Model.
    *   `postprocess(scores, original_post_ids)`: Combines the model scores with the original `post_id`s and sorts them by score in descending order.
    *   `rank_candidates(request_data)`: Orchestrates the flow: `preprocess` -> `RankingModelPredictor.predict` -> `postprocess`.

**Configuration:**

The service is configured via `config/ranking_service_config.yaml`. This file specifies:
*   `feast_repo_path`: Path to the Feast feature repository.
*   `ranking_model_uri`: MLflow URI for the trained Ranking Model.
*   `ranking_model_type`: Type of the model (e.g., "xgboost", "lightgbm", "tensorflow").
*   `user_features`: List of user features to fetch from Feast.
*   `post_features`: List of post features to fetch from Feast.
*   `computed_features`: List of features computed on-the-fly (e.g., `cosine_similarity_user_post_embedding`).

**Running the Local Test Script:**

A local test script `src/inference/online/ranking/run_local_test.py` demonstrates the Ranking Service.

1.  **Ensure Prerequisites:**
    *   A running MLflow server with the specified Ranking Model registered (e.g., `models:/RankingModelXGBoost/Production`).
    *   A running Feast instance with materialized user and post features.
    *   The configuration file `config/ranking_service_config.yaml` is correctly set up.

2.  **Execute the script:**
    From the project root directory:
    ```bash
    python src/inference/online/ranking/run_local_test.py --user_id "some_user_id" --post_ids "post_1" "post_2" "post_3"
    ```
    *   `--config`: Path to the service configuration YAML file. Defaults to `config/ranking_service_config.yaml`.
    *   `--user_id`: The ID of the user for whom to rank candidates.
    *   `--post_ids`: A list of candidate `post_id`s (space-separated).

    The script will:
    *   Load the configuration.
    *   Instantiate `RankingModelPredictor` and `RankingServiceTransformer`.
    *   Simulate a request with the provided `user_id` and `post_ids`.
    *   Call `transformer.rank_candidates()` and print the scored and sorted `post_id`s.

    **Note:** Ensure all external services (MLflow, Feast) are operational and populated with necessary data and models.

## Online Inference - Stage 4: Re-ranking & Presentation Service

This stage implements the `ReRankService`, which applies business logic, diversity rules, or other heuristics to a list of scored posts before presenting the final recommendations.

**Components:**

*   **`ReRankService` (`src/inference/online/re_ranking/re_rank_service.py`):**
    *   `__init__(self, config: dict)`: Initializes with configuration (e.g., diversity rules, promotion slots).
    *   `re_rank_and_present(self, scored_posts: list, num_final_recommendations: int) -> list`:
        *   Takes a list of `(post_id, score)` tuples.
        *   Applies re-ranking logic (e.g., ensure category diversity, promote certain items).
        *   Truncates the list to `num_final_recommendations`.
        *   Returns the final list of `post_id`s.

**Configuration:**

The service is configured via `config/re_ranking_service_config.yaml`. This file can specify:
*   `diversity_window_size`: How many recent items to consider for category diversity.
*   `max_items_per_category_in_window`: Max items from the same category within the diversity window.
*   `promoted_post_ids`: A list of post IDs to potentially boost or insert.

**Running the Local Test Script:**

A local test script `src/inference/online/re_ranking/run_local_test.py` demonstrates the `ReRankService`.

1.  **Ensure Prerequisites:**
    *   The configuration file `config/re_ranking_service_config.yaml` is present.
    *   (Optional) If fetching post details for diversity, ensure `posts.csv` or a similar data source is available as specified in the config.

2.  **Execute the script:**
    From the project root directory:
    ```bash
    python src/inference/online/re_ranking/run_local_test.py
    ```
    The script will:
    *   Load the configuration.
    *   Instantiate the `ReRankService`.
    *   Simulate a list of scored posts.
    *   Call `re_rank_service.re_rank_and_present()` with a specified number of final recommendations.
    *   Print the original scored posts and the final re-ranked list of `post_id`s.

    The test script includes sample data and demonstrates how diversity and promotion rules (if implemented) might affect the final output.


## End-to-End Online Inference Pipeline Test

This section describes how to run the complete end-to-end online inference pipeline locally using the [`run_inference_pipeline.py`](src/inference/online/pipeline/run_inference_pipeline.py:0) script. This script orchestrates all four stages: Candidate Generation, Filtering, Ranking, and Re-ranking.

**Flow Orchestrated:**

1.  **Candidate Generation:** Generates initial candidates using the User Tower and Milvus.
2.  **Filtering:** Removes posts the user has interacted with (seen, disliked, reported) using ScyllaDB.
3.  **Ranking:** Scores the filtered candidates using the Ranking Model and Feast features.
4.  **Re-ranking & Presentation:** Applies final business logic/diversity rules and selects the top N recommendations.

**Running the Script:**

To run the end-to-end pipeline, execute the following command from the project root:

```bash
python src/inference/online/pipeline/run_inference_pipeline.py \
  --user_id "user_123" \
  --categories 1 5 10 \
  --top_n_candidates 200 \
  --num_final_recommendations 25 \
  --cg_config "config/candidate_generation_service_config.yaml" \
  --filter_config "config/filtering_service_config.yaml" \
  --rank_config "config/ranking_service_config.yaml" \
  --re_rank_config "config/re_ranking_service_config.yaml"
```

**Command-Line Arguments:**

*   `--user_id` (required): ID of the user to get recommendations for.
*   `--categories` (required): Space-separated list of preferred category IDs for the user.
*   `--top_n_candidates` (optional, default: 100): Number of initial candidates to generate.
*   `--num_final_recommendations` (optional, default: 20): Number of final recommendations to return.
*   `--cg_config` (optional, default: `config/candidate_generation_service_config.yaml`): Path to Candidate Generation service config.
*   `--filter_config` (optional, default: `config/filtering_service_config.yaml`): Path to Filtering service config.
*   `--rank_config` (optional, default: `config/ranking_service_config.yaml`): Path to Ranking service config.
*   `--re_rank_config` (optional, default: `config/re_ranking_service_config.yaml`): Path to Re-ranking service config.

**Prerequisites:**

For the end-to-end pipeline script to run successfully, **ALL** of the following prerequisites must be met:

1.  **Synthetic Data Generated:** Ensure you have run `python src/data_generation/generate_synthetic_data.py`.
2.  **Feature Pipeline Run (Feast Materialized):**
    *   Feature transformation scripts executed:
        ```bash
        python src/feature_processing/generate_embeddings.py
        python src/feature_processing/generate_aggregated_features.py
        ```
    *   Feast definitions applied: `./scripts/feast_apply.sh`
    *   Features materialized into Feast: `./scripts/materialize_features.sh`
3.  **Models Trained and Registered:**
    *   **User Tower & Post Tower:** Trained using `src/model_training/two_tower/train.py` and registered in MLflow (e.g., `models:/TwoTowerUserTower/Production`, `models:/TwoTowerPostTower/Production`).
    *   **Ranking Model:** Trained using `src/model_training/ranking/train.py` and registered in MLflow (e.g., `models:/RankingModelXGBoost/Production`).
4.  **Milvus Populated:**
    *   A local Milvus instance must be running.
    *   Post embeddings must be indexed into Milvus using `src/inference/offline/index_posts_to_milvus.py`.
5.  **ScyllaDB Running with Data:**
    *   A local ScyllaDB instance must be running.
    *   The keyspace (e.g., `recsys_poc_interactions`) and table (`user_post_interactions`) must be created.
    *   It's recommended to have some sample interaction data in ScyllaDB for the filtering stage to be effective. The `src/inference/online/filtering/run_local_test.py` script can be used to populate some test data if needed, or you can manually insert data relevant to your test users.

Ensure all configuration files (`config/*.yaml`) point to the correct model URIs, service endpoints, and data paths for your local setup.