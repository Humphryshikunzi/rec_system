#!/bin/bash

# This script orchestrates the full feature engineering pipeline.
# It includes optional data generation, feature transformation,
# applying Feast definitions, and materializing features.

set -e # Exit immediately if a command exits with a non-zero status.
PROJECT_ROOT=$(git rev-parse --show-toplevel) # Assumes script is in a git repo and run from anywhere within it
cd "$PROJECT_ROOT"

echo "Starting Feature Engineering Pipeline..."

# --- 1. Synthetic Data Generation (Optional) ---
# Uncomment the following lines if you need to regenerate synthetic data.
# Ensure config/data_config.yaml is configured as needed.
# echo "Step 1: Generating synthetic data..."
# python src/data_generation/generate_synthetic_data.py
# if [ $? -ne 0 ]; then
#     echo "Error during synthetic data generation. Exiting."
#     exit 1
# fi
# echo "Synthetic data generation complete."

# --- 2. Feature Transformation ---
echo "Step 2: Running feature transformation scripts..."

echo "Running generate_embeddings.py..."
python src/feature_processing/generate_embeddings.py
if [ $? -ne 0 ]; then
    echo "Error during embedding generation. Exiting."
    exit 1
fi
echo "Embedding generation complete."

echo "Running generate_aggregated_features.py..."
python src/feature_processing/generate_aggregated_features.py
if [ $? -ne 0 ]; then
    echo "Error during aggregated feature generation. Exiting."
    exit 1
fi
echo "Aggregated feature generation complete."
echo "Feature transformation scripts completed."

# --- 3. Apply Feast Definitions ---
echo "Step 3: Applying Feast definitions..."
./scripts/feast_apply.sh
if [ $? -ne 0 ]; then
    echo "Error during feast apply. Exiting."
    exit 1
fi
echo "Feast definitions applied successfully."

# --- 4. Materialize Features ---
echo "Step 4: Materializing features..."
# You can customize the date range for materialization if needed by passing arguments
# to materialize_features.sh, e.g.,
# ./scripts/materialize_features.sh "2023-01-01T00:00:00" "$(date -u +'%Y-%m-%dT%H:%M:%S')"
./scripts/materialize_features.sh
if [ $? -ne 0 ]; then
    echo "Error during feature materialization. Exiting."
    exit 1
fi
echo "Feature materialization complete."

echo "Feature Engineering Pipeline completed successfully!"