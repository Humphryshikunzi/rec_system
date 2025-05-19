#!/bin/bash

# Navigate to the feature repository directory
cd "$(dirname "$0")/../src/feature_repo" || exit 1

# Activate virtual environment if it exists and is not already active
if [ -d "../../venv" ] && [ -z "$VIRTUAL_ENV" ]; then
  echo "Activating virtual environment..."
  source ../../venv/bin/activate
fi

# Apply the Feast definitions
echo "Running feast apply..."
feast apply

echo "Feast apply completed."