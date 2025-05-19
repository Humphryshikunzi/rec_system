#!/bin/bash

# Navigate to the feature repository directory
cd "$(dirname "$0")/../src/feature_repo" || exit 1

# Activate virtual environment if it exists and is not already active
if [ -d "../../venv" ] && [ -z "$VIRTUAL_ENV" ]; then
  echo "Activating virtual environment..."
  source ../../venv/bin/activate
fi

# Get current date in UTC for incremental materialization
# For the very first run, a date far in the past ensures all existing data is materialized.
# For subsequent runs, this will pick up new data since the last materialization.
# Feast's materialize-incremental uses the current time as the end_date by default if not specified.
# We need a start_date for incremental. For a full initial load, a very old start date is used.

# Default start date (far in the past for initial full materialization)
START_DATE="1970-01-01T00:00:00"
# Default end date (current time in UTC)
END_DATE=$(date -u +"%Y-%m-%dT%H:%M:%S")

# Allow overriding start and end dates via command line arguments
if [ -n "$1" ]; then
    START_DATE="$1"
fi
if [ -n "$2" ]; then
    END_DATE="$2"
fi

echo "Running feast materialize-incremental from $START_DATE to $END_DATE..."
# feast materialize-incremental <end_date> is the typical command.
# To specify a range, you use feast materialize <start_date> <end_date>
# However, materialize-incremental is designed to pick up from the last run.
# For simplicity and to align with the task's "materialize all available data" for PoC,
# using a fixed past date for the first run with `materialize` (not incremental) is safer.
# Let's use `materialize` for clarity with explicit start/end.
# `materialize-incremental` is more for ongoing updates.

# For initial full materialization, use a date far in the past.
# The task mentions `materialize-incremental YYYY-MM-DDTHH:MM:SS` which implies an end date.
# Let's stick to `materialize` for explicit range for now, which is clearer for initial load.
# If `materialize-incremental` is strictly required, it usually takes only the end_date.

# Using `materialize` with a start and end date for clarity in this PoC.
# For a true "incremental" after first load, `materialize-incremental <current_date>` would be used.
echo "Attempting to materialize features between $START_DATE and $END_DATE."
feast materialize "$START_DATE" "$END_DATE"

# Alternative using materialize-incremental (if preferred, typically for subsequent runs):
# echo "Running feast materialize-incremental up to $END_DATE..."
# feast materialize-incremental "$END_DATE"

echo "Feast materialize command executed."
echo "Note: For the very first materialization, ensure START_DATE is set to a date before any data timestamps."
echo "For subsequent incremental materializations, adjust START_DATE or use 'feast materialize-incremental <current_date>'."