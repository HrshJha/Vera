#!/bin/bash
# Run the judge simulator with the omniroute Anthropic configuration.
# Set environment variables from the user's omniroute setup before running.

export ANTHROPIC_BASE_URL="http://localhost:20128"
export ANTHROPIC_AUTH_TOKEN="sk-4b525ec28dec6cd2-1551b7-3a6610fa"

cd "$(dirname "$0")"
source venv/bin/activate

# Run judge with anthropic provider using the override script
python run_judge.py "$@"
