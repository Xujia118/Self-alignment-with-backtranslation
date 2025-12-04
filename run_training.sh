#!/bin/bash
set -euo pipefail

SCRIPT_PATH="scripts.03_fine_tune_backward_model"

echo "Starting training..."
python3 -m "$SCRIPT_PATH"
EXIT_CODE=$?