#!/bin/bash
set -euo pipefail

USERNAME="ubuntu"
SCRIPT_PATH="scripts.03_fine_tune_backward_model"

# Set HuggingFace cache directory to persistent volume
export HF_HOME="/workspace/huggingface"
mkdir -p "$HF_HOME"

python3 -m venv venv
source /venv/bin/activate
echo "Virtual environment activated"

pip install -r requirements.txt

# Run the training script
echo "Starting testing..."
python3 -m "$SCRIPT_PATH"
EXIT_CODE=$?
echo "Training script finished with exit code: $EXIT_CODE"
