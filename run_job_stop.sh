#!/bin/bash
set -euo pipefail

USERNAME="ubuntu"
SCRIPT_PATH="scripts.01_generate_instruction"

# Set HuggingFace cache directory to persistent volume
export HF_HOME="/workspace/huggingface"
mkdir -p "$HF_HOME"

# Create a symlink for HF cache to ensure nothing writes to container disk
mkdir -p /runpod-volume/hfcache
rm -rf ~/.cache/huggingface || true
ln -s /runpod-volume/hfcache ~/.cache/huggingface

python3 -m venv venv
source venv/bin/activate
echo "Virtual environment activated"

pip install -r requirements.txt

# Run the training script
echo "Starting training script..."
python3 -m "$SCRIPT_PATH"
EXIT_CODE=$?
echo "Training script finished with exit code: $EXIT_CODE"

echo "Shutting down instance..."
sudo shutdown -h now
