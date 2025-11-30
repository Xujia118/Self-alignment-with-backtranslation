#!/bin/bash
set -euo pipefail

USERNAME="ubuntu"
SCRIPT_PATH="scripts.01_generate_instruction"

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
