#!/bin/bash
set -euo pipefail

USERNAME="ubuntu"
PROJECT_FOLDER="Self-alignment-with-backtranslation"
SCRIPT_PATH="scripts.01_generate_instruction"

cd /home/"$USERNAME"/"$PROJECT_FOLDER"

# Run the training script
echo "Starting training script..."
python3 -m "$SCRIPT_PATH"
EXIT_CODE=$?
echo "Training script finished with exit code: $EXIT_CODE"

echo "Shutting down instance..."
sudo shutdown -h now
