#!/bin/bash
set -euo pipefail

USERNAME="ubuntu"
PROJECT_FOLDER="Self-alignment-with-backtranslation"
RUN_SCRIPT="python -m scripts.01_generate_instruction"

cd /home/"$USERNAME"/"$PROJECT_FOLDER"

# Run the training script
echo "Starting training script..."
"$RUN_SCRIPT"
EXIT_CODE=$?
echo "Training script finished with exit code: $EXIT_CODE"

echo "Shutting down instance..."
sudo shutdown -h now
