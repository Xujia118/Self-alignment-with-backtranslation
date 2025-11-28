BASE_MODEL_ID = "meta-llama/Llama-2-7b-hf"
NEW_MODEL_ID = "backwards-llama2-7b-guanaco"
DATASET_ID = "timdettmers/openassistant-guanaco"
HF_DATASET_REPO = "xujia118/guanaco"

CHECKPOINT_DIR="checkpoint"

# Training parameters
TRAIN_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
MAX_STEPS = 140 # The loss is decreasing very slowly after 120
MAX_LENGTH = 1024
LEARNING_RATE = 2e-5
EVAL_STEPS = 10
SAVE_STEPS = 25
LOGGING_STEPS = 10