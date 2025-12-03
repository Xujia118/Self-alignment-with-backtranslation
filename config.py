BASE_MODEL_ID = "meta-llama/Llama-2-7b-hf"
JUDGE_MODEL_DI = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_ID = "xujia118/backwards-llama2-7b-guanaco"
DATASET_ID = "timdettmers/openassistant-guanaco"

HF_DATASET_REPO = "xujia118/guanaco"
HF_DATASET_LIMA_SINGLE_TURN = "xujia118/lima-single-turn"
HF_CURATED_LIMA_V3 = "xujia118/curated_lima_v3"
HF_LIMA_BEST_SAMPLES = "xujia118/self_curated_lima_v2"

CHECKPOINT_DIR="./checkpoints"
BATCH_SIZE=10

# Training parameters
TRAIN_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
MAX_STEPS = 140 # The loss is decreasing very slowly after 120
MAX_LENGTH = 1024
LEARNING_RATE = 2e-5
EVAL_STEPS = 10
SAVE_STEPS = 25
LOGGING_STEPS = 10