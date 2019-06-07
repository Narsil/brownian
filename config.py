import datetime

SAVE_DIR = "checkpoints"
SEED = 42
LOG_INTERVAL = datetime.timedelta(seconds=1)
TEST_INTERVAL = datetime.timedelta(seconds=30)
EPOCHS = 200
PATIENCE = 25
CONTEXT_SIZE = 64
BATCH_SIZE = 32
VOCAB_SIZE = 1001
EMBEDDING_DIM = 128
LEARNING_RATE = 1e-3

# GPT-2
N_HEAD = 8
N_BLOCKS = 12
