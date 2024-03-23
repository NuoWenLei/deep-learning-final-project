LOG_FILEPATH = "./log/fullLog.txt"
RESULT_DIRPATH = "./results"
RESULT_SAMPLE_RATE = 10
SAMPLE_BATCH_SIZE = 32
LATENT_SAMPLE_PATH = "./data/sample_latents.npy"
CHECKPOINT_PATH = "./checkpoints"
CHECKPOINT_SAVE_RATE = 10

USE_SAMPLE_DATA = True

# Training Parameters
BATCH_SIZE = 128
NUM_EPOCHS = 100

# Model Parameters
LATENT_SHAPE = (32, 32, 4)
NUM_FILTERS=512
NUM_BLOCKS=3
NUM_PREV_FRAMES=20
DROPOUT_PROB=0
FILTER_SIZE=3
NOISE_LEVELS=10
LEAKY=0.05

# Optimizer
LEARNING_RATE = 1e-4