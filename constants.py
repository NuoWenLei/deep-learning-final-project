BASE_FILEPATH = "/content/gdrive/MyDrive/personal_stuff/personal_projects_college/CSCI2470_FINAL_PROJECT/Model_Playground/Action_Train_Results"
# BASE_FILEPATH = "."
LOG_FILEPATH = "log/fullLog.txt"
RESULT_DIRPATH = "results"
RESULT_SAMPLE_RATE = 300
SAMPLE_BATCH_SIZE = 32
LATENT_SAMPLE_PATH = [
	f"./data/sample_latents/sample_latents_{i}.npy" for i in range(1, 5)
]

# VQVAE/ACTION HYPERPARAMETERS
VQVAE_LOSS_LAMBDA = 0.1
VQVAE_COMMITMENT_COST = 0.25
VQVAE_DECAY = 0.99

VQVAE_NUM_EMBEDDINGS = 32
VQVAE_EMBEDDING_DIM = 512

VQVAE_INPUT_SHAPE = (4, 4, 512)
VQVAE_OUTPUT_SHAPE = (1, 1, 512)
VQVAE_NUM_BLOCKS_WITH_ACTION = 2

NUM_FUTURE_FRAMES = 10

# "./data/sample_latents.npy"
CHECKPOINT_PATH = "checkpoints"
CHECKPOINT_SAVE_RATE = 50

EXPLODING_LOSS_DETECTION = 5

USE_EMAIL_NOTIFICATION = False

USE_SAMPLE_DATA = True

# Training Parameters
BATCH_SIZE = 128
NUM_EPOCHS = 1000

# Model Parameters
LATENT_SHAPE = (32, 32, 4)
NUM_FILTERS=512
NUM_BLOCKS=3
NUM_PREV_FRAMES=120
DROPOUT_PROB=0
FILTER_SIZE=3
NOISE_LEVELS=10
LEAKY=0.05

# Optimizer
CLIPNORM = None
INITIAL_LEARNING_RATE = 0.0
LEARNING_RATE = 1e-5
LR_WARMUP = 1000
# Warmup to stabliize learning rate, otherwise loss may explode
# Inspiration:
# - https://stats.stackexchange.com/questions/324896/training-loss-increases-with-time
# - https://stackoverflow.com/questions/63213252/using-learning-rate-schedule-and-learning-rate-warmup-with-tensorflow2