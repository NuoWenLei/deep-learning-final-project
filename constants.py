BASE_FILEPATH = "/content/gdrive/MyDrive/personal_stuff/personal_projects_college/CSCI2470_FINAL_PROJECT/Model_Playground/Sample_Train_Results_120_prev_frames"
# BASE_FILEPATH = "."
LOG_FILEPATH = "log/fullLog.txt"
RESULT_DIRPATH = "results"
RESULT_SAMPLE_RATE = 300
SAMPLE_BATCH_SIZE = 32
LATENT_SAMPLE_PATH = [
	f"./data/sample_latents/sample_latents_{i}.npy" for i in range(1, 5)
]

# "./data/sample_latents.npy"
CHECKPOINT_PATH = "checkpoints"
CHECKPOINT_SAVE_RATE = 50

EXPLODING_LOSS_DETECTION = 5

USE_EMAIL_NOTIFICATION = False

USE_SAMPLE_DATA = True

# Training Parameters
BATCH_SIZE = 128
NUM_EPOCHS = 50

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