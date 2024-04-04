BASE_FILEPATH = "/content/gdrive/MyDrive/personal_stuff/personal_projects_college/CSCI2470_FINAL_PROJECT/Model_Playground/Sample_Train_Results_2"
# BASE_FILEPATH = "."
LOG_FILEPATH = "log/fullLog.txt"
RESULT_DIRPATH = "results"
RESULT_SAMPLE_RATE = 300
SAMPLE_BATCH_SIZE = 32
LATENT_SAMPLE_PATH = [
	f"./data/sample_latents_{i}.npy" for i in range(1, 5)
]

# "./data/sample_latents.npy"
CHECKPOINT_PATH = "checkpoints"
CHECKPOINT_SAVE_RATE = 50

EXPLODING_LOSS_DETECTION = 5

USE_EMAIL_NOTIFICATION = False

USE_SAMPLE_DATA = True

# Training Parameters
BATCH_SIZE = 128
NUM_EPOCHS = 200

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