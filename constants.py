
# TRAINING ORGANIZATION
BASE_FILEPATH = "/content/gdrive/MyDrive/personal_stuff/personal_projects_college/CSCI2470_FINAL_PROJECT/Model_Playground/Action_Train_Results_8"
LOG_FILEPATH = "log/fullLog.txt"
RESULT_DIRPATH = "results"
RESULT_SAMPLE_RATE = 300 # number of epochs to pass before generating an example
SAMPLE_BATCH_SIZE = 32 # batch size for sampling
LATENT_SAMPLE_PATH = [ # paths to all preprocessed data files
	f"./data/sample_latents/sample_latents_{i}.npy" for i in range(1, 14)
]
CHECKPOINT_PATH = "checkpoints" # folder path to save model checkpoints
CHECKPOINT_SAVE_RATE = 30 # number of epochs per model parameter checkpoint

# VQVAE/ACTION
VQVAE_LOSS_LAMBDA = 1.0 # scale of VQ-VAE loss term in total loss
VQVAE_COMMITMENT_COST = 0.25 # scale of commitment loss in relation to codebook loss
VQVAE_DECAY = 0.99 # DEPRICATED, parameter for EMA VQ-VAE which is not used in this project

VQVAE_NUM_EMBEDDINGS = 8 # number of actions
VQVAE_EMBEDDING_DIM = 512 # dimension of each action

VQVAE_INPUT_SHAPE = (4, 4, 512) # input dimension for encoder in Vector Quantizer
VQVAE_OUTPUT_SHAPE = (1, 1, 512) # output dimension for encoder in Vector Quantizer
VQVAE_NUM_BLOCKS_WITH_ACTION = 2 # number of U-Net decoder blocks to condition with action embeddings

VQVAE_EXPLORE_STEPS = 5000 # DEPRICATED
VQVAE_WARMUP_STEPS = 0 # DEPRICATED

NUM_FUTURE_FRAMES = 4 # frame offset between future and current frames for Latent Action Model
UNCONDITION_PROB=0.15 # unconditional probability during Latent Action Model training

# TRAINING
EXPLODING_LOSS_DETECTION = 5 # scale for detecting exploding loss
USE_EMAIL_NOTIFICATION = False # DEPRICATED, set to false
USE_SAMPLE_DATA = True # DEPRICATED, set to true
BATCH_SIZE = 128 # training batch size
NUM_EPOCHS = 500 # number of epochs to train

# UNET
LATENT_SHAPE = (32, 32, 4) # input shape per frame
NUM_FILTERS=512 # number of filters for each convolutional layer
NUM_BLOCKS=3 # number of U-Net Encoder and Decoder blocks
NUM_PREV_FRAMES=120 # number of previous frames
DROPOUT_PROB=0 # dropout percentage for each convolutional layer
FILTER_SIZE=3 # kernel size for each convolutional layer
NOISE_LEVELS=10 # number of noise levels for diffusion corruption
LEAKY=0.05 # leaky relu percentage
CONDITIONAL_SAMPLING_LAMBDA=5. # DEPRICATED, can be adjusted through sampling function
GRAM_MATRIX_LAMBDA=0.3 # DEPRICATED, gram matrix loss term no longer in use

# OPTIMIZER
CLIPNORM = None # DEPRICATED, set to None
INITIAL_LEARNING_RATE = 0.0 # DEPRICATED, set to 0.0
LEARNING_RATE = 1e-4 # model learning rate
LR_WARMUP = 500 # number of steps to warm up learning rate, may be turned off through ccv_action_trainer.py function parameters
# Warmup to stabliize learning rate, otherwise loss may explode
# Inspiration:
# - https://stats.stackexchange.com/questions/324896/training-loss-increases-with-time
# - https://stackoverflow.com/questions/63213252/using-learning-rate-schedule-and-learning-rate-warmup-with-tensorflow2