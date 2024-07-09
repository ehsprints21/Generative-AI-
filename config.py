import os

# Directory paths
BASE_DIR = '/home/abhishek/svg-model-project'
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
GENERATED_DATA_DIR = os.path.join(DATA_DIR, 'generated')

# Model parameters
LATENT_DIM = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
BATCH_SIZE = 16
