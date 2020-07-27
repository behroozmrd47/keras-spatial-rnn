import numpy as np

# MODEL_ID = 'All_data_UNET3D_256_16_BC1_EP100'
MODEL_ID_2D = 'All_data_UNET2D_RES_CR_256_16_BC1_EP100'
# MODEL_ID_HYB = 'All_data_HYBD3D_256_16_BC1_EP100'

INCLUDED_GROUPS = ['Female_Negative', 'Male_Negative', 'Female_Positive', 'Male_Positive']
# INCLUDED_GROUPS = ['Male_Negative', 'Female_Positive', 'Male_Positive']
# INCLUDED_GROUPS = ['Male_Positive_Limited']
TRAIN_RAW_DATA_FOLDER = 'data/raw/train/'
TEST_RAW_DATA_FOLDER = 'data/raw/test/'
PROCESSED_DATA_FOLDER = 'data/processed/'
SAVED_MODEL_FOLDER = 'weights/'
SAVED_LOGS_FOLDER = 'logs/'
SAVED_OUTPUT_FOLDER = 'test/output/'
TRAIN_INPUT_FOLDER = PROCESSED_DATA_FOLDER

TRAIN_TEST_RATIO = 0.01
BATCH_SIZE = 1
EPOCH = 100

IMAGE_DATA_TYPE = np.int16
IMG_ROWS = 256
IMG_COLS = 256
IMG_DEPTH = 16
SMOOTH = 1.

from src.utils.utility_functions import *

check_exist_folder(TRAIN_RAW_DATA_FOLDER, create_if_not_exist=True)
check_exist_folder(TEST_RAW_DATA_FOLDER, create_if_not_exist=True)
check_exist_folder(PROCESSED_DATA_FOLDER, create_if_not_exist=True)
check_exist_folder(SAVED_MODEL_FOLDER, create_if_not_exist=True)
check_exist_folder(SAVED_LOGS_FOLDER, create_if_not_exist=True)
check_exist_folder(SAVED_OUTPUT_FOLDER, create_if_not_exist=True)
