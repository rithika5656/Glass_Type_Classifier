import os
import logging


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(BASE_DIR, 'glass.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Data Column Names
COLUMNS = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
FEATURE_COLUMNS = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
TARGET_COLUMN = 'Type'

# Class Mappings
GLASS_TYPES = {
    1: 'Building Windows (Float)',
    2: 'Building Windows (Non-Float)',
    3: 'Vehicle Windows (Float)',
    5: 'Containers',
    6: 'Tableware',
    7: 'Headlamps'
}

# Model Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
KNN_N_NEIGHBORS = 3
SVM_KERNEL = 'rbf'
RF_N_ESTIMATORS = 100
GB_LEARNING_RATE = 0.1
GB_N_ESTIMATORS = 100

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO
