import pandas as pd
import numpy as np
import random
import os
import platform
import sys


def summary_stats(df):
    """Return summary statistics."""
    return df.describe()

def check_missing(df):
    """Check for missing values."""
    return df.isnull().sum()

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_system_info():
    """Return system information."""
    return {
        "OS": platform.system(),
        "Python": sys.version,
        "Pandas": pd.__version__,
        "NumPy": np.__version__
    }

