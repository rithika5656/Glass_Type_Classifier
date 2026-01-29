from sklearn.preprocessing import StandardScaler, RobustScaler
from . import config
import numpy as np


def split_data(df, target_col=None, test_size=None, random_state=None, stratify=True):

    """
    Split data into features (X) and target (y), then train/test sets.
    
    Args:
        df (pd.DataFrame): Dataframe.
        target_col (str): Target column name.
        test_size (float): Proportion of test set.
        random_state (int): Seed.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    if target_col is None:
        target_col = config.TARGET_COLUMN
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE
        
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)


def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features.
        X_test: Test features.
        
    Returns:
        tuple: Scaled X_train, Scaled X_test, scaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler
