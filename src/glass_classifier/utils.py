import pandas as pd

def summary_stats(df):
    """Return summary statistics."""
    return df.describe()

def check_missing(df):
    """Check for missing values."""
    return df.isnull().sum()
