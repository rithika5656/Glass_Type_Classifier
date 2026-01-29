import logging
import requests
from . import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def download_dataset(url, dest):
    """Download dataset from URL."""
    try:
        logger.info(f"Downloading data from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(dest, 'wb') as f:
            f.write(response.content)
        logger.info("Download complete.")
        return True
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        return False


def validate_schema(df):
    """Validate that the dataframe has the required columns."""
    missing = [col for col in config.FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if config.TARGET_COLUMN not in df.columns:
        # It's ok if target column is missing for prediction data, but here we assume mainly for training
        logger.warning(f"Target column '{config.TARGET_COLUMN}' not found.")
    return True


def load_data(filepath=None):
    """
    Load the glass dataset from CSV.
    
    Args:
        filepath (str): Path to the CSV file. If None, uses default from config.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if filepath is None:
        filepath = config.DATA_FILE
        
    logger.info(f"Loading data from {filepath}")
    
    if not os.path.exists(filepath):
        logger.warning(f"File not found at {filepath}. Attempting download...")
        if download_dataset(config.DATA_URL, filepath):
             # verify existence after download
             if not os.path.exists(filepath):
                  raise FileNotFoundError("Download appears successful but file is missing.")
        else:
             logger.error("Download failed.")
             raise FileNotFoundError(f"Dataset not found at {filepath} and download failed.")

    
    # Check if header exists by peeking
    try:
        # Based on notebook, the file seems to have no headers sometimes or it might have them.
        # The notebook assigns names manually: names=['Id'] + column_names or just column_names.
        # Looking at the file listing, it's 'glass.csv'. Let's assume standard format or handle it.
        # The notebook code: df = pd.read_csv(local_file) then checks if 'Type' is in columns.
        
        df = pd.read_csv(filepath)
        
        # If columns don't match expected features, we might need to assign them
        # Note: The notebook output showed headers in the DataFrame display, so the CSV likely has headers.
        # However, one cell does: df = pd.read_csv(url, names=['Id'] + column_names); df = df.drop('Id', ...)
        # The local file might have been saved WITH headers in the notebook: df.to_csv(local_file, index=False)
        
        if config.TARGET_COLUMN not in df.columns:
             # Assume it's raw UCI data without headers if 'Type' is missing
             # UCI data usually has ID as first col
             if df.shape[1] == 11:
                 df.columns = ['Id'] + config.COLUMNS
                 df = df.drop('Id', axis=1)
             elif df.shape[1] == 10:
                 df.columns = config.COLUMNS
                 
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise OSError(f"Error loading data: {e}")
    
    validate_schema(df)
    logger.info(f"Data loaded successfully with shape {df.shape}")
        
    return df


def get_class_name(class_id):
    """Return the name of the glass type."""
    return config.GLASS_TYPES.get(class_id, "Unknown")
