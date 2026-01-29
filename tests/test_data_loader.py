import pytest
import pandas as pd
import os
from src.glass_classifier import data_loader, config

def test_config_paths():
    assert config.DATA_FILE.endswith('glass.csv')

def test_get_class_name():
    assert data_loader.get_class_name(1) == 'Building Windows (Float)'
    assert data_loader.get_class_name(999) == 'Unknown'
    
# Mock loading validation could be added here
