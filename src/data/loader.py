import pandas as pd
from src.config.settings import ProjectSettings
from src.config.paths import ProjectPaths

def load_data(file_path=None):
    """Load dataset and display basic info"""
    if file_path is None:
        file_path = ProjectPaths.RAW_DATA_FILE
    
    print(f"Loading data from: {file_path}")
    data = pd.read_csv(file_path)
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    return data