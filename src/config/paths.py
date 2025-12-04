import os
from pathlib import Path

class ProjectPaths:
    """Centralized paths for the project"""
    ROOT = Path(__file__).parent.parent
    
    # Data paths
    RAW_DATA_DIR = ROOT / "data"
    RAW_DATA_FILE = RAW_DATA_DIR / "airline.csv"
    
    # Processed data paths
    PROCESSED_DIR = ROOT / "processed_data"
    FULL_DATASET = PROCESSED_DIR / "full_processed_dataset.csv"
    
    # Results paths
    RESULTS_DIR = ROOT / "results"
    METRICS_DIR = RESULTS_DIR / "metrics"
    FIGURES_DIR = RESULTS_DIR / "figures"
    
    # Feature selection paths
    FEATURE_SELECTION_DIR = ROOT / "feature_selection"
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories if they don't exist"""
        for directory in [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DIR,
            cls.METRICS_DIR,
            cls.FIGURES_DIR,
            cls.FEATURE_SELECTION_DIR
        ]:
            os.makedirs(directory, exist_ok=True)