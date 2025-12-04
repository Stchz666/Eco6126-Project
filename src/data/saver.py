import os
import pandas as pd
from src.config.paths import ProjectPaths

def save_processed_data(data_dict, data_type):
    """Save processed data to CSV files"""
    # Save each dataset
    datasets = {
        'train': (data_dict['X_train'], data_dict['y_train']),
        'val': (data_dict['X_val'], data_dict['y_val']),
        'test': (data_dict['X_test'], data_dict['y_test'])
    }
    
    for dataset_name, (X, y) in datasets.items():
        # Create DataFrame with features and target
        data_df = pd.DataFrame(X, columns=data_dict['feature_names'])
        data_df['target'] = y
        
        # Save to CSV
        filename = ProjectPaths.PROCESSED_DIR / f"{data_type}_{dataset_name}.csv"
        data_df.to_csv(filename, index=False)
        print(f"Saved: {filename}")
    
    # Save feature names
    feature_names_df = pd.DataFrame({'feature_name': data_dict['feature_names']})
    feature_names_df.to_csv(ProjectPaths.PROCESSED_DIR / f"{data_type}_features.csv", index=False)
    print(f"Saved feature names: {ProjectPaths.PROCESSED_DIR / f'{data_type}_features.csv'}")

def load_saved_data(data_type):
    """Load previously saved processed data"""
    print(f"Loading {data_type} data from CSV files...")
    
    datasets = {}
    for dataset_name in ['train', 'val', 'test']:
        filename = ProjectPaths.PROCESSED_DIR / f"{data_type}_{dataset_name}.csv"
        if os.path.exists(filename):
            data_df = pd.read_csv(filename)
            
            # Separate features and target
            X = data_df.drop('target', axis=1)
            y = data_df['target'].values
            
            datasets[f'X_{dataset_name}'] = X
            datasets[f'y_{dataset_name}'] = y
            
            print(f"Loaded: {filename} - Shape: {X.shape}")
        else:
            print(f"File not found: {filename}")
    
    # Load feature names
    feature_file = ProjectPaths.PROCESSED_DIR / f"{data_type}_features.csv"
    if os.path.exists(feature_file):
        feature_names = pd.read_csv(feature_file)['feature_name'].tolist()
        datasets['feature_names'] = feature_names
        print(f"Loaded feature names from: {feature_file}")
    
    return datasets