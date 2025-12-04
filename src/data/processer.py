import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from src.config.settings import ProjectSettings
from src.config.paths import ProjectPaths

def _encode_categorical_features(X):
    """Encode categorical features with LabelEncoder for tree models"""
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    encoders = {}
    
    X_encoded = X.copy()
    for col in cat_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col] = le
    
    return X_encoded, encoders

def prepare_non_scaled_data(data, label_col=ProjectSettings.TARGET_COLUMN):
    """Prepare data for tree models (preserves raw values)"""
    df = data.copy()
    X = df.drop(label_col, axis=1)
    y = LabelEncoder().fit_transform(df[label_col])
    
    # Encode categorical columns
    X_encoded, encoders = _encode_categorical_features(X)
    
    # Remove log columns (not needed for trees)
    X_encoded = X_encoded.drop(['Departure_Delay_Log', 'Arrival_Delay_Log'], axis=1, errors='ignore')
    
    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_encoded, y, test_size=ProjectSettings.TEST_SIZE, 
        random_state=ProjectSettings.RANDOM_STATE, stratify=y
    )
    val_ratio = ProjectSettings.VAL_SIZE / (1 - ProjectSettings.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, 
        random_state=ProjectSettings.RANDOM_STATE, stratify=y_train_val
    )
    
    print(f"Non-scaled data - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Prepare data dictionary
    data_dict = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'feature_names': X_encoded.columns.tolist(),
        'encoders': encoders
    }
    
    return data_dict

def prepare_scaled_data(data, label_col=ProjectSettings.TARGET_COLUMN):
    """Prepare data for linear models, KNN, MLP, etc. (standardized with one-hot encoding)"""
    df = data.copy()
    X = df.drop(label_col, axis=1)
    y = LabelEncoder().fit_transform(df[label_col])
    
    # Use log-transformed delays
    if 'Departure_Delay_Log' in X.columns:
        X = X.drop(['Departure Delay in Minutes', 'Arrival Delay in Minutes'], axis=1, errors='ignore')
    
    # One-hot encode categorical variables
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # Split and standardize
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=ProjectSettings.TEST_SIZE, 
        random_state=ProjectSettings.RANDOM_STATE, stratify=y
    )
    val_ratio = ProjectSettings.VAL_SIZE / (1 - ProjectSettings.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, 
        random_state=ProjectSettings.RANDOM_STATE, stratify=y_train_val
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    
    print(f"Scaled data - Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
    
    # Prepare data dictionary
    data_dict = {
        'X_train': X_train_scaled, 'X_val': X_val_scaled, 'X_test': X_test_scaled,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'scaler': scaler
    }
    
    return data_dict