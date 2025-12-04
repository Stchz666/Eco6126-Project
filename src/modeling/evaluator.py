import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from config.paths import ProjectPaths

def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, model_name):
    """Evaluate a model and return metrics"""
    
    # Train model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred_test = model.predict(X_test)
    y_pred_val = model.predict(X_val)
    
    # Calculate metrics on test set
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='weighted')
    recall = recall_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    
    # Calculate AUC if possible
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = np.nan
    
    # Calculate validation accuracy
    val_accuracy = accuracy_score(y_val, y_pred_val)
    
    metrics = {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'auc_curve': round(auc, 4) if not np.isnan(auc) else np.nan,
        'val_accuracy': round(val_accuracy, 4),
        'training_time': round(training_time, 4),
        'n_features': X_train.shape[1]
    }
    
    return metrics

def tune_hyperparameters(model, param_grid, X_train, y_train, X_val, y_val):
    """Tune hyperparameters using GridSearchCV"""
    print("    Tuning hyperparameters...")
    
    # Use GridSearchCV with validation set
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=[(np.arange(len(X_train)), np.arange(len(X_train), len(X_train)+len(X_val)))],
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0
    )
    
    # Combine train and validation for tuning
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    grid_search.fit(X_combined, y_combined)
    
    print(f"    Best parameters: {grid_search.best_params_}")
    print(f"    Best score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_