import pandas as pd
import numpy as np
import os
from data import load_saved_data
from modeling import (get_all_models, apply_feature_selection, 
                     evaluate_model, tune_hyperparameters)
from visualization.results import (
    plot_round_results, plot_round3_comparison, plot_comparison_table
)
from config.paths import ProjectPaths
from config.settings import ProjectSettings

def _load_data_for_model(model_info):
    """Load appropriate data based on model type"""
    if model_info['data_type'] == 'scaled':
        return load_saved_data('scaled')
    else:
        return load_saved_data('non_scaled')

def run_round1():
    """
    ROUND 1: Preprocessed dataset with default model functions
    """
    print("\n" + "="*60)
    print("ROUND 1: Preprocessed Data + Default Parameters")
    print("="*60)
    
    # Get all models
    all_models = get_all_models()
    
    # Store results
    results = []
    
    # Train and evaluate each model
    for model_name, model_info in all_models.items():
        print(f"\nTraining {model_name} (default parameters)...")
        
        # Load appropriate data
        data = _load_data_for_model(model_info)
        X_train, X_val, X_test, y_train, y_val, y_test = (
            data['X_train'], data['X_val'], data['X_test'],
            data['y_train'], data['y_val'], data['y_test']
        )
        
        # Get model with default parameters
        model = model_info['model']
        
        # Evaluate model
        metrics = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, model_name)
        
        # Store results
        results.append({
            'round': 1,
            'model_name': model_name,
            'data_type': model_info['data_type'],
            'parameters': 'default',
            'feature_selection': 'no',
            **metrics
        })
        
        print(f"  Test Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, Features: {metrics['n_features']}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(ProjectPaths.METRICS_DIR / 'round1_results.csv', index=False)
    print(f"Saved round1 results to {ProjectPaths.METRICS_DIR / 'round1_results.csv'}")
    
    return results_df

def run_round2():
    """
    ROUND 2: Preprocessed dataset with tuned model functions
    """
    print("\n" + "="*60)
    print("ROUND 2: Preprocessed Data + Tuned Parameters")
    print("="*60)
    
    # Get all models
    all_models = get_all_models()
    
    # Store results
    results = []
    best_params_dict = {}  # Store best parameters for each model
    
    # Train and evaluate each model
    for model_name, model_info in all_models.items():
        print(f"\nTraining {model_name} (tuned parameters)...")
        
        # Load appropriate data
        data = _load_data_for_model(model_info)
        X_train, X_val, X_test, y_train, y_val, y_test = (
            data['X_train'], data['X_val'], data['X_test'],
            data['y_train'], data['y_val'], data['y_test']
        )
        
        # Get model and parameter grid
        model = model_info['model']
        param_grid = model_info['param_grid']
        
        # Tune hyperparameters
        tuned_model, best_params = tune_hyperparameters(model, param_grid, X_train, y_train, X_val, y_val)
        
        # Store best parameters
        best_params_dict[model_name] = best_params
        
        # Evaluate tuned model
        metrics = evaluate_model(tuned_model, X_train, X_val, X_test, y_train, y_val, y_test, model_name)
        
        # Store results
        results.append({
            'round': 2,
            'model_name': model_name,
            'data_type': model_info['data_type'],
            'parameters': 'tuned',
            'feature_selection': 'no',
            **metrics
        })
        
        print(f"  Test Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, Features: {metrics['n_features']}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(ProjectPaths.METRICS_DIR / 'round2_results.csv', index=False)
    print(f"Saved round2 results to {ProjectPaths.METRICS_DIR / 'round2_results.csv'}")
    
    return results_df, best_params_dict

def run_round3():
    """
    ROUND 3: Feature selected dataset with default model functions
    """
    print("\n" + "="*60)
    print("ROUND 3: Feature Selected Data + Default Parameters")
    print("="*60)
    
    # Get all models
    all_models = get_all_models()
    
    # Store results
    results = []
    feature_selectors = {}  # Store feature selectors for each model
    n_features_dict = {}    # Store number of features selected for each model
    
    # Get Round 1 results for comparison
    round1_results = pd.read_csv(ProjectPaths.METRICS_DIR / 'round1_results.csv')
    
    # Train and evaluate each model
    for model_name, model_info in all_models.items():
        print(f"\nTraining {model_name} (feature selected, default parameters)...")
        
        # Load appropriate data
        data = _load_data_for_model(model_info)
        X_train, X_val, X_test, y_train, y_val, y_test = (
            data['X_train'], data['X_val'], data['X_test'],
            data['y_train'], data['y_val'], data['y_test']
        )
        
        # Apply feature selection
        print(f"  Applying feature selection for {model_name}...")
        X_train_fs, X_val_fs, X_test_fs, n_selected, selector = apply_feature_selection(
            model_name, X_train, X_val, X_test, y_train,
            n_features=ProjectSettings.NUM_FEATURES_TO_SELECT
        )
        
        # Store selector and number of features
        feature_selectors[model_name] = selector
        n_features_dict[model_name] = n_selected
        
        # Get model with default parameters
        model = model_info['model']
        
        # Evaluate model with feature selected data
        metrics = evaluate_model(model, X_train_fs, X_val_fs, X_test_fs, y_train, y_val, y_test, model_name)
        
        # Store results
        results.append({
            'round': 3,
            'model_name': model_name,
            'data_type': model_info['data_type'],
            'parameters': 'default',
            'feature_selection': 'yes',
            'n_selected_features': n_selected,
            **metrics
        })
        
        print(f"  Test Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, Features: {n_selected}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(ProjectPaths.METRICS_DIR / 'round3_results.csv', index=False)
    print(f"Saved round3 results to {ProjectPaths.METRICS_DIR / 'round3_results.csv'}")
    
    # Create comparison table with Round 1
    comparison_data = []
    
    for model_name in all_models.keys():
        # Get Round 1 results
        r1_result = round1_results[round1_results['model_name'] == model_name].iloc[0]
        
        # Get Round 3 results
        r3_result = results_df[results_df['model_name'] == model_name].iloc[0]
        
        comparison_data.append({
            'model_name': model_name,
            'round1_accuracy': r1_result['accuracy'],
            'round3_accuracy': r3_result['accuracy'],
            'round1_f1': r1_result['f1_score'],
            'round3_f1': r3_result['f1_score'],
            'accuracy_change': r3_result['accuracy'] - r1_result['accuracy'],
            'f1_change': r3_result['f1_score'] - r1_result['f1_score'],
            'n_features_round1': r1_result['n_features'],
            'n_features_round3': r3_result['n_selected_features']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison
    comparison_df.to_csv(ProjectPaths.METRICS_DIR / 'round3_comparison.csv', index=False)
    print(f"Saved round3 comparison to {ProjectPaths.METRICS_DIR / 'round3_comparison.csv'}")
    
    return results_df, feature_selectors, n_features_dict, comparison_df

def run_round4(round2_best_params, round3_selectors, round3_n_features):
    """
    ROUND 4: Feature selected dataset with tuned model functions
    Uses best parameters from Round 2 and feature selectors from Round 3
    """
    print("\n" + "="*60)
    print("ROUND 4: Feature Selected Data + Tuned Parameters")
    print("="*60)
    
    # Get all models
    all_models = get_all_models()
    
    # Store results
    results = []
    
    # Train and evaluate each model
    for model_name, model_info in all_models.items():
        print(f"\nTraining {model_name} (feature selected, tuned parameters)...")
        
        # Load appropriate data
        data = _load_data_for_model(model_info)
        X_train, X_val, X_test, y_train, y_val, y_test = (
            data['X_train'], data['X_val'], data['X_test'],
            data['y_train'], data['y_val'], data['y_test']
        )
        
        # Apply feature selection using selector from Round 3
        if model_name in round3_selectors:
            selector = round3_selectors[model_name]
            X_train_fs = selector.transform(X_train)
            X_val_fs = selector.transform(X_val)
            X_test_fs = selector.transform(X_test)
            n_selected = round3_n_features[model_name]
            print(f"  Using feature selector from Round 3, selected {n_selected} features")
        else:
            # If no selector found, apply new feature selection
            print(f"  No feature selector found, applying new feature selection...")
            X_train_fs, X_val_fs, X_test_fs, n_selected, _ = apply_feature_selection(
                model_name, X_train, X_val, X_test, y_train,
                n_features=ProjectSettings.NUM_FEATURES_TO_SELECT
            )
        
        # Get model with best parameters from Round 2
        model = model_info['model']
        if model_name in round2_best_params:
            # Set the best parameters from Round 2
            best_params = round2_best_params[model_name]
            model.set_params(**best_params)
            print(f"  Using best parameters from Round 2: {best_params}")
        else:
            print(f"  No best parameters found, using default")
        
        # Evaluate model with feature selected data and tuned parameters
        metrics = evaluate_model(model, X_train_fs, X_val_fs, X_test_fs, y_train, y_val, y_test, model_name)
        
        # Store results
        results.append({
            'round': 4,
            'model_name': model_name,
            'data_type': model_info['data_type'],
            'parameters': 'tuned',
            'feature_selection': 'yes',
            'n_selected_features': n_selected,
            **metrics
        })
        
        print(f"  Test Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, Features: {n_selected}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(ProjectPaths.METRICS_DIR / 'round4_results.csv', index=False)
    print(f"Saved round4 results to {ProjectPaths.METRICS_DIR / 'round4_results.csv'}")
    
    return results_df

def run_all_rounds():
    """Run all four rounds of experiments"""
    
    print("="*80)
    print("MODEL EXPERIMENT PIPELINE - ALL FOUR ROUNDS")
    print("="*80)
    
    # Run Round 1: Preprocessed data + Default parameters
    print("\n" + "="*80)
    print("STARTING ROUND 1")
    print("="*80)
    round1_results = run_round1()
    plot_round_results(round1_results, 1, "Preprocessed Data + Default Parameters")
    
    # Run Round 2: Preprocessed data + Tuned parameters
    print("\n" + "="*80)
    print("STARTING ROUND 2")
    print("="*80)
    round2_results, round2_best_params = run_round2()
    plot_round_results(round2_results, 2, "Preprocessed Data + Tuned Parameters")
    
    # Run Round 3: Feature selected data + Default parameters
    print("\n" + "="*80)
    print("STARTING ROUND 3")
    print("="*80)
    round3_results, round3_selectors, round3_n_features, round3_comparison = run_round3()
    plot_round_results(round3_results, 3, "Feature Selected Data + Default Parameters")
    plot_round3_comparison(round3_comparison)
    
    # Run Round 4: Feature selected data + Tuned parameters
    print("\n" + "="*80)
    print("STARTING ROUND 4")
    print("="*80)
    round4_results = run_round4(round2_best_params, round3_selectors, round3_n_features)
    plot_round_results(round4_results, 4, "Feature Selected Data + Tuned Parameters")
    
    # Create final comparison of all rounds
    print("\n" + "="*80)
    print("FINAL COMPARISON OF ALL ROUNDS")
    print("="*80)
    plot_comparison_table(round1_results, round2_results, round3_results, round4_results)
    
    print("\n" + "="*80)
    print("EXPERIMENT PIPELINE COMPLETE!")
    print("="*80)
    print("\nSaved files:")
    print(f"1. {ProjectPaths.METRICS_DIR / 'round1_results.csv'} - Round 1 results")
    print(f"2. {ProjectPaths.METRICS_DIR / 'round2_results.csv'} - Round 2 results")
    print(f"3. {ProjectPaths.METRICS_DIR / 'round3_results.csv'} - Round 3 results")
    print(f"4. {ProjectPaths.METRICS_DIR / 'round4_results.csv'} - Round 4 results")
    print(f"5. {ProjectPaths.METRICS_DIR / 'round3_comparison.csv'} - Round3 vs Round1 comparison")
    print("\nVisualizations saved to:")
    print(f"- {ProjectPaths.FIGURES_DIR} directory")
    
    return {
        'round1': round1_results,
        'round2': round2_results,
        'round3': round3_results,
        'round4': round4_results,
        'round2_best_params': round2_best_params,
        'round3_comparison': round3_comparison
    }