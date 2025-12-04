import os
import pandas as pd
from config.paths import ProjectPaths
from config.settings import ProjectSettings
from data import load_data, engineer_features, prepare_non_scaled_data, prepare_scaled_data, save_processed_data
from visualization.eda import generate_eda_figures
from modeling.trainer import run_all_rounds

def process_data(file_path=None):
    """Complete data processing pipeline"""
    print("="*60)
    print("Airline Passenger Satisfaction - Data Processing Pipeline")
    print("="*60)
    
    # Create directories
    ProjectPaths.create_directories()
    
    # Load and process
    raw_data = load_data(file_path)
    processed_data = engineer_features(raw_data)
    
    # Prepare both datasets
    print("\nPreparing datasets...")
    non_scaled_data = prepare_non_scaled_data(processed_data, ProjectSettings.TARGET_COLUMN)
    scaled_data = prepare_scaled_data(processed_data, ProjectSettings.TARGET_COLUMN)
    
    # Compare datasets
    print(f"\nComparison:")
    print(f"Non-scaled data features: {len(non_scaled_data['feature_names'])}")
    print(f"Scaled data features: {len(scaled_data['feature_names'])}")
    
    # Save processed data
    save_processed_data(non_scaled_data, 'non_scaled')
    save_processed_data(scaled_data, 'scaled')
    
    # Save full processed dataset
    processed_data.to_csv(ProjectPaths.FULL_DATASET, index=False)
    print(f"Saved full processed dataset to: {ProjectPaths.FULL_DATASET}")
    
    return {
        'non_scaled_data': non_scaled_data,
        'scaled_data': scaled_data,
        'processed_data': processed_data
    }

def main():
    """Main pipeline execution"""
    print("="*80)
    print("✈️  AIRLINE PASSENGER SATISFACTION ANALYSIS - FULL PIPELINE")
    print("="*80)
    
    # Step 1: Data processing
    print("\n🔄 STEP 1: DATA PROCESSING")
    print("-"*80)
    data_dict = process_data(ProjectPaths.RAW_DATA_FILE)
    
    # Step 2: EDA visualizations
    print("\n📊 STEP 2: EXPLORATORY DATA ANALYSIS")
    print("-"*80)
    generate_eda_figures()
    
    # Step 3: Model experiments
    print("\n🤖 STEP 3: MODEL EXPERIMENTS (4 ROUNDS)")
    print("-"*80)
    results = run_all_rounds()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {ProjectPaths.RESULTS_DIR}")
    print(f"All figures saved to: {ProjectPaths.FIGURES_DIR}")

if __name__ == "__main__":
    main()