# run_pipeline.py
"""
Flexible pipeline execution script for airline passenger satisfaction analysis
"""

import argparse
import time
import pandas as pd
from pathlib import Path
from src.config.settings import ProjectSettings
from src.data.loader import load_data
from src.data.engineer import engineer_features
from src.data.processor import prepare_non_scaled_data, prepare_scaled_data
from src.data.saver import save_processed_data
from src.modeling.registry import get_all_models
from src.modeling.evaluator import evaluate_model
from src.modeling.selector import apply_feature_selection
from src.modeling.trainer import (
    run_round1, run_round2, 
    run_round3, run_round4
)
from src.visualization.eda import generate_eda_figures
from src.visualization.results import (
    plot_round_results, 
    plot_round3_comparison, 
    plot_comparison_table
)

def process_data(quick_mode=False):
    """Process data with optional quick mode for testing"""
    # Create necessary directories
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading raw data...")
    raw_data = load_data()
    
    # Quick mode: use only 10% of data
    if quick_mode:
        raw_data = raw_data.sample(frac=0.1, random_state=Settings.RANDOM_STATE)
        print(f"⚠️  Quick mode enabled - using {len(raw_data)} samples ({len(raw_data)/len(load_data())*100:.1f}%)")
    
    # Feature engineering
    print("Engineering features...")
    processed_data = engineer_features(raw_data)
    
    # Prepare datasets
    print("Preparing non-scaled data for tree models...")
    non_scaled_data = prepare_non_scaled_data(processed_data)
    print("Preparing scaled data for linear models...")
    scaled_data = prepare_scaled_data(processed_data)
    
    # Save processed data
    print("Saving processed data...")
    save_processed_data(non_scaled_data, 'non_scaled')
    save_processed_data(scaled_data, 'scaled')
    
    # Save full dataset
    processed_data.to_csv(Path("data/processed/full_processed_dataset.csv"), index=False)
    
    return {
        'non_scaled_data': non_scaled_data,
        'scaled_data': scaled_data,
        'processed_data': processed_data
    }

def main():
    """Main execution function with selective pipeline stages"""
    parser = argparse.ArgumentParser(description='Run airline satisfaction analysis pipeline selectively')
    
    # Pipeline stage selection
    parser.add_argument('--data', action='store_true', help='Run only data processing')
    parser.add_argument('--eda', action='store_true', help='Generate EDA visualizations only')
    parser.add_argument('--round', type=int, choices=[1, 2, 3, 4], help='Run specific experiment round')
    parser.add_argument('--model', type=str, help='Run specific model(s) only (comma-separated)')
    
    # Execution options
    parser.add_argument('--quick', action='store_true', help='Use 10% of data for quick testing')
    parser.add_argument('--no-save', action='store_true', help='Skip saving results to disk')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.data, args.eda, args.round]):
        print("Please specify at least one stage to run:")
        print("  --data    : Run data processing only")
        print("  --eda     : Generate EDA visualizations only")
        print("  --round N : Run experiment round N (1-4)")
        print("\nExample usage:")
        print("  python run_pipeline.py --data")
        print("  python run_pipeline.py --round 1 --model RandomForest,XGBoost --quick")
        return
    
    start_time = time.time()
    print(f"{'='*60}")
    print(f"Airline Passenger Satisfaction Analysis Pipeline")
    print(f"{'='*60}")
    
    # 1. Data processing stage
    if args.data:
        print(f"\n[STAGE 1] Processing data...")
        start_stage = time.time()
        process_data(quick_mode=args.quick)
        print(f"✓ Data processing completed in {time.time() - start_stage:.2f}s")
    
    # 2. EDA visualization stage
    if args.eda:
        print(f"\n[STAGE 2] Generating EDA visualizations...")
        start_stage = time.time()
        generate_eda_figures()
        print(f"✓ EDA visualizations completed in {time.time() - start_stage:.2f}s")
    
    # 3. Modeling experiments stage
    if args.round:
        print(f"\n[STAGE 3] Running experiment round {args.round}...")
        start_stage = time.time()
        
        # Parse model names if provided
        model_names = None
        if args.model:
            model_names = [m.strip() for m in args.model.split(',')]
            print(f"  • Models: {', '.join(model_names)}")
        
        # Select appropriate round function
        round_functions = {
            1: run_round1,
            2: run_round2,
            3: run_round3,
            4: run_round4
        }
        
        # Execute the round
        results = round_functions[args.round](
            model_names=model_names,
            quick_mode=args.quick
        )
        
        # Save and visualize results
        if not args.no_save:
            output_path = Path("results/metrics") / f"round{args.round}_results.csv"
            results.to_csv(output_path, index=False)
            print(f"✓ Results saved to {output_path}")
            
            # Generate visualizations for this round
            plot_round_results(results, args.round, f"Round {args.round} Results")
        
        print(f"✓ Round {args.round} completed in {time.time() - start_stage:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Pipeline completed successfully in {total_time:.2f} seconds")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()