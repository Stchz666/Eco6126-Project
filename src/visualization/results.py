import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.table import Table
from config.paths import ProjectPaths

def plot_round_results(results_df, round_num, title_suffix):
    """Plot results for a specific round"""
    
    # Filter results for this round
    round_results = results_df[results_df['round'] == round_num]
    
    # Prepare data
    models = round_results['model_name'].tolist()
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_curve', 'training_time']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(metrics[:6]):  # Only plot first 6 metrics
        ax = axes[i]
        values = round_results[metric].values
        
        # Skip if all values are NaN
        if all(np.isnan(v) for v in values):
            ax.text(0.5, 0.5, f'No {metric} data', 
                   ha='center', va='center', fontsize=12)
            ax.set_title(metric.replace('_', ' ').title())
            continue
        
        # Create bar chart
        bars = ax.bar(models, values, color=plt.cm.Set3(np.arange(len(models))))
        ax.set_title(metric.replace('_', ' ').title(), fontsize=14, fontweight='bold')
        ax.set_ylabel('Score' if metric != 'training_time' else 'Seconds')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Round {round_num}: {title_suffix}', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(ProjectPaths.FIGURES_DIR / f'round{round_num}_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary table
    print(f"\nRound {round_num} Results Summary:")
    print("-" * 80)
    
    # Select relevant columns for display
    display_cols = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'auc_curve', 'training_time']
    if 'n_selected_features' in round_results.columns:
        display_cols.append('n_selected_features')
    
    display_df = round_results[display_cols].copy()
    display_df = display_df.set_index('model_name')
    print(display_df.round(4))
    
    return display_df

def plot_round3_comparison(comparison_df):
    """Plot Round 1 vs Round 3 comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy comparison
    ax1 = axes[0, 0]
    x = np.arange(len(comparison_df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, comparison_df['round1_accuracy'], width, label='Round 1', color='skyblue')
    bars2 = ax1.bar(x + width/2, comparison_df['round3_accuracy'], width, label='Round 3', color='lightgreen')
    
    ax1.set_title('Accuracy: Round 1 vs Round 3', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df['model_name'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: F1 Score comparison
    ax2 = axes[0, 1]
    bars3 = ax2.bar(x - width/2, comparison_df['round1_f1'], width, label='Round 1', color='skyblue')
    bars4 = ax2.bar(x + width/2, comparison_df['round3_f1'], width, label='Round 3', color='lightgreen')
    
    ax2.set_title('F1 Score: Round 1 vs Round 3', fontsize=14, fontweight='bold')
    ax2.set_ylabel('F1 Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(comparison_df['model_name'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy change
    ax3 = axes[1, 0]
    colors = ['green' if x > 0 else 'red' for x in comparison_df['accuracy_change']]
    bars5 = ax3.bar(comparison_df['model_name'], comparison_df['accuracy_change'], color=colors)
    
    ax3.set_title('Accuracy Change (Round 3 - Round 1)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Accuracy Change')
    ax3.set_xticklabels(comparison_df['model_name'], rotation=45, ha='right')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Feature reduction
    ax4 = axes[1, 1]
    bars6 = ax4.bar(x - width/2, comparison_df['n_features_round1'], width, label='Round 1', color='skyblue')
    bars7 = ax4.bar(x + width/2, comparison_df['n_features_round3'], width, label='Round 3', color='lightgreen')
    
    ax4.set_title('Number of Features: Round 1 vs Round 3', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Number of Features')
    ax4.set_xticks(x)
    ax4.set_xticklabels(comparison_df['model_name'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Round 3 vs Round 1: Feature Selection Impact Analysis', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(ProjectPaths.FIGURES_DIR / 'round3_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print comparison table
    print("\nRound 3 vs Round 1 Comparison Table:")
    print("-" * 80)
    print(comparison_df.round(4))

def plot_comparison_table(round1_results, round2_results, round3_results, round4_results):
    """Create comparison table of all rounds"""
    
    # Combine all results
    all_results = pd.concat([round1_results, round2_results, round3_results, round4_results])
    
    # Create pivot table for comparison
    comparison = all_results.pivot_table(
        index='model_name',
        columns='round',
        values=['accuracy', 'f1_score', 'auc_curve', 'training_time'],
        aggfunc='first'
    )
    
    # Flatten column names
    comparison.columns = [f'{metric}_round{round}' for metric, round in comparison.columns]
    comparison = comparison.round(4)
    
    # Save to CSV
    comparison.to_csv(ProjectPaths.METRICS_DIR / 'all_rounds_comparison.csv')
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Model'] + [col.replace('_', ' ').title() for col in comparison.columns]
    
    for model in comparison.index:
        row = [model] + [comparison.loc[model, col] for col in comparison.columns]
        table_data.append(row)
    
    # Create the table
    table = ax.table(cellText=table_data, colLabels=headers, 
                     loc='center', cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color the header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#f5f5f5')
    
    plt.title('All Rounds Comparison: Model Performance Across Experiments', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(ProjectPaths.FIGURES_DIR / 'all_rounds_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison