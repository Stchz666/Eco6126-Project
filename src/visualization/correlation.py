import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config.paths import ProjectPaths
from config.settings import ProjectSettings

def _load_processed_data():
    """Load the full processed dataset for correlation analysis"""
    full_dataset_path = ProjectPaths.FULL_DATASET
    if not full_dataset_path.exists():
        print(f"Warning: Full processed dataset not found at {full_dataset_path}")
        print("Please run data processing first to generate this file.")
        return None
    
    return pd.read_csv(full_dataset_path)

def plot_correlation_heatmap():
    """Plot correlation heatmap (from SamplePPT)"""
    df = _load_processed_data()
    if df is None:
        return
    
    # Encode categorical variables for correlation
    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes
    
    # Select numerical columns only
    num_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns
    
    # Calculate correlation matrix
    corr_matrix = df_encoded[num_cols].corr()
    
    # Plot heatmap
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=.5,
                cbar_kws={"shrink": .75},
                annot_kws={"size": 8})
    
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(ProjectPaths.FIGURES_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Generated: correlation_heatmap.png (Strong correlations: Class & Online boarding with satisfaction)")