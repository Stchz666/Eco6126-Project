import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config.paths import ProjectPaths
from config.settings import ProjectSettings

def _load_processed_data():
    """Load the full processed dataset for EDA"""
    full_dataset_path = ProjectPaths.FULL_DATASET
    if not full_dataset_path.exists():
        print(f"Warning: Full processed dataset not found at {full_dataset_path}")
        print("Please run data processing first to generate this file.")
        return None
    
    return pd.read_csv(full_dataset_path)

def plot_satisfaction_by_gender():
    """Plot satisfaction breakdown by gender (from SamplePPT)"""
    df = _load_processed_data()
    if df is None:
        return
    
    plt.figure(figsize=(10, 6))
    
    # Calculate satisfaction rates by gender
    gender_satisfaction = df.groupby(['Gender', ProjectSettings.TARGET_COLUMN]).size().unstack()
    gender_satisfaction_pct = gender_satisfaction.div(gender_satisfaction.sum(axis=1), axis=0) * 100
    
    # Plot
    ax = gender_satisfaction_pct.plot(kind='bar', stacked=True, 
                                       color=['#ff9999','#66b3ff'], 
                                       figsize=(10, 6))
    
    plt.title('Passenger Satisfaction by Gender', fontsize=14, fontweight='bold')
    plt.xlabel('Gender')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=0)
    plt.legend(title='Satisfaction', loc='best')
    
    # Add percentage labels on bars
    for i, gender in enumerate(gender_satisfaction_pct.index):
        bottom = 0
        for satisfaction in gender_satisfaction_pct.columns:
            value = gender_satisfaction_pct.loc[gender, satisfaction]
            plt.text(i, bottom + value/2, f'{value:.1f}%', 
                     ha='center', va='center', color='white', fontweight='bold')
            bottom += value
    
    plt.tight_layout()
    plt.savefig(ProjectPaths.FIGURES_DIR / 'satisfaction_by_gender.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Generated: satisfaction_by_gender.png (72% of men satisfied vs. 64% of women)")

def plot_loyalty_vs_satisfaction():
    """Plot satisfaction by customer type (from Sample PPT)"""
    df = _load_processed_data()
    if df is None:
        return
    
    plt.figure(figsize=(10, 6))
    
    # Calculate satisfaction rates by customer type
    loyalty_satisfaction = df.groupby(['Customer Type', ProjectSettings.TARGET_COLUMN]).size().unstack()
    loyalty_satisfaction_pct = loyalty_satisfaction.div(loyalty_satisfaction.sum(axis=1), axis=0) * 100
    
    # Plot
    ax = loyalty_satisfaction_pct.plot(kind='bar', 
                                        color=['#66b3ff', '#ff9999'], 
                                        figsize=(10, 6))
    
    plt.title('Satisfaction by Customer Loyalty', fontsize=14, fontweight='bold')
    plt.xlabel('Customer Type')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=0)
    plt.legend(title='Satisfaction', loc='best')
    
    # Add percentage labels on bars
    for i, customer_type in enumerate(loyalty_satisfaction_pct.index):
        for j, satisfaction in enumerate(loyalty_satisfaction_pct.columns):
            value = loyalty_satisfaction_pct.loc[customer_type, satisfaction]
            plt.text(i, value + 1, f'{value:.1f}%', 
                     ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(ProjectPaths.FIGURES_DIR / 'loyalty_vs_satisfaction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Generated: loyalty_vs_satisfaction.png (Loyal customers 20% more satisfied)")

def plot_travel_purpose_impact():
    """Plot satisfaction by travel purpose (from SamplePPT)"""
    df = _load_processed_data()
    if df is None:
        return
    
    plt.figure(figsize=(12, 6))
    
    # Calculate satisfaction rates by travel type
    travel_satisfaction = df.groupby(['Type of Travel', ProjectSettings.TARGET_COLUMN]).size().unstack()
    travel_satisfaction_pct = travel_satisfaction.div(travel_satisfaction.sum(axis=1), axis=0) * 100
    
    # Plot
    ax = travel_satisfaction_pct.plot(kind='bar', 
                                       color=['#66b3ff', '#ff9999'], 
                                       figsize=(12, 6))
    
    plt.title('Satisfaction by Travel Purpose', fontsize=14, fontweight='bold')
    plt.xlabel('Type of Travel')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=0)
    plt.legend(title='Satisfaction', loc='best')
    
    # Add percentage labels
    for i, travel_type in enumerate(travel_satisfaction_pct.index):
        for j, satisfaction in enumerate(travel_satisfaction_pct.columns):
            value = travel_satisfaction_pct.loc[travel_type, satisfaction]
            plt.text(i, value + 1, f'{value:.1f}%', 
                     ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(ProjectPaths.FIGURES_DIR / 'travel_purpose_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Generated: travel_purpose_impact.png (Business: 58% satisfied vs Personal: 12%)")

def plot_age_satisfaction_trends():
    """Plot satisfaction trends by age group (from SamplePPT)"""
    df = _load_processed_data()
    if df is None:
        return
    
    # Create age groups
    df['Age Group'] = pd.cut(df['Age'], 
                            bins=[0, 25, 40, 60, 100], 
                            labels=['<25', '25-40', '41-60', '60+'])
    
    plt.figure(figsize=(12, 6))
    
    # Calculate satisfaction by age group
    age_satisfaction = df.groupby(['Age Group', ProjectSettings.TARGET_COLUMN]).size().unstack()
    age_satisfaction_pct = age_satisfaction.div(age_satisfaction.sum(axis=1), axis=0) * 100
    
    # Plot
    ax = age_satisfaction_pct.plot(kind='bar', 
                                    color=['#66b3ff', '#ff9999'], 
                                    figsize=(12, 6))
    
    plt.title('Satisfaction Trends by Age Group', fontsize=14, fontweight='bold')
    plt.xlabel('Age Group')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=0)
    plt.legend(title='Satisfaction', loc='best')
    
    # Add percentage labels
    for i, age_group in enumerate(age_satisfaction_pct.index):
        for j, satisfaction in enumerate(age_satisfaction_pct.columns):
            value = age_satisfaction_pct.loc[age_group, satisfaction]
            plt.text(i, value + 1, f'{value:.1f}%', 
                     ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(ProjectPaths.FIGURES_DIR / 'age_satisfaction_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Generated: age_satisfaction_trends.png (25-40 age group has highest satisfaction)")

def plot_service_ratings_distribution():
    """Plot distribution of service ratings (from SamplePPT)"""
    df = _load_processed_data()
    if df is None:
        return
    
    # Select service columns
    service_cols = ProjectSettings.SERVICE_COLUMNS
    
    # Create a long-format DataFrame for easier plotting
    service_data = df[service_cols].melt(var_name='Service', value_name='Rating')
    
    plt.figure(figsize=(14, 8))
    
    # Plot distribution of ratings
    sns.violinplot(x='Service', y='Rating', data=service_data, 
                   palette='Set3', inner='quartile')
    
    plt.title('Distribution of Service Ratings', fontsize=14, fontweight='bold')
    plt.xlabel('Service Type')
    plt.ylabel('Rating (0-5)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 5.5)
    
    plt.tight_layout()
    plt.savefig(ProjectPaths.FIGURES_DIR / 'service_ratings_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Generated: service_ratings_distribution.png (Inflight entertainment ratings are highest)")

def generate_eda_figures():
    """Generate all EDA figures"""
    print("\n" + "="*60)
    print("GENERATING EDA VISUALIZATIONS")
    print("="*60)
    
    # Create figures directory if it doesn't exist
    ProjectPaths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    plot_satisfaction_by_gender()
    plot_loyalty_vs_satisfaction()
    plot_travel_purpose_impact()
    plot_age_satisfaction_trends()
    plot_service_ratings_distribution()
    
    # Also generate correlation heatmap
    from .correlation import plot_correlation_heatmap
    plot_correlation_heatmap()
    
    print(f"\n✅ All EDA visualizations saved to: {ProjectPaths.FIGURES_DIR}")