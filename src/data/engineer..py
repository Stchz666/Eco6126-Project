import numpy as np
import pandas as pd
from src.config.settings import ProjectSettings

def engineer_features(data):
    """Create business-oriented features"""
    df = data.copy().dropna()
    
    # Create new features based on SamplePPT insights
    df['Total_Service_Score'] = df[ProjectSettings.SERVICE_COLUMNS].sum(axis=1)
    df['Avg_Service_Score'] = df[ProjectSettings.SERVICE_COLUMNS].mean(axis=1)
    df['Is_Delayed'] = (df['Departure Delay in Minutes'] > 0).astype(int)
    df['Departure_Delay_Log'] = np.log1p(df['Departure Delay in Minutes'])
    df['Arrival_Delay_Log'] = np.log1p(df['Arrival Delay in Minutes'])
    
    # Additional features inspired by SamplePPT
    df['Delay_Difference'] = np.abs(df['Departure Delay in Minutes'] - df['Arrival Delay in Minutes'])
    df['Flight_Distance_Category'] = pd.cut(
        df['Flight Distance'], 
        bins=[0, 500, 1500, 2500, float('inf')],
        labels=['Short', 'Medium', 'Long', 'Very Long']
    )
    
    print(f"Engineered features: Total_Service_Score, Avg_Service_Score, Is_Delayed, Departure_Delay_Log, Arrival_Delay_Log, Delay_Difference, Flight_Distance_Category")
    
    return df