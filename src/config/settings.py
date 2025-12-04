class ProjectSettings:
    """Global configuration settings"""
    # Target variable
    TARGET_COLUMN = "satisfaction"
    
    # Random seed for reproducibility
    RANDOM_STATE = 42
    
    # Test/validation split ratios
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    # Feature engineering columns
    SERVICE_COLUMNS = [
        'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
        'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 
        'Inflight service', 'Cleanliness'
    ]
    
    # Number of features to select in feature selection
    NUM_FEATURES_TO_SELECT = 30