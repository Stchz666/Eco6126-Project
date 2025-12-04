"""
Experiment configuration for the Airline Passenger Satisfaction project.
Defines settings for the four rounds of experiments and model categories.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# ===================== EXPERIMENT SETTINGS =====================
@dataclass
class ExperimentSettings:
    """Base experiment settings"""
    random_seed: int = 42
    test_size: float = 0.2
    validation_size: float = 0.1
    cv_folds: int = 5
    scoring_metric: str = "accuracy"
    
# ===================== MODEL CATEGORIZATION =====================
# Scientific categorization of machine learning models
MODEL_CATEGORIES = {
    # Parametric models with linear assumptions
    "LINEAR_MODELS": [
        "LogisticRegression",
        "LinearSVM"
    ],
    
    # Non-parametric tree-based models
    "TREE_BASED_MODELS": [
        "DecisionTree",
        "RandomForest",
        "ExtraTrees"
    ],
    
    # Gradient boosting models
    "GRADIENT_BOOSTING_MODELS": [
        "XGBoost",
        "LightGBM",
        "CatBoost"
    ],
    
    # Adaptive boosting models
    "ADAPTIVE_BOOSTING_MODELS": [
        "AdaBoost"
    ],
    
    # Distance-based models
    "DISTANCE_BASED_MODELS": [
        "KNN"
    ],
    
    # Probabilistic models
    "PROBABILISTIC_MODELS": [
        "GaussianNB"
    ],
    
    # Neural network models
    "NEURAL_NETWORK_MODELS": [
        "MLP",
        "CNN1D"
    ],
    
    # Meta-ensemble models
    "META_ENSEMBLE_MODELS": [
        "Bagging",
        "Voting"
    ]
}

# Alternative classification by learning paradigm
MODEL_CLASSIFICATION = {
    "Supervised_Learning": {
        "Classification": [
            "LogisticRegression",
            "DecisionTree",
            "RandomForest",
            "XGBoost",
            "LightGBM",
            "KNN",
            "GaussianNB",
            "MLP",
            "CNN1D"
        ]
    },
    "Ensemble_Methods": {
        "Bagging": ["RandomForest", "ExtraTrees", "Bagging"],
        "Boosting": ["XGBoost", "LightGBM", "AdaBoost"],
        "Voting": ["Voting"]
    }
}

# ===================== EXPERIMENT ROUNDS CONFIGURATION =====================
@dataclass
class Round1Config:
    """Configuration for Round 1: Preprocessed Data + Default Parameters"""
    name: str = "Round 1: Preprocessed Data + Default Parameters"
    description: str = "Baseline experiment with default model parameters on preprocessed data"
    models_to_include: List[str] = None
    
    def __post_init__(self):
        if self.models_to_include is None:
            # Include all models for baseline
            self.models_to_include = [
                "LogisticRegression",
                "DecisionTree",
                "RandomForest",
                "XGBoost",
                "LightGBM",
                "AdaBoost",
                "KNN",
                "GaussianNB",
                "MLP",
                "Bagging"
            ]
    
    data_config: Dict = None
    
    def __post_init__(self):
        if self.data_config is None:
            self.data_config = {
                "preprocessing_level": "full",
                "feature_selection": "none",
                "feature_scaling": "per_model",  # Each model gets appropriate scaling
                "encoding_method": "per_model"   # Each model gets appropriate encoding
            }

@dataclass
class Round2Config:
    """Configuration for Round 2: Preprocessed Data + Tuned Parameters"""
    name: str = "Round 2: Preprocessed Data + Tuned Parameters"
    description: str = "Hyperparameter tuning on preprocessed data"
    models_to_tune: List[str] = None
    
    def __post_init__(self):
        if self.models_to_tune is None:
            # Focus on key models for tuning to save time
            self.models_to_tune = [
                "LogisticRegression",
                "RandomForest",
                "XGBoost",
                "KNN",
                "MLP"
            ]
    
    tuning_config: Dict = None
    
    def __post_init__(self):
        if self.tuning_config is None:
            self.tuning_config = {
                "method": "grid_search",  # Options: grid_search, random_search, bayesian
                "cv_folds": 3,
                "n_iter": 20,  # For random search
                "scoring": "accuracy",
                "verbose": 0
            }

@dataclass
class Round3Config:
    """Configuration for Round 3: Feature Selected Data + Default Parameters"""
    name: str = "Round 3: Feature Selected Data + Default Parameters"
    description: str = "Default models on feature-selected data"
    
    feature_selection_methods: Dict = None
    
    def __post_init__(self):
        if self.feature_selection_methods is None:
            self.feature_selection_methods = {
                "PCA": {
                    "variance_threshold": 0.95,
                    "apply_to": ["linear", "knn", "neural"]  # Which model categories
                },
                "RF_Importance": {
                    "threshold": "median",
                    "apply_to": ["tree", "boosting"]
                },
                "Correlation": {
                    "threshold": 0.8,
                    "apply_to": ["all"]
                },
                "Univariate": {
                    "k": 20,
                    "apply_to": ["linear", "knn"]
                }
            }
    
    models_to_include: List[str] = None
    
    def __post_init__(self):
        if self.models_to_include is None:
            self.models_to_include = [
                "LogisticRegression",
                "RandomForest",
                "XGBoost",
                "KNN"
            ]

@dataclass
class Round4Config:
    """Configuration for Round 4: Feature Selected Data + Tuned Parameters"""
    name: str = "Round 4: Feature Selected Data + Tuned Parameters"
    description: str = "Combined approach: feature selection + hyperparameter tuning"
    
    best_feature_methods: List[str] = None
    
    def __post_init__(self):
        if self.best_feature_methods is None:
            self.best_feature_methods = [
                "PCA",          # For linear/neural models
                "RF_Importance" # For tree/boosting models
            ]
    
    models_to_tune: List[str] = None
    
    def __post_init__(self):
        if self.models_to_tune is None:
            self.models_to_tune = [
                "LogisticRegression",  # With PCA
                "RandomForest",        # With RF Importance
                "XGBoost"              # With RF Importance
            ]

# ===================== EXPERIMENT EVALUATION METRICS =====================
EVALUATION_METRICS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc",
    "confusion_matrix"
]

CLASSIFICATION_REPORT_METRICS = [
    "precision",
    "recall",
    "f1-score",
    "support"
]

# ===================== DATA TRANSFORMATION SETTINGS =====================
# Settings for different types of data transformations per model category
MODEL_SPECIFIC_TRANSFORMATIONS = {
    "linear": {
        "scaling": "standardization",
        "encoding": "onehot",
        "handle_skewness": True,
        "handle_outliers": True
    },
    "tree": {
        "scaling": "none",
        "encoding": "label",
        "handle_skewness": False,
        "handle_outliers": False
    },
    "knn": {
        "scaling": "standardization",
        "encoding": "onehot",
        "handle_skewness": True,
        "handle_outliers": True
    },
    "neural": {
        "scaling": "standardization",
        "encoding": "onehot",
        "handle_skewness": True,
        "handle_outliers": True,
        "reshape": True  # For CNN
    },
    "bayesian": {
        "scaling": "none",  # GaussianNB assumes normal distribution
        "encoding": "label",
        "handle_skewness": False,
        "handle_outliers": False
    }
}

# ===================== FEATURE ENGINEERING SETTINGS =====================
FEATURE_ENGINEERING_CONFIG = {
    "create_is_delayed": True,
    "delay_threshold": 0,  # Minutes threshold for is_delayed
    "create_service_scores": True,
    "service_columns": [  # List of service-related columns
        'Inflight wifi service',
        'Departure/Arrival time convenient',
        'Ease of Online booking',
        'Gate location',
        'Food and drink',
        'Online boarding',
        'Seat comfort',
        'Inflight entertainment',
        'On-board service',
        'Leg room service',
        'Baggage handling',
        'Checkin service',
        'Inflight service',
        'Cleanliness'
    ],
    "create_delay_categories": True,
    "delay_bins": [-1, 0, 30, 60, float('inf')],
    "delay_labels": ["No Delay", "Short Delay", "Medium Delay", "Long Delay"],
    "create_low_service_flag": True,
    "low_service_threshold": 2
}

# ===================== VISUALIZATION SETTINGS =====================
VISUALIZATION_CONFIG = {
    "style": "seaborn-v0_8-darkgrid",
    "palette": "husl",
    "figure_sizes": {
        "default": (12, 8),
        "comparison": (16, 10),
        "small": (8, 6)
    },
    "font_sizes": {
        "title": 16,
        "axis_labels": 14,
        "ticks": 12,
        "legend": 12
    },
    "colors": {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e",
        "tertiary": "#2ca02c",
        "quaternary": "#d62728"
    }
}

# ===================== NEURAL NETWORK CONFIGURATION =====================
NEURAL_NETWORK_CONFIG = {
    "MLP": {
        "default_architecture": [100, 50],
        "activation": "relu",
        "output_activation": "sigmoid",
        "loss": "binary_crossentropy",
        "optimizer": "adam",
        "metrics": ["accuracy"],
        "callbacks": ["early_stopping", "reduce_lr"],
        "early_stopping_patience": 10,
        "reduce_lr_patience": 5,
        "reduce_lr_factor": 0.5,
        "epochs": 100,
        "batch_size": 32,
        "validation_split": 0.2,
        "verbose": 0
    },
    "CNN1D": {
        "conv_layers": [
            {"filters": 64, "kernel_size": 3, "activation": "relu"},
            {"filters": 128, "kernel_size": 3, "activation": "relu"},
            {"filters": 256, "kernel_size": 3, "activation": "relu"}
        ],
        "pooling": "max",
        "pool_size": 2,
        "dense_layers": [256, 128, 64],
        "dropout_rate": 0.5,
        "output_activation": "sigmoid",
        "loss": "binary_crossentropy",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "metrics": ["accuracy"],
        "callbacks": ["early_stopping", "reduce_lr"],
        "early_stopping_patience": 10,
        "reduce_lr_patience": 5,
        "reduce_lr_factor": 0.5,
        "epochs": 50,
        "batch_size": 32,
        "validation_split": 0.2,
        "verbose": 0
    }
}

# ===================== EXPERIMENT CONSTANTS =====================
TARGET_COLUMN = "satisfaction"
POSITIVE_CLASS = "Satisfied"
NEGATIVE_CLASS = "Neutral/Dissatisfied"

# Feature columns to exclude from modeling
EXCLUDE_COLUMNS = ["id", "Unnamed: 0"]

# Service-related columns
SERVICE_COLUMNS = [
    'Inflight wifi service',
    'Departure/Arrival time convenient',
    'Ease of Online booking',
    'Gate location',
    'Food and drink',
    'Online boarding',
    'Seat comfort',
    'Inflight entertainment',
    'On-board service',
    'Leg room service',
    'Baggage handling',
    'Checkin service',
    'Inflight service',
    'Cleanliness'
]

# Delay-related columns
DELAY_COLUMNS = [
    'Departure Delay in Minutes',
    'Arrival Delay in Minutes'
]

# ===================== EXPERIMENT FLOW CONTROL =====================
EXPERIMENT_FLOW = {
    "round1": {
        "enabled": True,
        "models": "all",
        "data": "preprocessed",
        "parameters": "default"
    },
    "round2": {
        "enabled": True,
        "models": "selected",  # Focus on key models
        "data": "preprocessed",
        "parameters": "tuned"
    },
    "round3": {
        "enabled": True,
        "models": "selected",
        "data": "feature_selected",
        "parameters": "default"
    },
    "round4": {
        "enabled": True,
        "models": "selected",
        "data": "feature_selected",
        "parameters": "tuned"
    }
}

# ===================== LOGGING CONFIGURATION =====================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "file_logging": True,
    "log_file": "experiment.log",
    "console_logging": True
}

# ===================== RANDOM SEEDS =====================
RANDOM_SEEDS = {
    "numpy": 42,
    "sklearn": 42,
    "tensorflow": 42,
    "xgboost": 42,
    "lightgbm": 42
}