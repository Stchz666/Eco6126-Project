from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, 
    BaggingClassifier, ExtraTreesClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

def get_all_models():
    """Get all 11 models with default parameters"""
    models = {
        # Linear models (use scaled data)
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'data_type': 'scaled',
            'param_grid': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        },
        'GaussianNB': {
            'model': GaussianNB(),
            'data_type': 'scaled',
            'param_grid': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'data_type': 'scaled',
            'param_grid': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'MLP': {
            'model': MLPClassifier(random_state=42, max_iter=1000),
            'data_type': 'scaled',
            'param_grid': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        },
        # Tree-based models (use non-scaled data)
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=42),
            'data_type': 'non_scaled',
            'param_grid': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'data_type': 'non_scaled',
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'max_features': ['sqrt', 'log2']
            }
        },
        'ExtraTrees': {
            'model': ExtraTreesClassifier(random_state=42),
            'data_type': 'non_scaled',
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'AdaBoost': {
            'model': AdaBoostClassifier(random_state=42),
            'data_type': 'non_scaled',
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0]
            }
        },
        'Bagging': {
            'model': BaggingClassifier(random_state=42),
            'data_type': 'non_scaled',
            'param_grid': {
                'n_estimators': [10, 50, 100],
                'max_samples': [0.5, 0.7, 1.0],
                'max_features': [0.5, 0.7, 1.0]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
            'data_type': 'non_scaled',
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.7, 0.8, 1.0]
            }
        },
        'LightGBM': {
            'model': LGBMClassifier(random_state=42),
            'data_type': 'non_scaled',
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'num_leaves': [31, 63, 127]
            }
        }
    }
    return models