import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

def apply_feature_selection(model_name, X_train, X_val, X_test, y_train, n_features=None):
    """
    Apply feature selection based on model type
    Returns selected datasets and number of features selected
    """
    n_original_features = X_train.shape[1]
    
    # Determine number of features to select if not specified
    if n_features is None:
        n_features = min(30, n_original_features // 2)
    
    print(f"  Original features: {n_original_features}, Selecting: {n_features}")
    
    # Different feature selection methods for different models
    if model_name in ['LogisticRegression', 'GaussianNB']:
        # For linear models, use SelectKBest with f_classif
        selector = SelectKBest(score_func=f_classif, k=n_features)
    elif model_name in ['DecisionTree', 'RandomForest', 'ExtraTrees', 'XGBoost', 'LightGBM']:
        # For tree-based models, use feature importance from the model
        if model_name == 'DecisionTree':
            estimator = DecisionTreeClassifier(random_state=42)
        elif model_name == 'RandomForest':
            estimator = RandomForestClassifier(random_state=42, n_estimators=50)
        elif model_name == 'ExtraTrees':
            estimator = ExtraTreesClassifier(random_state=42, n_estimators=50)
        elif model_name == 'XGBoost':
            estimator = XGBClassifier(random_state=42, n_estimators=50, eval_metric='logloss', use_label_encoder=False)
        elif model_name == 'LightGBM':
            estimator = LGBMClassifier(random_state=42, n_estimators=50)
        
        selector = SelectFromModel(estimator, max_features=n_features)
    elif model_name in ['KNN', 'MLP', 'AdaBoost', 'Bagging']:
        # For other models, use RFE with appropriate estimator
        if model_name == 'KNN':
            estimator = KNeighborsClassifier()
        elif model_name == 'MLP':
            estimator = MLPClassifier(random_state=42, max_iter=500)
        elif model_name == 'AdaBoost':
            estimator = AdaBoostClassifier(random_state=42, n_estimators=50)
        elif model_name == 'Bagging':
            estimator = BaggingClassifier(random_state=42, n_estimators=10)
        
        selector = RFE(estimator, n_features_to_select=n_features, step=5)
    else:
        # Default to SelectKBest
        selector = SelectKBest(score_func=f_classif, k=n_features)
    
    # Fit and transform
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    selected_features = np.sum(selector.get_support()) if hasattr(selector, 'get_support') else n_features
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features, selector