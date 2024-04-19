import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_feature_importances(model, params, X, y):
    # Set model parameters
    model.set_params(**params)
    # Fit model to the data
    model.fit(X, y)
    # Check if the model has the attribute 'feature_importances_'
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:  # For models like Logistic Regression and SVC with a linear kernel
        importances = np.abs(model.coef_[0])
    return importances

def perform_feature_selection(dataset_path, X_train, X_test, y_train, y_test, k):
    # Load dataset from the provided file path
    data = pd.read_csv(dataset_path)
    
    # Define the models and their parameters to be used in feature selection
    model_pool = [
        ('LogisticRegression', LogisticRegression(max_iter=1000), {
            'C': np.logspace(-4, 4, 10),
            'penalty': ['l2']
        }),
        ('SVC', SVC(kernel='linear', probability=True), {
            'C': np.logspace(-4, 4, 10)
        }),
        ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        })
    ]

    # Concatenate model names for output file naming
    combo_name = '-'.join([name for name, _, _ in model_pool])
    # Initialize an array to store feature importances
    feature_importances = np.zeros(X_train.shape[1])
    # Initialize a list to store model weights based on their ROC AUC scores
    weights = []
    
    for name, model, params in model_pool:
        # Setup GridSearchCV with 5-fold cross-validation
        search = GridSearchCV(model, params, scoring='roc_auc', cv=5)
        search.fit(X_train, y_train)
        best_params = search.best_params_
        best_score = search.best_score_
        print(f"Best parameters for {name}: {best_params} with score: {best_score}")
        
        # Append model score to weights list
        weights.append(best_score)
        # Get feature importances from the model
        importances = get_feature_importances(model, best_params, X_train, y_train)
        # Update total feature importances by adding weighted importances
        feature_importances += importances * best_score  

    # Normalize feature importances by total weight of scores
    total_weight = sum(weights)
    weighted_importances = feature_importances / total_weight

    # Create a DataFrame of weighted importances
    feature_importances_df = pd.DataFrame(weighted_importances, index=X_train.columns, columns=[combo_name])
    
    # Normalize importances by the maximum to scale between 0 and 1
    normalized_importances = feature_importances_df / feature_importances_df.max()
    
    # Select top k features for 'Trans' and 'Micro' prefixed features
    trans_features = normalized_importances[normalized_importances.index.str.startswith('Trans_')].nlargest(k, combo_name)
    micro_features = normalized_importances[normalized_importances.index.str.startswith('Micro_')].nlargest(k, combo_name)
    
    # Save the top features to CSV files
    trans_features.to_csv(f'{combo_name}_Trans_top{k}.csv')
    micro_features.to_csv(f'{combo_name}_Micro_top{k}.csv')

    print("Feature selection and saving complete.")
