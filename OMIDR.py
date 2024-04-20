import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_feature_importances(model, params, X, y):
    model.set_params(**params)
    model.fit(X, y)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])
    return importances

def perform_feature_selection(dataset_path, X_train, X_test, y_train, y_test, k):
    data = pd.read_csv(dataset_path)
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

    feature_importances = np.zeros(X_train.shape[1])
    weights = []
    
    for name, model, params in model_pool:
        search = GridSearchCV(model, params, scoring='roc_auc', cv=5)
        search.fit(X_train, y_train)
        best_params = search.best_params_
        best_score = search.best_score_
        print(f"Best parameters for {name}: {best_params} with score: {best_score}")
        
        weights.append(best_score)
        importances = get_feature_importances(model, best_params, X_train, y_train)
        feature_importances += importances * best_score  

    total_weight = sum(weights)
    weighted_importances = feature_importances / total_weight
    feature_importances_df = pd.DataFrame(weighted_importances, index=X_train.columns, columns=['OMIDR'])
    
    normalized_importances = feature_importances_df / feature_importances_df.max()
    trans_features = normalized_importances[normalized_importances.index.str.startswith('Trans_')].nlargest(k, 'OMIDR')
    micro_features = normalized_importances[normalized_importances.index.str.startswith('Micro_')].nlargest(k, 'OMIDR')
    
    trans_features.to_csv(f'OMIDR_Trans_top{k}.csv')
    micro_features.to_csv(f'OMIDR_Micro_top{k}.csv')

    # Separate extraction and visualization for Trans and Micro features
    for feature_group, features_df in [('Trans', trans_features), ('Micro', micro_features)]:
        # Extract top feature values from the data
        feature_values_df = data.loc[:, features_df.index]
        feature_values_df.to_csv(f'OMIDR_{feature_group}_Feature_values_top{k}.csv')

        # Visualization of feature importances
        plt.figure(figsize=(10, 6))
        features_df.plot(kind='barh', legend=None)
        plt.title(f'Top {feature_group} Feature Importances')
        plt.xlabel('Normalized Importance')
        plt.ylabel('Features')
        plt.savefig(f'OMIDR_{feature_group}_Feature_importances_top{k}.png')
        plt.show()

    print("Feature selection, saving, and visualization complete.")

