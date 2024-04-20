# OMIDR (Omics Integration for Dimensionality Reduction)

**OMIDR** (Omics Integration for Dimensionality Reduction) is a tool designed for performing feature selection on multi-omics data, mainly for transcriptomic (trans) and microbiomic (micro) data. It helps in identifying the top features from each omic type, aiding in dimensionality reduction and enhancing data analysis efficiency.

## OMIDR Methodology Overview

OMIDR utilizes an ensemble modeling approach combined with feature scoring techniques to perform feature selection on multi-omics data. The methodology can be summarized as follows:

### Model Creation and Hyperparameter Tuning:

- OMIDR creates three models: Support Vector Classifier (SVC), Logistic Regression, and XGBoost.
- Hyperparameter tuning is performed using grid search with fivefold cross-validation to select the best hyperparameters for each model.

### Feature Scoring:

- Each feature is scored three times, once for each model, using the area under the ROC curve (ROC-AUC) metric.
- The weighted information gain (feature importance) is calculated from both Logistic Regression and XGBoost models.
- Additionally, the model coefficient is obtained from the SVC model.

### Normalization:

- Min-max normalization is applied to ensure that all model scores are on the same scale, facilitating fair comparison.

### Weighted Average Score:

- A weighted average of these normalized scores is computed for each feature.

## OMIDR Output Files:

## Omics Top Features Files:
  - `trans_top_features.csv`: Contains the top features from the transcriptomic data along with their importance scores.
  - `micro_top_features.csv`: Includes the top features from the microbiomic data along with their importance scores.

## Top K Features Dataframes:

- `trans_top_k_features_df`: A pandas dataframe containing the top K features selected from the transcriptomic data.
- `micro_top_k_features_df`: A pandas dataframe containing the top K features selected from the microbiomic data.

## Visualization

- **omic_features_visualization.png:** A graphical representation showing the top 10 features from both omic types (transcriptomic and microbiomic) and their respective importance scores.

## Installation and Usage

### 1. Download OMIDR_Selector.py

Please download the `OMIDR_Selector.py` script from the OMIDR repository.
### 2. Naming
- name your transcriptomic features each to start with "Trans_"
- name your microbiomic features each to start with "Micro_"


## Demo
In this demo, we employ the OMIDR method (perform_feature_selection) to select the top 50 significant features (K) for each omic data separately.

   ```python
import pandas as pd
from sklearn.model_selection import train_test_split
from omidr import perform_feature_selection

# Load your multi-omics dataset (e.g., CSV format)
dataset_path = 'path_to_your_dataset.csv'
data = pd.read_csv(dataset_path)

# Prepare your data for feature selection
X = data.drop(['drop your y_label'])
y = data['your y_label name']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform feature selection with OMIDR (adjust parameters as needed)
perform_feature_selection(dataset_path, X_train, X_test, y_train, y_test, k=50)

