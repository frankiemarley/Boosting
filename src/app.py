import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# Load the dataset
data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv')

# Replace 0 values with NaN for specific columns
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[zero_columns] = data[zero_columns].replace(0, np.nan)

# Impute missing values with the median strategy
imputer = SimpleImputer(strategy='median')
data[zero_columns] = imputer.fit_transform(data[zero_columns])

# Split the dataset into features and target
X = data.drop(columns='Outcome')
y = data['Outcome']

# Create a holdout set
X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the feature set
scaler = StandardScaler()
X_train_val_scaled = scaler.fit_transform(X_train_val)
X_holdout_scaled = scaler.transform(X_holdout)

# Apply SMOTE to handle class imbalance (only on training data)
smote = SMOTE(random_state=42, sampling_strategy=0.6)
X_resampled, y_resampled = smote.fit_resample(X_train_val_scaled, y_train_val)

# Function to print feature importances
def print_feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\n{title} Feature Importances:")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

# Random Forest with hyperparameter tuning
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=rf_params, n_iter=50, cv=StratifiedKFold(n_splits=5), random_state=42, n_jobs=-1)
rf_random.fit(X_resampled, y_resampled)

print("Best parameters for Random Forest:", rf_random.best_params_)
print_feature_importance(rf_random.best_estimator_, X.columns, "Random Forest")

# Gradient Boosting with hyperparameter tuning
gb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gb = GradientBoostingClassifier(random_state=42)
gb_random = RandomizedSearchCV(estimator=gb, param_distributions=gb_params, n_iter=50, cv=StratifiedKFold(n_splits=5), random_state=42, n_jobs=-1)
gb_random.fit(X_resampled, y_resampled)

print("Best parameters for Gradient Boosting:", gb_random.best_params_)
print_feature_importance(gb_random.best_estimator_, X.columns, "Gradient Boosting")

# XGBoost with hyperparameter tuning
xgb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5]
}

xgb = XGBClassifier(random_state=42)
xgb_random = RandomizedSearchCV(estimator=xgb, param_distributions=xgb_params, n_iter=50, cv=StratifiedKFold(n_splits=5), random_state=42, n_jobs=-1)
xgb_random.fit(X_resampled, y_resampled)

print("Best parameters for XGBoost:", xgb_random.best_params_)
print_feature_importance(xgb_random.best_estimator_, X.columns, "XGBoost")

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('rf', rf_random.best_estimator_), 
                ('gb', gb_random.best_estimator_),
                ('xgb', xgb_random.best_estimator_)],
    voting='soft'
)
voting_clf.fit(X_resampled, y_resampled)

# Function to plot confusion matrices
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()  # Changed to show the plot instead of saving

# Evaluate all models on the holdout set
models = {
    'Random Forest': rf_random.best_estimator_,
    'Gradient Boosting': gb_random.best_estimator_,
    'XGBoost': xgb_random.best_estimator_,
    'Voting Classifier': voting_clf
}

for name, model in models.items():
    y_pred = model.predict(X_holdout_scaled)
    print(f"\n{name} Accuracy on Holdout Set: {accuracy_score(y_holdout, y_pred):.4f}")
    print(f"\n{name} Classification Report on Holdout Set:")
    print(classification_report(y_holdout, y_pred))
    plot_confusion_matrix(y_holdout, y_pred, name)

# Print summary of results
print("\nSummary of Model Performances:")
for name, model in models.items():
    y_pred = model.predict(X_holdout_scaled)
    accuracy = accuracy_score(y_holdout, y_pred)
    f1 = classification_report(y_holdout, y_pred, output_dict=True)['1']['f1-score']
    print(f"{name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# Print summary with ROC AUC
print("\nSummary of Model Performances with ROC AUC:")
for name, model in models.items():
    y_pred = model.predict(X_holdout_scaled)
    accuracy = accuracy_score(y_holdout, y_pred)
    f1 = classification_report(y_holdout, y_pred, output_dict=True)['1']['f1-score']
    roc_auc = roc_auc_score(y_holdout, model.predict_proba(X_holdout_scaled)[:, 1])
    print(f"{name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

# Print cross-validation results
def print_cv_results(model, X, y, name):
    cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5))
    print(f"\n{name} Cross-Validation Scores:")
    print(f"Mean CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

for name, model in models.items():
    print_cv_results(model, X_resampled, y_resampled, name)

# Plot feature importances
def plot_feature_importances(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.show()  # Changed to show the plot instead of saving

plot_feature_importances(rf_random.best_estimator_, X.columns, "Random Forest Feature Importances")
plot_feature_importances(gb_random.best_estimator_, X.columns, "Gradient Boosting Feature Importances")
plot_feature_importances(xgb_random.best_estimator_, X.columns, "XGBoost Feature Importances")

import joblib
joblib.dump(rf_random.best_estimator_, 'random_forest_model.pkl')
joblib.dump(gb_random.best_estimator_, 'gradient_boosting_model.pkl')
joblib.dump(xgb_random.best_estimator_, 'xgboost_model.pkl')
joblib.dump(voting_clf, 'voting_classifier_model.pkl')
