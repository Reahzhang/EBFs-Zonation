import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load your dataset
file_path = 'E:/Data/Humid/Normalized_EBBF.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# 删除包含缺失值的行
data.dropna(inplace=True)

# Selecting features and target variable
features =['Wssd_min','Sssd_min',
           'Wpre_min','Spre_max',
           'Sevp_max',
           'Wlst','Aspect', 'Slope',  'Soil_pH','Elevation',  'SWC', 'SOC', 'SBD', 'STC', 'Soil_class','Slst']  # List of feature names
X = data[features]  # Features
y = data['EBF']  # Target variable

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets using scaled features
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [100, 200],  # Reduced number of options for trees
    'max_depth': [None, 20],  # Reduced options for maximum depth of trees
    'min_samples_split': [2, 10],  # Reduced options for minimum samples to split a node
    'min_samples_leaf': [1, 4],  # Reduced options for minimum samples at a leaf node
    'max_features': ['sqrt']  # Limited method of feature selection, only one option retained
}

# Create a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=0)

# Create a GridSearchCV object with 5-fold cross-validation
grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1)  # Use all available CPU cores

# Fit the model with training data and find the best parameters
grid_search.fit(X_train, y_train)

# Get the best parameters from grid search
best_params = grid_search.best_params_
print(f"best_params : {best_params }")
# Create a Random Forest Classifier with the best parameters from grid search
best_rf_model = RandomForestClassifier(random_state=0, **best_params)

# Fit the model with training data
best_rf_model.fit(X_train, y_train)

# Get the feature importances
feature_importances = best_rf_model.feature_importances_

# Print the feature importances
print("Feature Importances:")
for feature, importance in zip(features, feature_importances):
    print(f"{feature}: {importance:.4f}")

# Make predictions on the test set
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

