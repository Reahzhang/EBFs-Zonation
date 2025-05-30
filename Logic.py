import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load your dataset
# Load your dataset
file_path = 'E:/Data/Humid/高密度/Normalized_Humid_VIF.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# 删除包含缺失值的行
data.dropna(inplace=True)

# Selecting features and target variable
features =['Wssd_max',
           'Wpre_max','Spre_min',
           'Sevp_min',
           'Wlst','Aspect', 'Slope',  'Soil_pH','Elevation',  'SWC', 'SOC', 'SBD', 'STC', 'Soil_class','Slst']  # List of feature names
X = data[features]  # Features
y = data['HEBF']  # Target variable


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Creating a pipeline with imputer, scaler, and logistic regression model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Imputer to fill NaN values
    ('scaler', StandardScaler()),                # Scaler to standardize features
    ('logistic', LogisticRegression(max_iter=1000))
])

# 定义要尝试的C值的范围
param_grid = {'logistic__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}

# 使用StratifiedKFold进行交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# 创建GridSearchCV对象
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy')

# 拟合模型并寻找最佳参数C
grid_search.fit(X_train, y_train)

# 打印最佳参数C和对应的交叉验证分数
best_C = grid_search.best_params_['logistic__C']
best_score = grid_search.best_score_
print(f"Best C: {best_C}")
print(f"Best Cross-Validation Score: {best_score}")

# Making predictions on the test set using the best model
y_pred = grid_search.predict(X_test)

# Evaluating the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Printing the weights of each feature
best_model = grid_search.best_estimator_.named_steps['logistic']
weights = best_model.coef_[0]
print("\nFeature Weights:")
for feature, weight in zip(features, weights):
    print(f"{feature}: {weight}")
