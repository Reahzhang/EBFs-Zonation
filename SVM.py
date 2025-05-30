import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load your dataset
file_path = 'E:/Data/Humid/高密度/Normalized_Semi_VIF.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# 删除包含缺失值的行
data.dropna(inplace=True)

# Selecting features and target variable
features =['Wssd_min','Sssd_min',
           'Wpre_min','Spre_max',
           'Sevp_max',
           'Wlst','Aspect', 'Slope',  'Soil_pH','Elevation',  'SWC', 'SOC', 'SBD', 'STC', 'Soil_class','Slst']  # List of feature names
X = data[features]  # Features
y = data['SEBF']  # Target variable

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets using scaled features
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

# Create a linear SVM model
svm_model = SVC(C=1, kernel='linear')

# Fit the model with training data
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get the model weights (coefficients)
weights = svm_model.coef_
print("\nModel weights (coefficients):")
print(weights)

# Print the number of support vectors
print("Number of support vectors:", len(svm_model.support_vectors_))
