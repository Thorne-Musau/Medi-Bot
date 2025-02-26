import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load Data
file_path = "C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/data/symptoms.csv"
test_path = "C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/data/Testing.csv"

training = pd.read_csv(file_path)
testing = pd.read_csv(test_path)

# Extract features and labels
cols = training.columns[:-1]  # All columns except the last one
X = training[cols]
y = training['prognosis']

# Label Encoding for categorical target
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Standardizing features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# SVM Model
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
svm_preds = svm.predict(X_test_scaled)

# Evaluation
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
print("Random Forest Classification Report:\n", classification_report(y_test, rf_preds))

print(f"SVM Accuracy: {accuracy_score(y_test, svm_preds):.4f}")
print("SVM Classification Report:\n", classification_report(y_test, svm_preds))

# Cross-validation
rf_cv = cross_val_score(rf, X_train, y_train, cv=5)
svm_cv = cross_val_score(svm, X_train_scaled, y_train, cv=5)

print(f"Random Forest Cross-validation Accuracy: {rf_cv.mean():.4f}")
print(f"SVM Cross-validation Accuracy: {svm_cv.mean():.4f}")

# Testing Data Evaluation
X_test_final = testing[cols]
y_test_final = le.transform(testing['prognosis'])

X_test_final_scaled = scaler.transform(X_test_final)

rf_test_preds = rf.predict(X_test_final)
svm_test_preds = svm.predict(X_test_final_scaled)

print(f"Random Forest Test Accuracy: {accuracy_score(y_test_final, rf_test_preds):.4f}")
print(f"SVM Test Accuracy: {accuracy_score(y_test_final, svm_test_preds):.4f}")
