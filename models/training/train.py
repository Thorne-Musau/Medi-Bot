import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import subprocess

save_dir = "C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/models/saved models"
# Load Data
file_path = "C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/data/symptoms.csv"

training = pd.read_csv(file_path)

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

# SVM Model
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)

# Save Models
joblib.dump(rf, os.path.join(save_dir, "random_forest_model.pkl"))
joblib.dump(svm, os.path.join(save_dir, "svm_model.pkl"))
joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
joblib.dump(le, os.path.join(save_dir, "label_encoder.pkl"))


print("Models trained and saved successfully.")
