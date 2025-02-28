import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

# Load Data
test_path = "C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/data/Testing.csv"
testing = pd.read_csv(test_path)

# Load Models
rf = joblib.load("C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/models/saved models/random_forest_model.pkl")
svm = joblib.load("C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/models/saved models/svm_model.pkl")
scaler = joblib.load("C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/models/saved models/scaler.pkl")
le = joblib.load("C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/models/saved models/label_encoder.pkl")

# Extract features
cols = testing.columns[:-1]
X_test_final = testing[cols]
y_test_final = le.transform(testing['prognosis'])

# Scale Data for SVM
X_test_final_scaled = scaler.transform(X_test_final)

# Predictions
rf_test_preds = rf.predict(X_test_final)
svm_test_preds = svm.predict(X_test_final_scaled)

# Regular Test Set Evaluation
print(f"Random Forest Test Accuracy: {accuracy_score(y_test_final, rf_test_preds):.4f}")
print("Random Forest Classification Report:\n", classification_report(y_test_final, rf_test_preds))

print(f"SVM Test Accuracy: {accuracy_score(y_test_final, svm_test_preds):.4f}")
print("SVM Classification Report:\n", classification_report(y_test_final, svm_test_preds))

# Load the full training data for cross-validation
train_path = "C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/data/symptoms.csv"
training = pd.read_csv(train_path)
X_full = training[cols]
y_full = le.transform(training['prognosis'])

# Cross-validation on full dataset
rf_cv = cross_val_score(rf, X_full, y_full, cv=5)
svm_cv = cross_val_score(svm, scaler.transform(X_full), y_full, cv=5)

print(f"Random Forest Cross-validation Accuracy: {rf_cv.mean():.4f}")
print(f"SVM Cross-validation Accuracy: {svm_cv.mean():.4f}")
