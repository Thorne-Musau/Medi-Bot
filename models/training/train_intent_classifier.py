import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

# Load the dataset
data_path = "C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/data/intents.csv"  # Adjust path if needed
df = pd.read_csv(data_path, encoding ='latin-1')

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r"\W", " ", text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return " ".join(tokens)

# Apply preprocessing
df["Processed Query"] = df["User Query"].apply(preprocess_text)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(df["Processed Query"], df["Intent"], test_size=0.2, random_state=42)

# Vectorization (TF-IDF with bigrams and trigrams)
vectorizer = TfidfVectorizer(ngram_range=(1,3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Intent Classification Model (Logistic Regression & Random Forest)
logistic_classifier = LogisticRegression(max_iter=300, random_state=42)
random_forest_classifier = RandomForestClassifier(n_estimators=200, random_state=42)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(random_forest_classifier, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)
best_rf_classifier = grid_search.best_estimator_

# Train models
logistic_classifier.fit(X_train_tfidf, y_train)

# Predictions
logistic_pred = logistic_classifier.predict(X_test_tfidf)
best_rf_pred = best_rf_classifier.predict(X_test_tfidf)

# Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, logistic_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, best_rf_pred))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, logistic_pred))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, best_rf_pred))


# Save models
joblib.dump(vectorizer, "C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/models/saved models/vectorizer.pkl")
joblib.dump(logistic_classifier, "C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/models/saved models/intent_classifier.pkl")
joblib.dump(best_rf_classifier, "C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/models/saved models/random_forest_classifier.pkl")
print("✅ Model training complete. Vectorizer and classifiers saved!")
