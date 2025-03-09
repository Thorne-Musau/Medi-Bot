import spacy
import json
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load or download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy's English model and medical entity recognition
nlp = spacy.load("en_core_web_sm")  # Consider using a medical NER model like scispacy

# Load symptom list from dataset
with open("data/symptom_list.json", "r") as file:
    SYMPTOM_LIST = json.load(file)  # JSON file containing known symptoms

# Load intent classification model
INTENT_MODEL = joblib.load("models/intent_classifier.pkl")
VECTORIZER = joblib.load("models/vectorizer.pkl")

def preprocess_text(text):
    """Tokenization, stopword removal, and lemmatization."""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stopwords & non-alphabetic words
    doc = nlp(" ".join(tokens))
    return " ".join([token.lemma_ for token in doc])  # Lemmatization

def extract_symptoms(text):
    """Extract symptoms from user input using rule-based and NLP methods."""
    text = preprocess_text(text)
    detected_symptoms = []

    # Rule-based matching
    for symptom in SYMPTOM_LIST:
        if re.search(rf"\b{re.escape(symptom.lower())}\b", text):
            detected_symptoms.append(symptom)

    # Named Entity Recognition (NER) for medical terms
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["DISEASE", "SYMPTOM"]:  # Adjust based on the NER model used
            detected_symptoms.append(ent.text)

    return list(set(detected_symptoms))  # Remove duplicates

def classify_intent(text):
    """Classifies user intent (e.g., symptom inquiry, diagnosis request)."""
    processed_text = preprocess_text(text)
    vectorized_text = VECTORIZER.transform([processed_text])
    intent = INTENT_MODEL.predict(vectorized_text)[0]
    return intent

if __name__ == "__main__":
    # Example usage
    user_input = input("Enter your symptoms or query: ")
    symptoms = extract_symptoms(user_input)
    intent = classify_intent(user_input)

    print(f"Extracted Symptoms: {symptoms}")
    print(f"Detected Intent: {intent}")
