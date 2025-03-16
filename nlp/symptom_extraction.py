import json
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import process
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load or download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

# Load BioBERT or ClinicalBERT NER model
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"  # Use ClinicalBERT for better medical NER
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

# Create NER pipeline with aggregation strategy
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",  # Use the new aggregation strategy
    device=-1  # Use CPU; set to 0 for GPU
)

# Load symptom list from dataset
with open("C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/data/symptom-list.json", "r") as file:
    SYMPTOM_LIST = json.load(file)

# Normalize symptom list (replace hyphens, strip whitespace, lowercase)
SYMPTOM_LIST = [symptom.replace("-", " ").strip().lower() for symptom in SYMPTOM_LIST]

# Load intent classification model
INTENT_MODEL = joblib.load("C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/models/saved models/intent_classifier.pkl")
VECTORIZER = joblib.load("C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/models/saved models/vectorizer.pkl")

# Preprocess text (tokenization, stopword removal, lowercase)
def preprocess_text(text):
    """Tokenization, stopword removal, and lowercase conversion."""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

def extract_symptoms(text):
    """Extract symptoms using BioBERT/ClinicalBERT NER, rule-based, and fuzzy matching."""
    detected_symptoms = set()

    # Normalize input text (replace hyphens, lowercase)
    text = text.replace("-", " ").lower()

    # NER-based extraction using BioBERT/ClinicalBERT
    ner_results = ner_pipeline(text)
    for entity in ner_results:
        if entity['entity_group'] in ["SYMPTOM", "DISEASE"]:  # Adjust based on model's labels
            detected_symptoms.add(entity['word'].lower())

    # Rule-based matching (exact match)
    for symptom in SYMPTOM_LIST:
        if re.search(rf"\b{re.escape(symptom)}s?\b", text, re.IGNORECASE):
            detected_symptoms.add(symptom)

    # Fuzzy matching (skip stopwords and use higher threshold)
    stop_words = set(stopwords.words('english'))
    for word in text.split():
        if word not in stop_words:  # Skip stopwords
            match, score = process.extractOne(word, SYMPTOM_LIST)
            if score > 90:  # Higher threshold for better accuracy
                detected_symptoms.add(match)

    # Debugging logs
    print(f"Processed Text: {text}")
    print(f"NER Entities: {ner_results}")
    print(f"Rule-Based Matches: {[s for s in SYMPTOM_LIST if re.search(rf'\b{re.escape(s)}\b', text, re.IGNORECASE)]}")
    print(f"Fuzzy Matches: {[(word, process.extractOne(word, SYMPTOM_LIST)) for word in text.split() if word not in stop_words]}")

    return list(detected_symptoms)

def classify_intent(text):
    """Classify user intent (e.g., symptom inquiry, diagnosis request)."""
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