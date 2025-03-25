import joblib
import numpy as np

# Load the trained ML model and symptom encoder
MODEL_PATH = "C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/models/saved models/disease_predictor.pkl"
ENCODER_PATH = "C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/models/saved models/symptom_encoder.pkl"

# Load the model and encoder
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

def encode_symptoms(symptoms, all_symptoms):
    """Encode symptoms into a binary vector."""
    encoded = np.zeros(len(all_symptoms), dtype=int)
    for symptom in symptoms:
        if symptom in all_symptoms:
            index = all_symptoms.index(symptom)
            encoded[index] = 1
    return encoded

def predict_disease(symptoms, confidence_threshold=0.6):
    """Predict disease based on symptoms."""
    # Get all symptoms from the encoder
    all_symptoms = encoder.classes_

    # Encode the input symptoms
    encoded_symptoms = encode_symptoms(symptoms, all_symptoms)

    # Predict disease probabilities
    probabilities = model.predict_proba([encoded_symptoms])[0]

    # Get the predicted disease and its probability
    predicted_index = np.argmax(probabilities)
    predicted_disease = model.classes_[predicted_index]
    predicted_probability = probabilities[predicted_index]

    # Apply confidence threshold
    if predicted_probability >= confidence_threshold:
        return predicted_disease, predicted_probability
    else:
        return None, predicted_probability

if __name__ == "__main__":
    # Example usage
    symptoms = ["headache", "sore throat", "fever"]
    disease, probability = predict_disease(symptoms)

    if disease:
        print(f"Predicted Disease: {disease} (Confidence: {probability:.2f})")
    else:
        print(f"No confident prediction (Confidence: {probability:.2f})")