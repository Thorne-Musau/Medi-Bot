import symptom_extraction
import predict_disease
import recommend

def chatbot_response(user_input):
    """Process user input and return chatbot's medical response."""
    
    # Extract symptoms
    symptoms = symptom_extraction.extract_symptoms(user_input)
    
    if not symptoms:
        return "I'm sorry, I couldn't recognize any symptoms. Could you describe them differently?"

    # Predict disease
    disease, confidence = predict_disease.predict_disease(symptoms)

    # Retrieve recommendations
    if confidence >= 0.6:
        recommendations = recommend.get_precautions(disease)
        response = f"Based on your symptoms, you might have *{disease}* (Confidence: {confidence:.2f}).\n"
        response += "Recommended actions:\n" + "\n".join([f"- {rec}" for rec in recommendations])
    else:
        response = "I'm unsure about the diagnosis. Please provide more symptoms for a better prediction."

    return response

if __name__ == "__main__":
    while True:
        user_input = input("\nDescribe your symptoms: ")
        print(chatbot_response(user_input))
