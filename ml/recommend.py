import sys
import pandas as pd

# Load the dataset
file_path = "C:/Users/thorn/Desktop/SCHOOL/4.2/Project/Medi-bot/data/symptom_precaution.csv"
data = pd.read_csv(file_path)

def recommend_treatment(disease):
    recommendations = data[data["Disease"] == disease].iloc[:, 1:].values.flatten()
    recommendations = [rec for rec in recommendations if pd.notna(rec)]
    return recommendations

if __name__ == "__main__":
    if len(sys.argv) > 1:
        disease_name = sys.argv[1]
        recommendations = recommend_treatment(disease_name)
        print(f"Recommended Precautions for {disease_name}:")
        for rec in recommendations:
            print(f"- {rec}")
