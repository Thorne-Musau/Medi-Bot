# Medi-bot Project

## Overview
Medi-bot is a machine learning-based medical diagnosis system that predicts diseases based on symptoms and provides treatment recommendations.

## Features
- Disease prediction using Random Forest and SVM models
- Treatment recommendations for predicted diseases
- Model evaluation with cross-validation
- Support for multiple machine learning algorithms

## Models
The project uses two main machine learning models:
1. Random Forest Classifier
2. Support Vector Machine (SVM)

## Installation
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r docs/requirements.txt
   ```

## Usage
1. Train the models:
   ```bash
   python models/training/train.py
   ```
2. Evaluate model performance:
   ```bash
   python models/training/evaluation.py
   ```
3. Get treatment recommendations:
   ```bash
   python models/training/recommend.py "disease_name"
   ```

## Dependencies
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Model Performance
The system evaluates models using:
- Accuracy metrics
- Classification reports
- 5-fold cross-validation

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
[Add your license information here]

## Authors
[Add author information here]