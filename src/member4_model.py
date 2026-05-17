# src/member4_model.py
"""
Member 4 - Modeling & inference
Responsible for:
- TF-IDF vectorization
- Model training
- Prediction pipeline
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os


def create_vectorizer():
    """
    Create TF-IDF vectorizer.
    """
    return TfidfVectorizer(max_features=5000)


def train_model(X_train, y_train):
    """
    Train multi-label classifier.
    """
    model = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000))

    print("Member 4: model training started...")
    model.fit(X_train, y_train)
    print("Member 4: training comlete.")

    return model


def save_artifacts(model, vectorizer):
    """
    Save trained model and vectorizer.
    """
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model.joblib")
    joblib.dump(vectorizer, "models/vectorizer.joblib")
    print("Member 4: model saved to folder models/")


def predict(model, vectorizer, text_or_texts):
    """
    Predict labels and confidence scores for raw text(s).
    """
    if isinstance(text_or_texts, str):
        text_or_texts = [text_or_texts]

    transformed_text = vectorizer.transform(text_or_texts)

    predictions = model.predict(transformed_text)

    probabilities = model.predict_proba(transformed_text)

    return predictions, probabilities