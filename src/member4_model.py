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
    model = OneVsRestClassifier(LogisticRegression())

    model.fit(X_train, y_train)

    return model


def save_artifacts(model, vectorizer):
    """
    Save trained model and vectorizer.
    """
    # ensure target directory exists
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model.joblib")
    joblib.dump(vectorizer, "models/vectorizer.joblib")


def predict(model, vectorizer, text):
    """
    Predict labels for raw text.
    """
    transformed_text = vectorizer.transform([text])

    probabilities = model.predict_proba(transformed_text)

    return probabilities