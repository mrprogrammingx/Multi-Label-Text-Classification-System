# src/member3_preprocess.py
"""
Member 3 - Text preprocessing
Responsible for:
- Text cleaning
- Normalization
- Shared preprocessing functions
"""

import re


def clean_text(text):
    """
    Basic text cleaning function.
    """
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\\s+", " ", text).strip()

    return text


def preprocess_dataframe(df, text_column="text"):
    """
    Apply preprocessing to dataframe.
    """
    df[text_column] = df[text_column].apply(clean_text)

    return df