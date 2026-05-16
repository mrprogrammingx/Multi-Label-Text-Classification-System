# Member 3 — Text Preprocessing
import re
import string
import os
import joblib
import pandas as pd


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # remove html tags
    text = re.sub(r"<[^>]+>", " ", text)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # clean extra space
    text = re.sub(r"\s+", " ", text).strip()

    # Remove repeated consecutive words
    words = text.split()

    cleaned_words = []
    seen_words = set()

    for word in words:
        if word not in seen_words:
            cleaned_words.append(word)
            seen_words.add(word)

    return " ".join(cleaned_words)


def preprocess_dataframe(df, text_column="text"):
    df = df.copy()

    df[text_column] = df[text_column].apply(clean_text)

    df["is_empty_after_clean"] = (
        df[text_column].str.strip().eq("")
    )

    return df

class TextPreprocessor:
    def transform(self, texts):
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        return [clean_text(text) for text in texts]

    def save(self, path):
        folder = os.path.dirname(path)

        if folder:
            os.makedirs(folder, exist_ok=True)

        joblib.dump(self, path)

        print("Preprocessor saved to:", path)

    @staticmethod
    def load(path):    
        return joblib.load(path)


def show_examples(df, text_column="text", n=5):
    sample = df.head(n)

    for i, row in sample.iterrows():
        before = row[text_column]
        after = clean_text(before)

        print("Before:", before)
        print("After: ", after)
        print("-" * 40)

