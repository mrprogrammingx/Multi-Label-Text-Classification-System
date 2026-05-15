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

    text = re.sub(r"http\S+|www\.\S+", " ", text)

    text = re.sub(r"<[^>]+>", " ", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\d+", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    # Remove repeated consecutive words
    words = text.split()

    cleaned_words = []

    for word in words:
        if len(cleaned_words) == 0 or cleaned_words[-1] != word:
            cleaned_words.append(word)

    return " ".join(cleaned_words)


def preprocess_dataframe(df, text_col="text"):
    df = df.copy()

    df["text_clean"] = df[text_col].apply(clean_text)

    df["is_empty_after_clean"] = (
        df["text_clean"].str.strip().eq("")
    )

    return df


class TextPreprocessor:
    def __init__(self):
        self.version = "1.0"

    def fit(self, texts):
        return self

    def transform(self, texts):
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        return [clean_text(text) for text in texts]

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def save(self, path):
        folder = os.path.dirname(path)

        if folder:
            os.makedirs(folder, exist_ok=True)

        joblib.dump(self, path)

        print("Preprocessor saved to:", path)

    @staticmethod
    def load(path):
        preprocessor = joblib.load(path)

        print("Preprocessor loaded from:", path)

        return preprocessor


def show_examples(df, text_col="text", n=5):
    sample = df.head(n)

    for i, row in sample.iterrows():
        before = row[text_col]
        after = clean_text(before)

        print("Before:", before)
        print("After: ", after)
        print("-" * 40)


def run_sanity_checks(train_path, val_path):
    print("=" * 60)
    print("PREPROCESSING SANITY CHECKS")
    print("=" * 60)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print("\nTrain shape:", train_df.shape)
    print("Val shape:", val_df.shape)

    print("\nColumns:")
    print(train_df.columns.tolist())

    print("\nBefore/After examples from train:")
    show_examples(train_df, text_col="text", n=5)

    train_clean = preprocess_dataframe(
        train_df,
        text_col="text"
    )

    val_clean = preprocess_dataframe(
        val_df,
        text_col="text"
    )

    print("\nEmpty rows after cleaning:")
    print(
        "Train:",
        train_clean["is_empty_after_clean"].sum()
    )

    print(
        "Val:  ",
        val_clean["is_empty_after_clean"].sum()
    )

    changed_train = (
        train_df["text"].apply(clean_text)
        != train_df["text"]
    ).sum()

    changed_val = (
        val_df["text"].apply(clean_text)
        != val_df["text"]
    ).sum()

    print("\nTexts changed by preprocessing:")
    print("Train:", changed_train)
    print("Val:  ", changed_val)

    print("\nRepeated word test:")

    repeated_example = (
        "feedback feedback system delay delay request"
    )

    print("Before:", repeated_example)

    print(
        "After: ",
        clean_text(repeated_example)
    )

    print("\nEdge case tests:")

    edge_cases = [
        "",
        "   ",
        None,
        "HELLO!!!",
        "Visit https://example.com",
        "<b>urgent problem</b>",
        "error 404 warning",
        "reply reply please please"
    ]

    for case in edge_cases:
        print(
            repr(case),
            "->",
            repr(clean_text(case))
        )

    print("\nSanity checks complete.")