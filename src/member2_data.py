# src/member2_data.py
"""
Member 2 - Data ingestion & schema
Responsible for:
- Loading datasets
- Validating schema
- Basic cleaning/checks
"""

import pandas as pd


def load_data(train_path, val_path):
    """
    Load training and validation datasets.
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    return train_df, val_df


def validate_schema(df):
    """
    Validate dataset structure.
    Modify based on actual dataset columns.
    """
    required_columns = ["text"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    print("Schema validation passed.")


def basic_cleaning(df):
    """
    Handle missing values and duplicates.
    """
    df = df.drop_duplicates()
    df = df.dropna()

    return df