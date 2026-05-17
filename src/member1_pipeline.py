# src/member1_pipeline.py

"""
Member 1 - Pipeline / Integration
Responsible for:
- Connecting all modules
- Running full workflow
"""

from member2_data import (
    load_data,
    validate_schema,
    basic_cleaning
)

from member3_preprocess import preprocess_dataframe

from member4_model import (
    create_vectorizer,
    train_model,
    save_artifacts
)

from member5_evaluation import evaluate_model


def run_pipeline():

    # =========================
    # 1. LOAD DATA
    # =========================
    train_df, val_df = load_data(
        "data/raw/dataset_C_train.csv",
        "data/raw/dataset_C_val.csv"
    )

    # =========================
    # 2. VALIDATE SCHEMA
    # =========================
    validate_schema(train_df)

    # =========================
    # 3. BASIC CLEANING
    # =========================
    train_df = basic_cleaning(train_df)
    val_df = basic_cleaning(val_df)

    # =========================
    # 4. PREPROCESS TEXT
    # =========================
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
  
    # =========================
    # 5. DEFINE FEATURES/LABELS
    # =========================
    text_column = "text"

    label_columns = [
        col for col in train_df.columns
        if col != text_column
    ]

    X_train_text = train_df[text_column]
    y_train = train_df[label_columns]

    X_val_text = val_df[text_column]
    y_val = val_df[label_columns]
    
    # Remove any label columns that are constant in the training set
    # (e.g. all 0 or all 1). These cause warnings from scikit-learn
    # because the classifier cannot learn from a label with no variation.
    constant_labels = [c for c in label_columns if y_train[c].nunique() <= 1]
    if constant_labels:
        print(f"Warning: removing constant label columns (same value in all training samples): {constant_labels}")
        # keep only non-constant labels
        label_columns = [c for c in label_columns if c not in constant_labels]
        y_train = y_train[label_columns]
        y_val = y_val[label_columns]
    
    # =========================
    # 6. TF-IDF VECTORIZATION
    # =========================
    vectorizer = create_vectorizer()

    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    
    # =========================
    # 7. TRAIN MODEL
    # =========================
    model = train_model(X_train, y_train)

    # print("Cleaning text:", train_df["text"].iloc[0])
    # exit(0)
    # =========================
    # 8. PREDICT
    # =========================
    y_pred = model.predict(X_val)

    # =========================
    # 9. EVALUATE
    # =========================
    evaluate_model(
        y_true=y_val,
        y_pred=y_pred,
        target_names=label_columns
    )

    # =========================
    # 10. SAVE ARTIFACTS
    # =========================
    save_artifacts(model, vectorizer)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()