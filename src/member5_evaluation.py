# src/member5_evaluation.py
"""
Member 5 - Evaluation & reporting
Responsible for:
- Metrics
- Error analysis
- Reporting
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support


def evaluate_model(y_true, y_pred, target_names):
    """Print readable per-label precision/recall/F1 table."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    df = pd.DataFrame({
        "Precision": precision.round(3),
        "Recall":    recall.round(3),
        "F1":        f1.round(3),
        "Support":   support.astype(int),
    }, index=target_names)

    print(df.to_string())
    return df


def create_error_analysis(df, y_true, y_pred, target_names, text_column="text", max_examples=5):
    """
    For each label, print misclassified examples split by
    False Positives and False Negatives.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    error_tables = {}
    for i, label in enumerate(target_names):
        fp = (y_pred_arr[:, i] == 1) & (y_true_arr[:, i] == 0)
        fn = (y_pred_arr[:, i] == 0) & (y_true_arr[:, i] == 1)

        print(f"\n[{label}]  FP={fp.sum()}  FN={fn.sum()}")
        for mask, kind in [(fp, "False Positive"), (fn, "False Negative")]:
            samples = df[mask][text_column].head(max_examples)
            if samples.empty:
                continue
            print(f"  {kind}:")
            for text in samples:
                print(f"    - {str(text)[:120]}")

        combined = pd.concat([
            df[fp][[text_column]].assign(error="FP"),
            df[fn][[text_column]].assign(error="FN"),
        ])
        error_tables[label] = combined

    return error_tables


def plot_f1_bar_chart(metrics_df, output_path="reports/f1_bar_chart.png"):
    """Save a horizontal bar chart of per-label F1 scores."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, len(metrics_df) * 0.5 + 1))
    metrics_df["F1"].plot.barh(ax=ax, color="steelblue")
    ax.axvline(metrics_df["F1"].mean(), color="orange", linestyle="--", label="Macro avg")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Label F1")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Chart saved to {output_path}")