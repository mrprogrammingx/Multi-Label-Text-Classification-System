# src/member5_evaluation.py
"""
Member 5 - Evaluation & reporting
Responsible for:
- Metrics
- Error analysis
- Reporting
"""

from sklearn.metrics import classification_report
import pandas as pd


def evaluate_model(y_true, y_pred, target_names):
    """
    Print evaluation metrics.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0
    )

    print(report)


def create_error_analysis(df, y_true, y_pred, label_name):
    """
    Generate simple error analysis table.
    """
    errors = df[(y_true[label_name] != y_pred[label_name])]

    return errors.head(10)