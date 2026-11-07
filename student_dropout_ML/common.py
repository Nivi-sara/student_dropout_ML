"""Shared helpers: metric computation used by both train.py and app.py."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef)

TARGET_COL = "Target"


def clean_columns(df):
    """UCI file has stray tabs/spaces in some column names - normalise them."""
    df = df.copy()
    df.columns = [c.replace("﻿", "").replace("\t", "").strip() for c in df.columns]
    return df


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    """Accuracy, AUC, Precision, Recall, F1, MCC.

    Multi-class problem (Dropout / Enrolled / Graduate), so precision, recall
    and F1 are weighted averages, and AUC is one-vs-rest (weighted).
    """
    if n_classes == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# Columns that hold category codes (a median of codes is meaningless), so the
# single-student form fills these with the most common value instead.
CODE_COLS = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
    "Educational special needs", "International",
]


def build_student_row(df, features, overrides):
    """One-row DataFrame with every model feature.

    Defaults: most common value for code columns, median for the rest, taken
    from the uploaded data. `overrides` (dict) holds the values typed in the form.
    """
    row = {}
    for c in features:
        if c in CODE_COLS:
            row[c] = df[c].mode().iloc[0]
        else:
            row[c] = df[c].median()
    row.update(overrides)
    return pd.DataFrame([row], columns=features)