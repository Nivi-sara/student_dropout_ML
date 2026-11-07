"""Train 5 classifiers on the UCI 'Predict Students' Dropout and Academic Success'
dataset, save them to model/, write test_data.csv and results/metrics.*

Usage:  python train.py [path_to_data.csv]
"""
import sys
import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from common import clean_columns, compute_metrics, TARGET_COL

SEED = 42
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data.csv"

# ---- load -----------------------------------------------------------------
# UCI file is semicolon separated; fall back to comma if needed.
raw = pd.read_csv(DATA_PATH, sep=None, engine="python")
df = clean_columns(raw)
assert TARGET_COL in df.columns, f"'{TARGET_COL}' column not found: {list(df.columns)[-3:]}"
print("Dataset shape:", df.shape)
print(df[TARGET_COL].value_counts())

# ---- preprocessing checks -------------------------------------------------
n_dup = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
assert df.isna().sum().sum() == 0, "missing values present - impute before training"
os.makedirs("results", exist_ok=True)
df.to_csv("results/cleaned_data.csv", index=False)
print(f"Preprocessing: removed {n_dup} duplicates, 0 missing values, shape {df.shape}")

X = df.drop(columns=[TARGET_COL])
le = LabelEncoder()
y = le.fit_transform(df[TARGET_COL])

# ---- split ----------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# test_data.csv keeps the original string label so it is human readable
test_df = X_te.copy()
test_df[TARGET_COL] = le.inverse_transform(y_te)
test_df.to_csv("test_data.csv", index=False)

# ---- models ---------------------------------------------------------------
def pipe(clf):
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

models = {
    "Logistic Regression": pipe(LogisticRegression(max_iter=2000, random_state=SEED)),
    "Decision Tree": pipe(DecisionTreeClassifier(max_depth=6, min_samples_leaf=10,
                                                 random_state=SEED)),
    "KNN": pipe(KNeighborsClassifier(n_neighbors=11, weights="distance")),
    "Naive Bayes": pipe(GaussianNB()),
    "Support Vector Machine": pipe(SVC(kernel="rbf", C=1.0, probability=True,
                                       random_state=SEED)),
}

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)
rows = []
for name, m in models.items():
    m.fit(X_tr, y_tr)
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(m, os.path.join("model", fname))
    res = compute_metrics(y_te, m.predict(X_te), m.predict_proba(X_te), len(le.classes_))
    rows.append({"ML Model Name": name, **res})
    print(name, {k: round(v, 4) for k, v in res.items()})

joblib.dump({"label_encoder": le, "feature_names": list(X.columns),
             "model_files": {n: n.lower().replace(" ", "_") + ".joblib" for n in models}},
            os.path.join("model", "artifacts.joblib"))

res_df = pd.DataFrame(rows).round(4)
res_df.to_csv("results/metrics.csv", index=False)
with open("results/metrics.md", "w") as f:
    f.write(res_df.to_markdown(index=False))
print("\n", res_df.to_string(index=False))
