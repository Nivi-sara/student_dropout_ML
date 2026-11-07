"""Streamlit app: Student Dropout & Academic Success - model evaluation dashboard.

Loads the pre-trained models from model/ (no retraining) and evaluates them on an
uploaded test CSV that contains the feature columns plus the 'Target' column.
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, classification_report

from common import clean_columns, compute_metrics, TARGET_COL

st.set_page_config(page_title="Student Outcome Predictor", layout="wide")

# ---------------------------------------------------------------- pink styling
PINK_CMAP = LinearSegmentedColormap.from_list("soft_pink", ["#fff5f8", "#f7b6cd", "#d6336c"])
PINK_SHADES = ["#fbd3e0", "#f7a8c4", "#f17aa6", "#e75480", "#c2386a", "#8e2150"]
CLASS_PINKS = ["#f7a8c4", "#e75480", "#c2386a"]
MODEL_PINKS = ["#f7c1d6", "#f17aa6", "#e75480", "#c2386a", "#8e2150"]
HIGHLIGHT = "#f9c5d8"

st.markdown(
    """
    <style>
    h1, h2, h3 { color: #b0306a; }
    [data-testid="stMetricValue"] { color: #c2386a; }
    [data-testid="stMetric"] { background: #fde2ea; border-radius: 12px; padding: 10px 14px; }
    button[data-baseweb="tab"] { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Locate the model folder relative to this file, so it works wherever the app is
# launched from (local run, or a repo with app.py inside a sub-folder).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_candidates = [os.path.join(BASE_DIR, "model"),
               os.path.join(os.path.dirname(BASE_DIR), "model"),
               os.path.join(os.getcwd(), "model")]
MODEL_DIR = next((p for p in _candidates
                  if os.path.exists(os.path.join(p, "artifacts.joblib"))), _candidates[0])


@st.cache_resource
def load_artifacts():
    if not os.path.exists(os.path.join(MODEL_DIR, "artifacts.joblib")):
        st.error("Model files not found. Make sure the 'model' folder (with artifacts.joblib "
                 "and the five .joblib model files) is uploaded next to app.py in the repository.")
        st.stop()
    art = joblib.load(os.path.join(MODEL_DIR, "artifacts.joblib"))
    models = {n: joblib.load(os.path.join(MODEL_DIR, f)) for n, f in art["model_files"].items()}
    return art, models


art, models = load_artifacts()
le = art["label_encoder"]
features = art["feature_names"]
classes = list(le.classes_)

# ---------------------------------------------------------------- header
st.title("Student Dropout and Academic Success")

st.caption("Compare five pre-trained classifiers on higher-education student records "
           "(predicting Dropout / Enrolled / Graduate). Models are loaded from disk - "
           "nothing is retrained on your upload.")

st.caption("Student Name/ Student id: Nivetha E/ 2026AJ05140")

# ---------------------------------------------------------------- inputs (on the page)
in_left, in_right = st.columns(2)
with in_left:
    st.subheader("Data")
    up = st.file_uploader("Upload test CSV", type="csv")

# Nothing else is shown until a file has been uploaded
if up is None:
    st.info("Upload your test CSV to select a model and view the results.")
    st.stop()

with in_right:
    st.subheader("Model")
    model_name = st.selectbox("Choose a classifier", list(models.keys()))

st.divider()

# ---------------------------------------------------------------- data
df = clean_columns(pd.read_csv(up, sep=None, engine="python"))
missing = [c for c in features + [TARGET_COL] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {missing}")
    st.stop()

unknown = set(df[TARGET_COL].unique()) - set(classes)
if unknown:
    st.error(f"Unexpected Target labels {unknown}. Expected: {classes}")
    st.stop()

X = df[features]
y_true = le.transform(df[TARGET_COL])


def evaluate(name):
    m = models[name]
    return compute_metrics(y_true, m.predict(X), m.predict_proba(X), len(classes))


# ---------------------------------------------------------------- sections
tab_over, tab_eval, tab_cmp = st.tabs(
    ["Overview", "Evaluate model", "Compare models"])

# ---- 1. Overview ------------------------------------------------------------
with tab_over:
    st.subheader("Uploaded data at a glance")
    a, b, c = st.columns(3)
    a.metric("Students (rows)", f"{len(df):,}")
    b.metric("Features", len(features))
    c.metric("Classes", len(classes))

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Class distribution**")
        counts = df[TARGET_COL].value_counts().reindex(classes)
        fig_d, ax_d = plt.subplots(figsize=(5, 3.4))
        bars = ax_d.bar(counts.index, counts.values, color=CLASS_PINKS)
        ax_d.bar_label(bars, padding=3)
        ax_d.set_ylabel("Students")
        ax_d.spines[["top", "right"]].set_visible(False)
        fig_d.tight_layout()
        st.pyplot(fig_d)
    with right:
        st.markdown("**First rows of the file**")
        st.dataframe(df.head(10), use_container_width=True, height=300)

# ---- 2. Evaluate one model -------------------------------------------------------
with tab_eval:
    st.subheader(f"{model_name}")
    res = evaluate(model_name)
    cols = st.columns(6)
    for c, (k, v) in zip(cols, res.items()):
        c.metric(k, f"{v:.4f}")

    y_pred = models[model_name].predict(X)

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap=PINK_CMAP, xticklabels=classes,
                    yticklabels=classes, linewidths=1, linecolor="white", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        rep = classification_report(y_true, y_pred, target_names=classes,
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

# ---- 3. Compare models -----------------------------------------------------------
with tab_cmp:
    st.subheader("All classifiers on the uploaded data")
    comp = pd.DataFrame({n: evaluate(n) for n in models}).T.round(4)
    comp.index.name = "ML Model Name"
    st.dataframe(comp.style.highlight_max(axis=0, color=HIGHLIGHT), use_container_width=True)
    st.caption("Pink cells mark the best value in each metric column.")

    st.markdown("**Interactive metric comparison**")
    st.caption("Hover over a bar to see the exact score. Use the controls to choose "
               "what to compare.")
    c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
    chosen = c1.multiselect("Metrics", list(comp.columns), default=list(comp.columns))
    sel_models = c2.multiselect("Classifiers", list(comp.index), default=list(comp.index))
    sort_by = c3.selectbox("Sort classifiers by", list(comp.columns),
                           index=list(comp.columns).index("MCC"))
    view = c4.radio("Group bars by", ["Classifier", "Metric"], horizontal=True)

    if not chosen or not sel_models:
        st.info("Pick at least one metric and one classifier to draw the chart.")
    else:
        order = [m for m in comp.sort_values(sort_by, ascending=False).index if m in sel_models]
        try:
            import altair as alt
            long = (comp.loc[order, chosen].reset_index()
                    .melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
                    .rename(columns={"ML Model Name": "Model"}))
            if view == "Classifier":
                x = alt.X("Model:N", sort=order, title=None, axis=alt.Axis(labelAngle=-15))
                offset = alt.XOffset("Metric:N")
                color = alt.Color("Metric:N", sort=chosen,
                                  scale=alt.Scale(domain=list(comp.columns), range=PINK_SHADES),
                                  legend=alt.Legend(orient="top", title=None))
            else:
                x = alt.X("Metric:N", sort=chosen, title=None, axis=alt.Axis(labelAngle=0))
                offset = alt.XOffset("Model:N", sort=order)
                color = alt.Color("Model:N", sort=order,
                                  scale=alt.Scale(domain=order, range=MODEL_PINKS[:len(order)]),
                                  legend=alt.Legend(orient="top", title=None))
            chart = (alt.Chart(long)
                     .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                     .encode(x=x, xOffset=offset,
                             y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
                             color=color,
                             tooltip=["Model:N", "Metric:N", alt.Tooltip("Score:Q", format=".4f")])
                     .properties(height=380))
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            # simple fallback so the page never breaks
            st.bar_chart(comp.loc[order, chosen])

    best = comp["MCC"].idxmax()
    st.success(f"Highest MCC on this data: **{best}** ({comp.loc[best, 'MCC']:.4f})")
    st.download_button("Download comparison as CSV", comp.to_csv().encode(),
                       "model_comparison.csv", "text/csv")
