"""
Dry Bean Classifier - Streamlit web app.

Upload the test CSV, pick one of the six trained models, and view its
evaluation metrics, confusion matrix and classification report.
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
TARGET = "Class"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘", layout="wide")


@st.cache_resource
def load_artifacts():
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)
    models = {
        name: joblib.load(os.path.join(MODEL_DIR, fname))
        for name, fname in MODEL_FILES.items()
    }
    return label_encoder, feature_columns, models


def compute_metrics(y_true, y_pred, proba, classes):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(
            y_true, proba, multi_class="ovr", average="macro", labels=classes
        ),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


label_encoder, feature_columns, models = load_artifacts()
class_names = list(label_encoder.classes_)
class_ids = np.arange(len(class_names))

st.title("🫘 Dry Bean Multi-Class Classifier")
st.write(
    "Upload the **test data (CSV)**, choose a model, and explore its performance. "
    "The dataset has 16 shape features and 7 bean varieties "
    "(Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira)."
)

with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    model_name = st.selectbox("Select model", list(MODEL_FILES.keys()))
    st.caption("Tip: use the provided `test_data.csv`.")

if uploaded is None:
    st.info("👈 Upload `test_data.csv` from the sidebar to run predictions.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Preview of uploaded data")
st.dataframe(df.head(), use_container_width=True)

missing = [c for c in feature_columns if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing required feature columns: {missing}")
    st.stop()

X = df[feature_columns]
model = models[model_name]
y_pred = model.predict(X)
proba = model.predict_proba(X)
pred_labels = label_encoder.inverse_transform(y_pred)

has_labels = TARGET in df.columns

if has_labels:
    y_true = label_encoder.transform(df[TARGET])
    metrics = compute_metrics(y_true, y_pred, proba, class_ids)

    st.subheader(f"📊 Evaluation metrics — {model_name}")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_true, y_pred, labels=class_ids)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

    with right:
        st.subheader("Classification report")
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
else:
    st.warning(
        f"No `{TARGET}` column found — showing predictions only "
        "(metrics need ground-truth labels)."
    )

st.subheader("🔮 Predictions")
out = df.copy()
out["Predicted_Class"] = pred_labels
st.dataframe(out.head(50), use_container_width=True)
st.download_button(
    "Download predictions CSV",
    out.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv",
)
