import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(ROOT, "Dry_Bean_Dataset.csv")
MODEL_DIR = HERE
TARGET = "Class"
RANDOM_STATE = 42


def build_models():
    """Return the six classifiers, each wrapped in a scaling pipeline."""
    scaler = lambda est: Pipeline([("scaler", StandardScaler()), ("clf", est)])
    return {
        "Logistic Regression": scaler(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "Decision Tree": scaler(
            DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE
            )
        ),
        "kNN": scaler(KNeighborsClassifier(n_neighbors=7)),
        "Naive Bayes": scaler(GaussianNB()),
        "Random Forest (Ensemble)": scaler(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=RANDOM_STATE,
            )
        ),
    }


def evaluate(model, X_test, y_test, classes):
    """Compute the six required metrics for a fitted model."""
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(
            y_test, proba, multi_class="ovr", average="macro", labels=classes
        ),
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y_raw = df[TARGET]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    classes = np.arange(len(label_encoder.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(list(X.columns), f)

    test_df = X_test.copy()
    test_df[TARGET] = label_encoder.inverse_transform(y_test)
    test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)

    results = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, classes)
        results.append({"Model": name, **metrics})

        fname = name.split(" (")[0].lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, os.path.join(MODEL_DIR, fname))
        print(f"Saved {fname}: {metrics}")

    comparison = pd.DataFrame(results).set_index("Model").round(4)
    comparison.to_csv(os.path.join(MODEL_DIR, "metrics_comparison.csv"))
    print("\n=== Comparison Table ===")
    print(comparison.to_string())
    print("\nBest by Accuracy:", comparison["Accuracy"].idxmax())


if __name__ == "__main__":
    main()
