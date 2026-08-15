# Dry Bean Multi-Class Classification 🫘

An end-to-end machine learning project that trains five classification models on the
Kaggle **Dry Bean** dataset and serves them through an interactive **Streamlit** web app.

---

## a. Problem Statement

Given 16 numeric shape/geometry features extracted from images of dried beans,
predict the **bean variety** (one of 7 classes). This is a **multi-class
classification** problem. The app lets a user upload test data, pick a trained
model, and view its evaluation metrics, confusion matrix and classification report.

---

## b. Dataset Description

- **Source:** Kaggle - *Dry Bean Dataset*.
- **Instances:** 13,611
- **Features:** 16 numeric features (Area, Perimeter, MajorAxisLength,
  MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent,
  Solidity, roundness, Compactness, ShapeFactor1–4).
- **Target (Class):** 7 bean varieties — BARBUNYA, BOMBAY, CALI, DERMASON,
  HOROZ, SEKER, SIRA.
- **Split:** 80% train / 20% test (stratified). The 20% test split is exported to
  **test_data.csv** and used in the Streamlit app.

| Requirement | Minimum | This dataset |
|---|---|---|
| Feature size | 12 | 16 |
| Instance size | 500 | 13,611 |

---

## c. GitHub Repository Link

> [https://github.com/sanchit-singla-2025AC05596/dry-bean-classification](https://github.com/sanchit-singla-2025AC05596/dry-bean-classification)

**Repository structure**

```
project-folder/
│-- app.py                 # Streamlit web application
│-- requirements.txt
│-- README.md
│-- test_data.csv          # 20% held-out test split
│-- Dry_Bean_Dataset.csv   # full dataset
│-- model/
│    │-- train_models.py    # trains + saves all 5 models
│    │-- *.joblib           # saved model pipelines
│    │-- label_encoder.joblib
│    │-- feature_columns.json
│    │-- metrics_comparison.csv
```

---

## d. Models Used — Comparison Table

All models are `StandardScaler` + estimator pipelines. Metrics are computed on the
20% held-out test set. Precision / Recall / F1 use macro averaging; AUC uses macro
one-vs-rest.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9210 | 0.9947 | 0.9353 | 0.9322 | 0.9335 | 0.9046 |
| Decision Tree | 0.8928 | 0.9831 | 0.9044 | 0.9009 | 0.9021 | 0.8705 |
| kNN | 0.9115 | 0.9853 | 0.9273 | 0.9214 | 0.9241 | 0.8929 |
| Naive Bayes | 0.8979 | 0.9916 | 0.9116 | 0.9094 | 0.9092 | 0.8773 |
| Random Forest (Ensemble) | 0.9185 | 0.9943 | 0.9337 | 0.9290 | 0.9312 | 0.9014 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline; highest accuracy (0.9210) and best AUC (0.9947). The classes are largely linearly separable in the scaled feature space, so a linear model performs excellently. |
| Decision Tree | Regularized with `max_depth=6` and `min_samples_leaf=20`. Without these limits the tree memorises the training data (perfect 1.0 train scores); after regularization train/test are close (train ~0.91, test 0.8928) and AUC improves to 0.9831. Still the weakest model but no longer overfitting. |
| kNN | Solid performance (0.9115) after scaling, which is essential since kNN is distance-based. Slightly behind the linear and ensemble models. |
| Naive Bayes | Decent accuracy (0.8979) and surprisingly high AUC (0.9916) despite its strong feature-independence assumption, which is only partly true for these correlated geometric features. |
| Random Forest (Ensemble) | Regularized with `max_depth=12`, `min_samples_leaf=5`, `max_features='sqrt'`. This closes the overfitting gap (train ~0.96 vs test 0.9185 instead of a perfect 1.0 train score) while staying very strong and robust (0.9185 accuracy, 0.9943 AUC). |
| **Overall Winner for your dataset?** | **Logistic Regression** — highest accuracy (0.9210) and best AUC (0.9947), while also being the simplest and fastest model. Random Forest is a close second and the best non-linear alternative. |

---

## Streamlit App Features

- 📤 **CSV upload** — upload the provided `test_data.csv`.
- 🔽 **Model selection dropdown** — choose any of the 5 trained models.
- 📊 **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1, MCC.
- 🔲 **Confusion matrix** and **classification report**.
- 🔮 **Predictions table** with downloadable results.

### Live App

> [Dry Bean Multi-Class Classifier](https://dry-bean-classification-ml-end-to-end.streamlit.app/)

---
