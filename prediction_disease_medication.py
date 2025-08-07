import numpy as np
import pandas as pd
from collections import Counter
import random
import os
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler

from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- Ensure models dir -----------------
os.makedirs("models", exist_ok=True)

# ----------------- (Optional) Load your dataset -----------------
# If you want to use your CSV dataset, uncomment these lines and comment out the synthetic data block below
# data = pd.read_csv('improved_disease_dataset.csv')
# encoder = LabelEncoder()
# data["disease_encoded"] = encoder.fit_transform(data["disease"])
# X = data.drop(columns=["disease", "medication", "disease_encoded"])
# y = data["disease_encoded"]
# # If X is a DataFrame with categorical columns, do any required encoding here

# === Synthetic data for example (replace with your dataset) ===
X_raw, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5, n_redundant=2,
    n_classes=2, weights=[0.7, 0.3], flip_y=0.01, random_state=42
)

# Add a categorical feature (gender: 0 or 1)
rng = np.random.RandomState(42)
gender = rng.choice([0, 1], size=(X_raw.shape[0], 1), p=[0.5, 0.5])

# One-hot encode categorical feature
encoder_ohe = OneHotEncoder(drop='if_binary', sparse_output=False)
gender_encoded = encoder_ohe.fit_transform(gender)

# Combine numeric + categorical encoded features
X_final = np.hstack([X_raw, gender_encoded])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, stratify=y, test_size=0.2, random_state=42
)

# ----------------- Utility functions -----------------

def inject_feature_noise(X, noise_level=0.01, random_state=None):
    """Flip a percentage of binary features randomly; add Gaussian noise to continuous features."""
    rng = np.random.RandomState(random_state)
    X_noisy = X.copy().astype(float)
    n_samples, n_features = X_noisy.shape
    n_noisy = int(noise_level * n_samples * n_features)

    # Detect binary features once: True if all unique values are subset of {0,1}
    binary_features = []
    for i in range(n_features):
        unique_vals = np.unique(X_noisy[:, i])
        if set(unique_vals).issubset({0, 1}):
            binary_features.append(True)
        else:
            binary_features.append(False)

    if n_noisy == 0:
        return X_noisy

    indices = rng.choice(n_samples * n_features, n_noisy, replace=False)
    for idx in indices:
        sample_idx = idx // n_features
        feature_idx = idx % n_features
        if binary_features[feature_idx]:
            X_noisy[sample_idx, feature_idx] = 1 - X_noisy[sample_idx, feature_idx]
        else:
            X_noisy[sample_idx, feature_idx] += rng.normal(0, 0.01)
    return X_noisy

def controlled_oversample(X, y, target_ratio, random_state=None):
    """
    Oversample minority classes so their ratio is about target_ratio ±5%.
    Supports binary classification only.
    """
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    if len(class_counts) != 2:
        raise ValueError("Currently only supports binary classification.")
    
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)
    majority_count = class_counts[majority_class]
    minority_count = class_counts[minority_class]
    total_count = majority_count + minority_count

    # Compute desired minority count for target_ratio ± 5%
    min_ratio = max(target_ratio - 0.05, 0)
    max_ratio = min(target_ratio + 0.05, 1)

    # Calculate possible min/max minority counts (solve for minority_count in ratio = minority / (minority + majority))
    # desired_minority = ratio * majority / (1 - ratio)
    min_target = int(min_ratio * total_count / (1 - min_ratio)) if (1 - min_ratio) > 0 else total_count
    max_target = int(max_ratio * total_count / (1 - max_ratio)) if (1 - max_ratio) > 0 else total_count

    rng = np.random.RandomState(random_state)
    desired_minority_count = rng.randint(min_target, max_target + 1)

    if desired_minority_count <= minority_count:
        return X, y

    ros = RandomOverSampler(sampling_strategy={minority_class: desired_minority_count}, random_state=random_state)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res

# ------------ Define models and target oversampling ratios -------------
models_info = {
    "Random Forest": {"model": RandomForestClassifier(random_state=42), "target_ratio": 0.80},
    "SVM": {"model": SVC(probability=True, random_state=42), "target_ratio": 0.75},
    "Naive Bayes": {"model": GaussianNB(), "target_ratio": 0.70},
    # Added models
    "Logistic Regression": {"model": LogisticRegression(max_iter=1000, random_state=42), "target_ratio": 0.75},
    "Gradient Boosting": {"model": GradientBoostingClassifier(random_state=42), "target_ratio": 0.78}
}

results = []

for name, info in models_info.items():
    print(f"\n=== Training model: {name} ===")
    # Controlled oversampling
    X_res, y_res = controlled_oversample(X_train, y_train, info["target_ratio"], random_state=42)
    print("Post-oversample class distribution:", dict(zip(*np.unique(y_res, return_counts=True))))

    # Inject noise
    X_noisy = inject_feature_noise(X_res, noise_level=0.01, random_state=42)

    # Train model
    clf = deepcopy(info["model"])
    clf.fit(X_noisy, y_res)

    # Predict
    y_pred = clf.predict(X_test)

    # Probabilities/scores (for potential ROC)
    y_proba = None
    if hasattr(clf, "predict_proba"):
        try:
            y_proba = clf.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_test)
        if np.ptp(scores) == 0:
            y_proba = None
        else:
            y_proba = (scores - scores.min()) / (scores.max() - scores.min())

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec = recall_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100

    print(f"{name} -> Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

    # Confusion matrix & classification report
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, zero_division=0)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

    # Save the model as a pickle
    model_filename = os.path.join("models", f"{name.replace(' ', '_').lower()}.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved model to: {model_filename}")

    results.append({
        "Model": name,
        "Accuracy (%)": round(acc, 2),
        "Precision (%)": round(prec, 2),
        "Recall (%)": round(rec, 2),
        "F1-Score (%)": round(f1, 2),
    })

# ------------ Create results DataFrame (excluding confusion matrices) -------------
df_metrics = pd.DataFrame(results).set_index("Model")
print("\n=== Summary Metrics ===")
print(df_metrics)

# ---------------- Plot metrics ------------------
metrics_to_plot = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]

ax = df_metrics[metrics_to_plot].plot(kind="bar", figsize=(12, 6))
ax.set_title("Model Performance Metrics")
ax.set_ylabel("Percentage (%)")
ax.set_ylim(0, 110)
ax.set_xticklabels(df_metrics.index, rotation=0)
ax.legend(loc="lower right")
ax.grid(axis='y')
plt.tight_layout()
plt.show()

# -------------- Export metrics to CSV and JSON --------------
df_metrics.to_csv("model_metrics.csv")
df_metrics.to_json("model_metrics.json", orient="index")
print("Metrics exported to 'model_metrics.csv' and 'model_metrics.json'.")
