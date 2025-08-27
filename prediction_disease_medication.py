# ----------------- Import Required Libraries -----------------
import numpy as np
import pandas as pd
from collections import Counter
import random
import os
import pickle

# Scikit-learn imports for preprocessing, models, and evaluation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler

# Dataset generation for demonstration
from sklearn.datasets import make_classification
from copy import deepcopy

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- Ensure models directory exists -----------------
os.makedirs("models", exist_ok=True)

# ----------------- Load Dataset -----------------
# Option A: Use your custom dataset (uncomment this block if you have a CSV)
# data = pd.read_csv('improved_disease_dataset.csv')
# encoder = LabelEncoder()
# data["disease_encoded"] = encoder.fit_transform(data["disease"])
# X = data.drop(columns=["disease", "medication", "disease_encoded"])
# y = data["disease_encoded"]

# Option B: Generate synthetic dataset (used here for demo)
X_raw, y = make_classification(
    n_samples=1000,          # total rows
    n_features=10,           # total features
    n_informative=5,         # useful features
    n_redundant=2,           # redundant (correlated) features
    n_classes=2,             # binary classification
    weights=[0.7, 0.3],      # class imbalance
    flip_y=0.01,             # small label noise
    random_state=42
)

# Add categorical feature (gender: 0 or 1)
rng = np.random.RandomState(42)
gender = rng.choice([0, 1], size=(X_raw.shape[0], 1), p=[0.5, 0.5])

# One-hot encode categorical feature
encoder_ohe = OneHotEncoder(drop='if_binary', sparse_output=False)
gender_encoded = encoder_ohe.fit_transform(gender)

# Combine numerical + categorical encoded features
X_final = np.hstack([X_raw, gender_encoded])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, stratify=y, test_size=0.2, random_state=42
)

# ----------------- Utility Functions -----------------

def inject_feature_noise(X, noise_level=0.01, random_state=None):
    """
    Add controlled noise to features:
      - For binary features: randomly flip some values.
      - For continuous features: add small Gaussian noise.
    """
    rng = np.random.RandomState(random_state)
    X_noisy = X.copy().astype(float)
    n_samples, n_features = X_noisy.shape
    n_noisy = int(noise_level * n_samples * n_features)  # number of noisy points

    # Identify binary features (only values {0,1})
    binary_features = []
    for i in range(n_features):
        unique_vals = np.unique(X_noisy[:, i])
        binary_features.append(set(unique_vals).issubset({0, 1}))

    if n_noisy == 0:
        return X_noisy

    # Randomly pick indices to modify
    indices = rng.choice(n_samples * n_features, n_noisy, replace=False)
    for idx in indices:
        sample_idx = idx // n_features
        feature_idx = idx % n_features
        if binary_features[feature_idx]:
            # Flip binary value
            X_noisy[sample_idx, feature_idx] = 1 - X_noisy[sample_idx, feature_idx]
        else:
            # Add Gaussian noise
            X_noisy[sample_idx, feature_idx] += rng.normal(0, 0.01)
    return X_noisy

def controlled_oversample(X, y, target_ratio, random_state=None):
    """
    Oversample the minority class until it reaches approximately target_ratio ± 5%.
    (Only supports binary classification.)
    """
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    if len(class_counts) != 2:
        raise ValueError("Currently only supports binary classification.")
    
    # Identify majority and minority
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)
    majority_count = class_counts[majority_class]
    minority_count = class_counts[minority_class]
    total_count = majority_count + minority_count

    # Compute desired minority count range based on ratio
    min_ratio = max(target_ratio - 0.05, 0)
    max_ratio = min(target_ratio + 0.05, 1)

    min_target = int(min_ratio * total_count / (1 - min_ratio)) if (1 - min_ratio) > 0 else total_count
    max_target = int(max_ratio * total_count / (1 - max_ratio)) if (1 - max_ratio) > 0 else total_count

    rng = np.random.RandomState(random_state)
    desired_minority_count = rng.randint(min_target, max_target + 1)

    # If minority is already balanced enough, skip oversampling
    if desired_minority_count <= minority_count:
        return X, y

    # Perform oversampling
    ros = RandomOverSampler(sampling_strategy={minority_class: desired_minority_count}, random_state=random_state)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res

# ----------------- Define Models and Ratios -----------------
models_info = {
    "Random Forest": {"model": RandomForestClassifier(random_state=42), "target_ratio": 0.80},
    "SVM": {"model": SVC(probability=True, random_state=42), "target_ratio": 0.75},
    "Naive Bayes": {"model": GaussianNB(), "target_ratio": 0.70},
    "Logistic Regression": {"model": LogisticRegression(max_iter=1000, random_state=42), "target_ratio": 0.75},
    "Gradient Boosting": {"model": GradientBoostingClassifier(random_state=42), "target_ratio": 0.78}
}

results = []

# ----------------- Training and Evaluation Loop -----------------
for name, info in models_info.items():
    print(f"\n=== Training model: {name} ===")

    # Step 1: Oversample minority class
    X_res, y_res = controlled_oversample(X_train, y_train, info["target_ratio"], random_state=42)
    print("Post-oversample class distribution:", dict(zip(*np.unique(y_res, return_counts=True))))

    # Step 2: Add noise to features
    X_noisy = inject_feature_noise(X_res, noise_level=0.01, random_state=42)

    # Step 3: Train the model
    clf = deepcopy(info["model"])  # copy to avoid modifying original
    clf.fit(X_noisy, y_res)

    # Step 4: Predictions
    y_pred = clf.predict(X_test)

    # Step 5: Probabilities (for ROC curves etc.)
    y_proba = None
    if hasattr(clf, "predict_proba"):
        try:
            y_proba = clf.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_test)
        if np.ptp(scores) != 0:  # avoid division by zero
            y_proba = (scores - scores.min()) / (scores.max() - scores.min())

    # Step 6: Evaluation metrics
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec = recall_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100

    print(f"{name} -> Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

    # Confusion Matrix + Report
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, zero_division=0)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

    # Step 7: Save model as pickle
    model_filename = os.path.join("models", f"{name.replace(' ', '_').lower()}.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved model to: {model_filename}")

    # Save metrics
    results.append({
        "Model": name,
        "Accuracy (%)": round(acc, 2),
        "Precision (%)": round(prec, 2),
        "Recall (%)": round(rec, 2),
        "F1-Score (%)": round(f1, 2),
    })

# ----------------- Results Summary -----------------
df_metrics = pd.DataFrame(results).set_index("Model")
print("\n=== Summary Metrics ===")
print(df_metrics)

# ----------------- Plot Metrics -----------------
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

# ----------------- Export Results -----------------
df_metrics.to_csv("model_metrics.csv")
df_metrics.to_json("model_metrics.json", orient="index")
print("Metrics exported to 'model_metrics.csv' and 'model_metrics.json'.")
