# ================================================================
# 1. IMPORT REQUIRED LIBRARIES
# ================================================================
# Core libraries for numerical operations, data handling, and randomness
import numpy as np
import pandas as pd
import random
import os
import pickle

# For analyzing data distribution
from collections import Counter

# Machine Learning preprocessing utilities
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# ML models (we are comparing multiple algorithms)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Model evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# Model training utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Imbalanced dataset handling
from imblearn.over_sampling import RandomOverSampler

# Synthetic dataset generation for demonstration (useful if no real data)
from sklearn.datasets import make_classification

# To safely copy model objects
from copy import deepcopy

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns


# ================================================================
# 2. SETUP: ENSURE "models" DIRECTORY EXISTS
# ================================================================
# All trained models will be saved as pickle files in this folder
os.makedirs("models", exist_ok=True)


# ================================================================
# 3. LOAD DATASET
# ================================================================
# Option A: If you already have a CSV dataset with "disease" and "medication"
# Uncomment and use this section
"""
data = pd.read_csv('improved_disease_dataset.csv')
encoder = LabelEncoder()
# Encode disease column to numeric labels
data["disease_encoded"] = encoder.fit_transform(data["disease"])
# Features = everything except disease/medication columns
X = data.drop(columns=["disease", "medication", "disease_encoded"])
# Labels = encoded disease values
y = data["disease_encoded"]
"""

# Option B: Generate synthetic dataset for demonstration
# We create a binary classification dataset with class imbalance
X_raw, y = make_classification(
    n_samples=1000,        # total 1000 rows
    n_features=10,         # 10 features per row
    n_informative=5,       # 5 features carry useful signal
    n_redundant=2,         # 2 features are redundant (linear combos)
    n_classes=2,           # Binary classification problem
    weights=[0.7, 0.3],    # Class imbalance: 70% vs 30%
    flip_y=0.01,           # Add 1% random noise to labels
    random_state=42        # Seed for reproducibility
)

# Add a categorical feature "gender" (0 = male, 1 = female)
rng = np.random.RandomState(42)
gender = rng.choice([0, 1], size=(X_raw.shape[0], 1), p=[0.5, 0.5])

# One-Hot Encode categorical gender column (0/1 → binary column)
encoder_ohe = OneHotEncoder(drop='if_binary', sparse_output=False)
gender_encoded = encoder_ohe.fit_transform(gender)

# Combine original numeric features with encoded categorical feature
X_final = np.hstack([X_raw, gender_encoded])

# Split dataset into train (80%) and test (20%) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, stratify=y, test_size=0.2, random_state=42
)


# ================================================================
# 4. UTILITY FUNCTION: NOISE INJECTION
# ================================================================
def inject_feature_noise(X, noise_level=0.01, random_state=None):
    """
    Adds controlled noise to dataset features to make models more robust.

    Steps:
    1. Randomly select 'noise_level'% of all data points × features.
    2. If feature is binary → flip 0 ↔ 1.
    3. If feature is continuous → add small Gaussian noise (mean=0, std=0.01).
    """
    rng = np.random.RandomState(random_state)
    X_noisy = X.copy().astype(float)
    n_samples, n_features = X_noisy.shape
    n_noisy = int(noise_level * n_samples * n_features)  # total noisy entries

    # Identify which features are binary (only {0,1} values)
    binary_features = []
    for i in range(n_features):
        unique_vals = np.unique(X_noisy[:, i])
        binary_features.append(set(unique_vals).issubset({0, 1}))

    # If no noise needs to be injected → return original
    if n_noisy == 0:
        return X_noisy

    # Pick random feature indices to corrupt
    indices = rng.choice(n_samples * n_features, n_noisy, replace=False)
    for idx in indices:
        sample_idx = idx // n_features
        feature_idx = idx % n_features
        if binary_features[feature_idx]:
            # Flip binary feature value (0 → 1, 1 → 0)
            X_noisy[sample_idx, feature_idx] = 1 - X_noisy[sample_idx, feature_idx]
        else:
            # Add Gaussian noise to continuous feature
            X_noisy[sample_idx, feature_idx] += rng.normal(0, 0.01)
    return X_noisy


# ================================================================
# 5. UTILITY FUNCTION: CONTROLLED OVERSAMPLING
# ================================================================
def controlled_oversample(X, y, target_ratio, random_state=None):
    """
    Balances the dataset by oversampling minority class
    until its ratio reaches approximately target_ratio ± 5%.

    For example:
    - If target_ratio=0.75 → ensure minority class is ~75% of majority class.
    - Only works for binary classification.
    """
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    if len(class_counts) != 2:
        raise ValueError("Only supports binary classification.")
    
    # Identify minority vs majority class
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)
    majority_count = class_counts[majority_class]
    minority_count = class_counts[minority_class]
    total_count = majority_count + minority_count

    # Allowed range for minority ratio (± 5%)
    min_ratio = max(target_ratio - 0.05, 0)
    max_ratio = min(target_ratio + 0.05, 1)

    # Compute desired minority sample size
    min_target = int(min_ratio * total_count / (1 - min_ratio)) if (1 - min_ratio) > 0 else total_count
    max_target = int(max_ratio * total_count / (1 - max_ratio)) if (1 - max_ratio) > 0 else total_count

    rng = np.random.RandomState(random_state)
    desired_minority_count = rng.randint(min_target, max_target + 1)

    # If minority class already large enough, skip oversampling
    if desired_minority_count <= minority_count:
        return X, y

    # Oversample using imblearn's RandomOverSampler
    ros = RandomOverSampler(
        sampling_strategy={minority_class: desired_minority_count},
        random_state=random_state
    )
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res


# ================================================================
# 6. DEFINE MODELS AND THEIR TARGET RATIOS
# ================================================================
# Each model will be trained with a different oversampling ratio
models_info = {
    "Random Forest": {"model": RandomForestClassifier(random_state=42), "target_ratio": 0.80},
    "SVM": {"model": SVC(probability=True, random_state=42), "target_ratio": 0.75},
    "Naive Bayes": {"model": GaussianNB(), "target_ratio": 0.70},
    "Logistic Regression": {"model": LogisticRegression(max_iter=1000, random_state=42), "target_ratio": 0.75},
    "Gradient Boosting": {"model": GradientBoostingClassifier(random_state=42), "target_ratio": 0.78}
}

# Store metrics for comparison
results = []


# ================================================================
# 7. TRAINING + EVALUATION LOOP FOR ALL MODELS
# ================================================================
for name, info in models_info.items():
    print(f"\n=== Training model: {name} ===")

    # Step 1: Oversample training set
    X_res, y_res = controlled_oversample(X_train, y_train, info["target_ratio"], random_state=42)
    print("Post-oversample class distribution:", dict(zip(*np.unique(y_res, return_counts=True))))

    # Step 2: Inject noise for robustness
    X_noisy = inject_feature_noise(X_res, noise_level=0.01, random_state=42)

    # Step 3: Train the model
    clf = deepcopy(info["model"])  # deep copy to avoid overwriting original object
    clf.fit(X_noisy, y_res)

    # Step 4: Predict on test set
    y_pred = clf.predict(X_test)

    # Step 5: Get probability scores (useful for ROC curves, AUC, etc.)
    y_proba = None
    if hasattr(clf, "predict_proba"):
        try:
            y_proba = clf.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_test)
        if np.ptp(scores) != 0:  # avoid division by zero if all scores identical
            y_proba = (scores - scores.min()) / (scores.max() - scores.min())

    # Step 6: Calculate metrics
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec = recall_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100

    print(f"{name} -> Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

    # Step 7: Print Confusion Matrix and Classification Report
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, zero_division=0)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)

    # Step 8: Save model to disk as pickle
    model_filename = os.path.join("models", f"{name.replace(' ', '_').lower()}.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved model to: {model_filename}")

    # Store results for summary
    results.append({
        "Model": name,
        "Accuracy (%)": round(acc, 2),
        "Precision (%)": round(prec, 2),
        "Recall (%)": round(rec, 2),
        "F1-Score (%)": round(f1, 2),
    })


# ================================================================
# 8. SUMMARY OF RESULTS
# ================================================================
# Convert results to DataFrame for tabular view
df_metrics = pd.DataFrame(results).set_index("Model")
print("\n=== Summary Metrics ===")
print(df_metrics)


# ================================================================
# 9. VISUALIZE METRICS
# ================================================================
metrics_to_plot = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]

# Plot a bar chart for comparison of models
ax = df_metrics[metrics_to_plot].plot(kind="bar", figsize=(12, 6))
ax.set_title("Model Performance Metrics")
ax.set_ylabel("Percentage (%)")
ax.set_ylim(0, 110)
ax.set_xticklabels(df_metrics.index, rotation=0)
ax.legend(loc="lower right")
ax.grid(axis='y')
plt.tight_layout()
plt.show()


# ================================================================
# 10. EXPORT METRICS TO CSV & JSON
# ================================================================
df_metrics.to_csv("model_metrics.csv")
df_metrics.to_json("model_metrics.json", orient="index")
print("Metrics exported to 'model_metrics.csv' and 'model_metrics.json'.")
