import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------------
# STEP 1: Create 'models' directory if it does not exist
# This ensures all trained models and supporting objects are saved in one location
# ---------------------------------------------------------------------------------
os.makedirs('models', exist_ok=True)

# ---------------------------------------------------------------------------------
# STEP 2: Load dataset
# The dataset should have at least the following columns:
#   - symptoms (multiple binary/encoded columns for each symptom)
#   - disease (target label, categorical string)
#   - medication (the recommended drug for that disease)
# ---------------------------------------------------------------------------------
data = pd.read_csv('improved_disease_dataset.csv')

# ---------------------------------------------------------------------------------
# STEP 3: Encode target (disease column)
# Machine learning models cannot handle string labels directly, so we use LabelEncoder
# to convert each disease name into a unique integer ID (encoded label).
# Example:
#   "Flu" -> 0
#   "Diabetes" -> 1
#   "Migraine" -> 2
# The mapping is stored so we can decode predictions back to original diseases.
# ---------------------------------------------------------------------------------
label_encoder = LabelEncoder()
data["disease_encoded"] = label_encoder.fit_transform(data["disease"])

# ---------------------------------------------------------------------------------
# STEP 4: Extract features (X) and target (y)
# X: all symptom columns (independent variables)
# y: encoded disease labels (dependent variable)
# We drop:
#   - "disease" (string labels, not needed after encoding)
#   - "medication" (not used for model training, only for mapping later)
#   - "disease_encoded" is kept separately as the target
# ---------------------------------------------------------------------------------
X = data.drop(columns=["disease", "medication", "disease_encoded"])
y = data["disease_encoded"]

# ---------------------------------------------------------------------------------
# STEP 5: Save the list of symptoms (feature names)
# This is important because when a user inputs symptoms later, we must ensure
# they align with the same feature order used in training.
# Example:
#   symptoms_list = ["fever", "cough", "fatigue", ...]
# ---------------------------------------------------------------------------------
symptoms_list = list(X.columns)

# ---------------------------------------------------------------------------------
# STEP 6: Initialize models
# We will train multiple ML algorithms and save all of them.
#   - RandomForestClassifier: ensemble of decision trees, good for tabular data
#   - GaussianNB: Naive Bayes classifier, works well for probabilistic classification
#   - SVM (Support Vector Machine): finds decision boundaries, supports probability
#   - Logistic Regression: linear classifier, baseline benchmark
#   - GradientBoostingClassifier: boosting trees, powerful for structured datasets
# ---------------------------------------------------------------------------------
models = {
    "rf_model": RandomForestClassifier(random_state=42),
    "nb_model": GaussianNB(),
    "svm_model": SVC(probability=True, random_state=42),
    "logreg_model": LogisticRegression(max_iter=1000, random_state=42),
    "gb_model": GradientBoostingClassifier(random_state=42)
}

# ---------------------------------------------------------------------------------
# STEP 7: Train models and save them
# For each model:
#   - Train (fit) using training data (X, y)
#   - Save trained model as a .pkl file inside "models/" folder
# ---------------------------------------------------------------------------------
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X, y)  # Fit model on symptom features and disease labels
    
    # Save each trained model as pickle file for later inference
    model_path = os.path.join("models", f"{name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {name} -> {model_path}")

# ---------------------------------------------------------------------------------
# STEP 8: Save supporting objects for inference
# We need:
#   - LabelEncoder (to decode predicted disease IDs back to names)
#   - Symptoms list (to ensure same feature order in input preprocessing)
# ---------------------------------------------------------------------------------
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("Saved label_encoder -> models/label_encoder.pkl")

with open("models/symptoms_list.pkl", "wb") as f:
    pickle.dump(symptoms_list, f)
print("Saved symptoms_list -> models/symptoms_list.pkl")

# ---------------------------------------------------------------------------------
# STEP 9: Create disease-to-medication mapping
# This dictionary links encoded disease IDs to their recommended medication.
# Example:
#   {0: "Paracetamol", 1: "Metformin", 2: "Ibuprofen"}
# Only the first medication for each disease is stored (if duplicates exist).
# ---------------------------------------------------------------------------------
disease_to_medication = {}
for _, row in data.iterrows():
    encoded = row["disease_encoded"]
    medication = row["medication"]
    # Store medication only if disease not already mapped
    if encoded not in disease_to_medication:
        disease_to_medication[encoded] = medication

# Save mapping as pickle file
with open("models/disease_to_medication.pkl", "wb") as f:
    pickle.dump(disease_to_medication, f)
print("Saved disease_to_medication -> models/disease_to_medication.pkl")

# ---------------------------------------------------------------------------------
# FINAL MESSAGE
# At this point, all models and objects are saved in 'models/' folder:
#   - rf_model.pkl, nb_model.pkl, svm_model.pkl, logreg_model.pkl, gb_model.pkl
#   - label_encoder.pkl (for decoding predictions)
#   - symptoms_list.pkl (feature order reference)
#   - disease_to_medication.pkl (disease-to-drug mapping)
# These files will be used later in FastAPI backend for prediction & recommendation.
# ---------------------------------------------------------------------------------
print("All models and data objects saved to 'models/' as .pkl files.")
