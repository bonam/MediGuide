import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Create 'models' directory if it does not exist
os.makedirs('models', exist_ok=True)

# Load dataset
data = pd.read_csv('improved_disease_dataset.csv')

# Encode target
label_encoder = LabelEncoder()
data["disease_encoded"] = label_encoder.fit_transform(data["disease"])

# Extract features and target
X = data.drop(columns=["disease", "medication", "disease_encoded"])
y = data["disease_encoded"]

# Save the list of symptoms (feature names)
symptoms_list = list(X.columns)

# Initialize models
models = {
    "rf_model": RandomForestClassifier(random_state=42),
    "nb_model": GaussianNB(),
    "svm_model": SVC(probability=True, random_state=42),
    "logreg_model": LogisticRegression(max_iter=1000, random_state=42),
    "gb_model": GradientBoostingClassifier(random_state=42)
}

# Train models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X, y)
    # Save each model to models/<name>.pkl
    model_path = os.path.join("models", f"{name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {name} -> {model_path}")

# Save the label encoder and symptoms list
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("Saved label_encoder -> models/label_encoder.pkl")

with open("models/symptoms_list.pkl", "wb") as f:
    pickle.dump(symptoms_list, f)
print("Saved symptoms_list -> models/symptoms_list.pkl")

# Create disease-to-medication mapping (encoded)
disease_to_medication = {}
for _, row in data.iterrows():
    encoded = row["disease_encoded"]
    medication = row["medication"]
    # Only add if not present (first medication for disease)
    if encoded not in disease_to_medication:
        disease_to_medication[encoded] = medication

with open("models/disease_to_medication.pkl", "wb") as f:
    pickle.dump(disease_to_medication, f)
print("Saved disease_to_medication -> models/disease_to_medication.pkl")

print("All models and data objects saved to 'models/' as .pkl files.")
