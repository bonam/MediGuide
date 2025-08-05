import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

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

# Train models
rf_model = RandomForestClassifier(random_state=42)
nb_model = GaussianNB()
svm_model = SVC(probability=True, random_state=42)

rf_model.fit(X, y)
nb_model.fit(X, y)
svm_model.fit(X, y)

# Save models
with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
with open("models/nb_model.pkl", "wb") as f:
    pickle.dump(nb_model, f)
with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
with open("models/symptoms_list.pkl", "wb") as f:
    pickle.dump(symptoms_list, f)

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

print("All models and data objects saved to 'models/' as .pkl files.")