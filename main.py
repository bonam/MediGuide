from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import os
from statistics import mode

# -----------------------------------------------------------
# Initialize FastAPI app with metadata
# -----------------------------------------------------------
app = FastAPI(
    title="Disease Prediction API",
    description="API for predicting diseases and recommending medications based on symptoms.",
    version="1.1.0"
)

# -----------------------------------------------------------
# Request body definition using Pydantic
# - Defines the expected input format from clients
# - Here, user provides symptoms as a comma-separated string
# -----------------------------------------------------------
class SymptomInput(BaseModel):
    symptoms: str

# -----------------------------------------------------------
# Global variables to store trained models and data artifacts
# These will be loaded once during application startup
# -----------------------------------------------------------
rf_model = None          # Random Forest classifier
nb_model = None          # Naive Bayes classifier
svm_model = None         # Support Vector Machine classifier
logreg_model = None      # Logistic Regression classifier
gb_model = None          # Gradient Boosting classifier

label_encoder = None     # Encodes/decodes disease labels
symptoms_list = None     # List of all possible symptoms (features)
symptom_index = None     # Mapping of symptom -> index in feature vector
disease_to_medication = None  # Mapping of disease -> medication(s)

# -----------------------------------------------------------
# Function to load models and required artifacts from disk
# - Uses pickle to load serialized models
# - Also loads helper mappings (symptom index, label encoder, etc.)
# -----------------------------------------------------------
def load_models():
    global rf_model, nb_model, svm_model, logreg_model, gb_model
    global label_encoder, symptoms_list, symptom_index, disease_to_medication
    models_dir = 'models'  # Directory where all models are stored

    try:
        # Core models (trained separately and saved as .pkl)
        with open(os.path.join(models_dir, 'rf_model.pkl'), 'rb') as f:
            rf_model = pickle.load(f)
        with open(os.path.join(models_dir, 'nb_model.pkl'), 'rb') as f:
            nb_model = pickle.load(f)
        with open(os.path.join(models_dir, 'svm_model.pkl'), 'rb') as f:
            svm_model = pickle.load(f)

        # Additional models for better ensemble predictions
        with open(os.path.join(models_dir, 'logreg_model.pkl'), 'rb') as f:
            logreg_model = pickle.load(f)
        with open(os.path.join(models_dir, 'gb_model.pkl'), 'rb') as f:
            gb_model = pickle.load(f)

        # Utility data for decoding predictions
        with open(os.path.join(models_dir, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        with open(os.path.join(models_dir, 'symptoms_list.pkl'), 'rb') as f:
            symptoms_list = pickle.load(f)
        with open(os.path.join(models_dir, 'disease_to_medication.pkl'), 'rb') as f:
            disease_to_medication = pickle.load(f)

        # Create a dictionary mapping symptom name -> column index
        symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms_list)}

        print("✅ All models and data loaded successfully.")
    except FileNotFoundError as e:
        raise RuntimeError(
            f"❌ Model file not found: {e}. "
            f"Please ensure models are trained and stored in 'models/' directory."
        )
    except Exception as e:
        raise RuntimeError(f"❌ Error loading models: {e}")

# -----------------------------------------------------------
# FastAPI startup event
# - Runs once when the application launches
# - Ensures models and data are ready before serving requests
# -----------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    load_models()

# -----------------------------------------------------------
# Root endpoint: basic API health-check
# -----------------------------------------------------------
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Disease Prediction API. Use /predict to get predictions."}

# -----------------------------------------------------------
# Main prediction endpoint
# - Accepts symptoms as input
# - Generates disease prediction using ensemble of ML models
# - Returns recommended medication
# -----------------------------------------------------------
@app.post("/predict")
async def predict_disease_endpoint(input_data: SymptomInput):
    """
    Predicts a disease and recommends medication 
    based on a comma-separated string of symptoms.
    """
    # ✅ Ensure all required models and artifacts are loaded
    if not all([rf_model, nb_model, svm_model, logreg_model, gb_model,
                label_encoder, symptoms_list, symptom_index, disease_to_medication]):
        raise HTTPException(
            status_code=500,
            detail="Models not loaded. Please check that the server started correctly and model files exist."
        )

    # -----------------------------------------------------------
    # Step 1: Parse and clean input symptoms
    # -----------------------------------------------------------
    symptom_input_text = input_data.symptoms or ""  # Get raw string
    # Normalize symptoms: lowercase + remove spaces
    processed_symptoms = [s.strip().lower() for s in symptom_input_text.split(",") if s.strip()]

    # -----------------------------------------------------------
    # Step 2: Convert symptoms into feature vector
    # - Vector length = total number of possible symptoms
    # - Mark '1' if symptom present, else '0'
    # -----------------------------------------------------------
    input_vector = [0] * len(symptoms_list)

    for symptom in processed_symptoms:
        if symptom in symptom_index:
            input_vector[symptom_index[symptom]] = 1
        else:
            # Log warning for unknown symptoms (ignored, not fatal)
            print(f"⚠️ Warning: Symptom '{symptom}' not recognized and will be ignored.")

    # Convert Python list -> NumPy array for model input
    input_vector_np = np.array(input_vector).reshape(1, -1)

    try:
        # -----------------------------------------------------------
        # Step 3: Run predictions using all classifiers
        # - Store encoded predictions (numeric class IDs)
        # -----------------------------------------------------------
        preds_encoded = {}
        preds_encoded["Random Forest"] = int(rf_model.predict(input_vector_np)[0])
        preds_encoded["Naive Bayes"] = int(nb_model.predict(input_vector_np)[0])
        preds_encoded["SVM"] = int(svm_model.predict(input_vector_np)[0])
        preds_encoded["Logistic Regression"] = int(logreg_model.predict(input_vector_np)[0])
        preds_encoded["Gradient Boosting"] = int(gb_model.predict(input_vector_np)[0])

        # -----------------------------------------------------------
        # Step 4: Decode predictions to human-readable disease names
        # -----------------------------------------------------------
        preds_names = {name: label_encoder.classes_[enc] for name, enc in preds_encoded.items()}

        # -----------------------------------------------------------
        # Step 5: Ensemble decision (take majority vote using mode)
        # - If tie occurs (multi-modal), fall back to Random Forest
        # -----------------------------------------------------------
        encoded_values = list(preds_encoded.values())
        try:
            final_pred_encoded = mode(encoded_values)
        except Exception:
            final_pred_encoded = preds_encoded["Random Forest"]  # Fallback

        final_pred_name = label_encoder.classes_[final_pred_encoded]

        # -----------------------------------------------------------
        # Step 6: Fetch recommended medication for predicted disease
        # -----------------------------------------------------------
        recommended_medication = disease_to_medication.get(
            final_pred_encoded,
            "No specific medication recommendation available for this disease."
        )

        # -----------------------------------------------------------
        # Step 7: Return response to client
        # -----------------------------------------------------------
        return {
            "input_symptoms": symptom_input_text,            # Original symptoms provided by user
            "predictions": preds_names,                     # Predictions from all models
            "final_ensemble_prediction": final_pred_name,   # Majority-vote disease
            "recommended_medication": recommended_medication  # Suggested medication
        }

    except Exception as e:
        # Catch-all error handler for prediction failures
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
