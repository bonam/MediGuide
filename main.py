from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import os
from statistics import mode

# Initialize FastAPI app
app = FastAPI(
    title="Disease Prediction API",
    description="API for predicting diseases and recommending medications based on symptoms.",
    version="1.1.0"
)

# Define a Pydantic model for request body
class SymptomInput(BaseModel):
    symptoms: str

# Global variables to hold loaded models and data
rf_model = None
nb_model = None
svm_model = None
logreg_model = None
gb_model = None
label_encoder = None
symptoms_list = None
symptom_index = None
disease_to_medication = None

# Function to load models and data
def load_models():
    global rf_model, nb_model, svm_model, logreg_model, gb_model
    global label_encoder, symptoms_list, symptom_index, disease_to_medication
    models_dir = 'models'

    try:
        # Core models
        with open(os.path.join(models_dir, 'rf_model.pkl'), 'rb') as f:
            rf_model = pickle.load(f)
        with open(os.path.join(models_dir, 'nb_model.pkl'), 'rb') as f:
            nb_model = pickle.load(f)
        with open(os.path.join(models_dir, 'svm_model.pkl'), 'rb') as f:
            svm_model = pickle.load(f)

        # Newly added models
        with open(os.path.join(models_dir, 'logreg_model.pkl'), 'rb') as f:
            logreg_model = pickle.load(f)
        with open(os.path.join(models_dir, 'gb_model.pkl'), 'rb') as f:
            gb_model = pickle.load(f)

        # Utilities / metadata
        with open(os.path.join(models_dir, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        with open(os.path.join(models_dir, 'symptoms_list.pkl'), 'rb') as f:
            symptoms_list = pickle.load(f)
        with open(os.path.join(models_dir, 'disease_to_medication.pkl'), 'rb') as f:
            disease_to_medication = pickle.load(f)

        # Build symptom -> index map
        symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms_list)}
        print("All models and data loaded successfully.")
    except FileNotFoundError as e:
        raise RuntimeError(f"Model file not found: {e}. Please ensure you run the model training script first and that the models are present in 'models/'.")
    except Exception as e:
        raise RuntimeError(f"Error loading models: {e}")

# Load models when the application starts
@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Disease Prediction API. Use /predict to get predictions."}

@app.post("/predict")
async def predict_disease_endpoint(input_data: SymptomInput):
    """
    Predicts a disease and recommends medication based on a comma-separated string of symptoms.
    """
    # Ensure all models and artifacts are loaded
    if not all([rf_model, nb_model, svm_model, logreg_model, gb_model,
                label_encoder, symptoms_list, symptom_index, disease_to_medication]):
        raise HTTPException(status_code=500, detail="Models not loaded. Please ensure the server started correctly and model files exist.")

    symptom_input_text = input_data.symptoms or ""
    processed_symptoms = [s.strip().lower() for s in symptom_input_text.split(",") if s.strip()]

    # Initialize feature vector with all zeros
    input_vector = [0] * len(symptoms_list)

    # Mark 1 for symptoms present in input
    for symptom in processed_symptoms:
        if symptom in symptom_index:
            input_vector[symptom_index[symptom]] = 1
        else:
            # Log the unrecognized symptom (do not fail the request)
            print(f"Warning: Symptom '{symptom}' not recognized and will be ignored.")

    # Convert to numpy array
    input_vector_np = np.array(input_vector).reshape(1, -1)

    try:
        # Generate predictions from all models
        preds_encoded = {}

        preds_encoded["Random Forest"] = int(rf_model.predict(input_vector_np)[0])
        preds_encoded["Naive Bayes"] = int(nb_model.predict(input_vector_np)[0])
        preds_encoded["SVM"] = int(svm_model.predict(input_vector_np)[0])
        preds_encoded["Logistic Regression"] = int(logreg_model.predict(input_vector_np)[0])
        preds_encoded["Gradient Boosting"] = int(gb_model.predict(input_vector_np)[0])

        # Convert encoded predictions to names
        preds_names = {name: label_encoder.classes_[enc] for name, enc in preds_encoded.items()}

        # Combine predictions using mode
        encoded_values = list(preds_encoded.values())
        try:
            final_pred_encoded = mode(encoded_values)
        except Exception:
            # fallback: choose the Random Forest prediction if mode fails (e.g., multimodal)
            final_pred_encoded = preds_encoded["Random Forest"]

        final_pred_name = label_encoder.classes_[final_pred_encoded]

        # Get recommended medication
        recommended_medication = disease_to_medication.get(final_pred_encoded,
                                                          "No specific medication recommendation available for this disease.")

        return {
            "input_symptoms": symptom_input_text,
            "predictions": preds_names,
            "final_ensemble_prediction": final_pred_name,
            "recommended_medication": recommended_medication
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
