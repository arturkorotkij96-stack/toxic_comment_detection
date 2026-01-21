import os
import pickle
import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Essential imports for unpickling
# These must match the libraries used during training
from sklearn.feature_extraction.text import TfidfVectorizer
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

app = FastAPI(title="Toxic Comment Detection API")

# --- Helper: Model Definition ---
# We redefine create_model here in case pickle expects to find this function 
# in the __main__ scope when loading the KerasClassifier.
def create_model(meta):
    n_features_in = meta["n_features_in_"]
    model = Sequential([
        Dense(10, input_dim=n_features_in, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC'])
    return model

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Points to ../data/text_classification_pipeline.pkl relative to this file
MODEL_PATH = os.path.join(BASE_DIR, "..", "data", "text_classification_pipeline.pkl")

pipeline = None

# --- Pydantic Models ---
class CommentRequest(BaseModel):
    comments: List[str]

class PredictionResult(BaseModel):
    comment: str
    prediction: float
    label: str

class ResponseModel(BaseModel):
    results: List[PredictionResult]

# --- Lifecycle Events ---
@app.on_event("startup")
def load_pipeline():
    global pipeline
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        return

    try:
        with open(MODEL_PATH, "rb") as f:
            pipeline = pickle.load(f)
        print(f"Pipeline loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to load pipeline. {e}")

# --- Endpoints ---
@app.post("/predict", response_model=ResponseModel)
def predict_comments(request: CommentRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # KerasClassifier.predict returns class labels (0 or 1)
        predictions = pipeline.predict(request.comments)
        
        if hasattr(predictions, 'flatten'):
            predictions = predictions.flatten()
            
        results = []
        for comment, pred in zip(request.comments, predictions):
            pred_val = float(pred)
            # Assuming 1 = Toxic, 0 = Non-Toxic based on your training labels
            label = "Toxic" if pred_val == 1.0 else "Non-Toxic"
            results.append({"comment": comment, "prediction": pred_val, "label": label})
            
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": pipeline is not None}