import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load models
with open("models/resolution_model.pkl", "rb") as f:
    resolution_model = pickle.load(f)

with open("models/risk_model.pkl", "rb") as f:
    risk_model = pickle.load(f)

# Load scalers
with open("models/scaler_features.pkl", "rb") as f:
    scaler_features = pickle.load(f)

with open("models/scaler_target.pkl", "rb") as f:
    scaler_target = pickle.load(f)

app = FastAPI()

# Define request structure
class Incident(BaseModel):
    priority: int
    impact: int
    urgency: int
    reassignment_count: int
    reopen_count: int
    complexity_score: int

@app.post("/predict/")
def predict_risk(incident: Incident):
    # Convert JSON input to DataFrame
    incident_data = pd.DataFrame([incident.dict()])

    # Scale features for resolution time prediction
    incident_scaled = scaler_features.transform(incident_data)

    # Predict resolution time
    predicted_time_scaled = resolution_model.predict(incident_scaled)
    predicted_time = np.square(scaler_target.inverse_transform(predicted_time_scaled.reshape(-1, 1))).ravel()[0]
    predicted_time = max(0, float(predicted_time))  # Convert NumPy float to standard Python float

    # Ensure the order of features matches the training data
    feature_order = ["priority", "impact", "urgency", "reassignment_count", "reopen_count", "time_to_resolution", "complexity_score"]
    
    # Add predicted `time_to_resolution` to data
    incident_data["time_to_resolution"] = predicted_time

    # Reorder columns to match model's expected input
    incident_data = incident_data[feature_order]

    # Predict risk classification using updated data
    risk_prediction = risk_model.predict(incident_data)[0]

    # Convert numeric prediction to readable categories
    risk_category = "HIGH_RISK" if risk_prediction == 1 else "LOW_RISK"

    return {
        "predicted_resolution_time": round(predicted_time, 2),
        "risk_category": risk_category
    }
