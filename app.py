from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

# Load the model and scaler
model = joblib.load('svr_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize the FastAPI app
app = FastAPI()

# Define the input data model using Pydantic
class PredictionRequest(BaseModel):
    features: List[float]

# Define a route for prediction
@app.post('/predict')
def predict(request: PredictionRequest):
    # Extract features from the request
    features = np.array(request.features).reshape(1, -1)
    
    # Scale the features using the loaded scaler
    features_scaled = scaler.transform(features)
    
    # Predict using the trained model
    prediction = model.predict(features_scaled)
    
    # Return the prediction as a JSON response
    return {'prediction': prediction[0]}