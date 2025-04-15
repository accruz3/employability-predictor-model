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
    PracticumGrade: int
    WebDevGrade: int
    DSAGrade: int
    FundamentalsProgGrade: int
    OOPGrade: int	
    FoundationsCSGrade: int	
    NetworkingGrade: int	
    NumericComputationGrade: int	
    ExtracurricularsLevel: int	
    LatinHonors: int	
 
# Define a route for prediction
@app.post('/predict')
def predict(request: PredictionRequest):
    # Extract features from the request
    features = np.array([
        request.PracticumGrade,
        request.WebDevGrade,
        request.DSAGrade,
        request.FundamentalsProgGrade,
        request.OOPGrade,
        request.FoundationsCSGrade,
        request.NetworkingGrade,
        request.NumericComputationGrade,
        request.ExtracurricularsLevel,
        request.LatinHonors
    ]).reshape(1, -1)
    
    # Scale the features using the loaded scaler
    features_scaled = scaler.transform(features)
    
    # Predict using the trained model
    prediction = model.predict(features_scaled)
    
    # Return the prediction as a JSON response
    return {'prediction': prediction[0]}