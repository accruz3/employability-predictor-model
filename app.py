from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load the trained models and scaler
svr_model = joblib.load('svr_model.pkl')
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
    # Predict time to employment (SVR model)
    time_to_employment = svr_model.predict(features_scaled)[0]
    
    # Predict job title (SVM model)
    job_title = svm_model.predict(features_scaled)[0]
    
    # Return both predictions
    return {
        'predicted_time_to_employment': time_to_employment,
        'predicted_job_title': job_title
    }