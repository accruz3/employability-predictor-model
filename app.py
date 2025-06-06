from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

svr_model = joblib.load('svr_model.pkl')
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
scaler_reg = joblib.load('scaler_reg.pkl')
label_encoder = joblib.load('label_encoder.pkl')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
 
@app.post('/predict')
def predict(request: PredictionRequest):
    # classification
    model_input_class = np.array([
        request.WebDevGrade,
        request.FundamentalsProgGrade, 
        request.FoundationsCSGrade,
        request.ExtracurricularsLevel,
        request.LatinHonors
    ]).reshape(1, -1)

    features_scaled_class = scaler.transform(model_input_class)
    job_title_encoded = svm_model.predict(features_scaled_class)[0]
    job_title = label_encoder.inverse_transform([job_title_encoded])[0]

    # regression
    academic_grades = np.array([
        request.WebDevGrade, 
        request.DSAGrade,
        request.FundamentalsProgGrade,
        request.OOPGrade,
        request.FoundationsCSGrade,
        request.NetworkingGrade, 
        request.NumericComputationGrade
    ])

    academic_grade = np.mean(academic_grades)

    model_input_reg = np.array([
        academic_grade,
        request.PracticumGrade,
        request.ExtracurricularsLevel,
        request.LatinHonors
    ]).reshape(1, -1)

    features_scaled_reg = scaler_reg.transform(model_input_reg)
    time_to_employment = svr_model.predict(features_scaled_reg)[0]
    
    return {
        'predicted_time_to_employment': time_to_employment,
        'predicted_job_title': job_title
    }