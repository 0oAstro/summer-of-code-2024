from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
from typing import List

# Load the trained model pipeline
try:
    model_path = "random_forest_model.pkl"
    model_pipeline = joblib.load(model_path)
except FileNotFoundError:
    print(f"Error: Model file {model_path} not found.")
    model_pipeline = None
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API", 
    description="Fraud Prediction with V1-V28, Amount & Time"
)

# Pydantic model for input validation with enhanced features
class FraudFeatures(BaseModel):
    V1: float = Field(..., description="First anonymized feature")
    V2: float = Field(..., description="Second anonymized feature")
    V3: float = Field(..., description="Third anonymized feature")
    V4: float = Field(..., description="Fourth anonymized feature")
    V5: float = Field(..., description="Fifth anonymized feature")
    V6: float = Field(..., description="Sixth anonymized feature")
    V7: float = Field(..., description="Seventh anonymized feature")
    V8: float = Field(..., description="Eighth anonymized feature")
    V9: float = Field(..., description="Ninth anonymized feature")
    V10: float = Field(..., description="Tenth anonymized feature")
    V11: float = Field(..., description="Eleventh anonymized feature")
    V12: float = Field(..., description="Twelfth anonymized feature")
    V13: float = Field(..., description="Thirteenth anonymized feature")
    V14: float = Field(..., description="Fourteenth anonymized feature")
    V15: float = Field(..., description="Fifteenth anonymized feature")
    V16: float = Field(..., description="Sixteenth anonymized feature")
    V17: float = Field(..., description="Seventeenth anonymized feature")
    V18: float = Field(..., description="Eighteenth anonymized feature")
    V19: float = Field(..., description="Nineteenth anonymized feature")
    V20: float = Field(..., description="Twentieth anonymized feature")
    V21: float = Field(..., description="Twenty-first anonymized feature")
    V22: float = Field(..., description="Twenty-second anonymized feature")
    V23: float = Field(..., description="Twenty-third anonymized feature")
    V24: float = Field(..., description="Twenty-fourth anonymized feature")
    V25: float = Field(..., description="Twenty-fifth anonymized feature")
    V26: float = Field(..., description="Twenty-sixth anonymized feature")
    V27: float = Field(..., description="Twenty-seventh anonymized feature")
    V28: float = Field(..., description="Twenty-eighth anonymized feature")
    Amount: float = Field(..., description="Transaction amount", gt=0)
    Time: float = Field(..., description="Time of transaction")

    # Optional custom validators
    @validator('Amount')
    def validate_amount(cls, amount):
        if amount < 0:
            raise ValueError("Amount must be a positive number")
        return amount

@app.post("/predict/")
def predict_fraud(features: FraudFeatures):
    # Check if model is loaded
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Machine learning model not loaded")
    
    try:
        # Convert input to a numpy array
        input_data = np.array([[
            features.V1, features.V2, features.V3, features.V4, features.V5, 
            features.V6, features.V7, features.V8, features.V9, features.V10,
            features.V11, features.V12, features.V13, features.V14, features.V15, 
            features.V16, features.V17, features.V18, features.V19, features.V20,
            features.V21, features.V22, features.V23, features.V24, features.V25, 
            features.V26, features.V27, features.V28, features.Amount, features.Time
        ]])
        
        # Make prediction
        prediction = model_pipeline.predict(input_data)
        probability = model_pipeline.predict_proba(input_data)

        # Return response
        return {
            "prediction": int(prediction[0]),
            "fraud_probability": float(probability[0][1]),
            "fraud_confidence": float(probability[0][1]) * 100
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Health check endpoint
@app.get("/health/")
def health_check():
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Machine learning model not loaded")
    return {"status": "healthy", "model": f"Fraud Detection Model {model_path}"}

# Optional: Swagger UI will be available at /docs
# Optional: ReDoc will be available at /redoc