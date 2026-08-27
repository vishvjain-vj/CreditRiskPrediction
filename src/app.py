from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Initialize the API
app = FastAPI(title="Credit Risk Underwriting API")

# Load the trained model into memory
model = joblib.load('../models/xgboost_credit_model.pkl')

# Define the expected JSON payload
class LoanApplication(BaseModel):
    # In a full production app, you would list all 120+ features explicitly.
    # For this architecture, we accept a dictionary of the borrower's financials.
    features: dict 

@app.post("/predict")
def predict_risk(application: LoanApplication):
    try:
        # Convert the incoming JSON payload into a 1-row Pandas DataFrame
        df = pd.DataFrame([application.features])
        df = df.fillna(0)
        df = df[model.feature_names_in_]
        # Run inference
        probability = model.predict_proba(df)[0][1]
        
        # Apply the optimal threshold we found (0.40)
        decision = "REJECT" if probability >= 0.40 else "APPROVE"
        
        return {
            "status": "success",
            "default_probability": round(float(probability), 4),
            "underwriting_decision": decision
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "Active", "message": "Credit Risk Underwriting API is live. Go to /docs to test."}
