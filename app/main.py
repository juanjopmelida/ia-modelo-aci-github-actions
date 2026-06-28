from pathlib import Path
from typing import Dict
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field 
  
  
MODEL_PATH = Path(__file__).parent / "model" / "iris_model.joblib"

model_package = joblib.load(MODEL_PATH)
model = model_package["model"]
target_names = model_package["target_names"]
features = model_package["features"]
accuracy = model_package["accuracy"] 
  
app = FastAPI( 
    title="API Modelo Iris", 
    description="Endpoint simple de IA desplegado con Docker, ACR, ACI y GitHub Actions",
    version="1.0.0" 
) 


class IrisRequest(BaseModel): 
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)   
  
@app.get("/")
def root():
    return {
        "message": "API del modelo Iris en funcionamiento",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    } 
  
  
@app.get("/health") 
def health():     return { 
        "status": "ok", 
        "model_loaded": True, 
        "model_accuracy": round(accuracy, 3), 
        "features": features 
    } 
  
  
@app.post("/predict")
def predict(input_data: IrisRequest):
    input_array = np.array([[
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width
    ]])
    prediction_id = int(model.predict(input_array)[0])
    probabilities = model.predict_proba(input_array)[0] 
  
    probability_by_class: Dict[str, float] = { 
        str(target_names[i]): round(float(probabilities[i]), 4)         for i in range(len(target_names)) 
    }   
    return { 
        "prediction_id": prediction_id, 
        "prediction_name": str(target_names[prediction_id]), 
        "probabilities": probability_by_class     } 
