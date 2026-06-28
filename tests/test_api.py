import pathlib
import sys 
  
from fastapi.testclient import TestClient 
  
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1] APP_DIR = ROOT_DIR / "app" sys.path.append(str(APP_DIR)) 
  
from app.main import app  # noqa: E402 
    
client = TestClient(app) 
    def test_health_endpoint():     response = client.get("/health")     assert response.status_code == 200     data = response.json()     assert data["status"] == "ok"     assert data["model_loaded"] is True 
    def test_predict_endpoint():     payload = { 
        "sepal_length": 5.1, 
        "sepal_width": 3.5, 
        "petal_length": 1.4, 
        "petal_width": 0.2 
    } 
    response = client.post("/predict", json=payload)     assert response.status_code == 200     data = response.json()     assert "prediction_name" in data     assert "probabilities" in data 
