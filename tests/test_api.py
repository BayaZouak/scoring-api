import pytest
import importlib
import sys, os

# Ajout du chemin du projet pour que Python trouve ton fichier
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

# fake fonctions pour éviter le gros CSV et le modèle
def fake_read_csv(*args, **kwargs):
    import pandas as pd
    return pd.DataFrame([{"SK_ID_CURR": 1, "AMT_INCOME_TOTAL": 50000.0}])

def fake_joblib_load(path):
    class FakeModel:
        def predict_proba(self, X):
            import numpy as np
            return np.array([[0.3, 0.7]])
    return FakeModel()

@pytest.fixture
def client(monkeypatch):
    import pandas
    import joblib

    monkeypatch.setattr(pandas, "read_csv", fake_read_csv)
    monkeypatch.setattr(joblib, "load", fake_joblib_load)

    module = importlib.import_module("Zouak_Baya_1_API_082025")
    app = module.app
    return TestClient(app)

def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Bienvenue" in response.json()["message"]

def test_predict(client):
    payload = {"SK_ID_CURR": 1, "AMT_INCOME_TOTAL": 50000.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert "prediction" in data
