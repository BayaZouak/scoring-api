import pandas as pd
import numpy as np
import joblib
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import create_model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import shap
from scipy.sparse import issparse

BEST_THRESHOLD = 0.52

# --- Chargement modèle ---
try:
    model_pipeline = joblib.load('modele_de_scoring.pkl')
    print("✅ Modèle chargé avec succès.")
except Exception as e:
    raise RuntimeError(f"Erreur chargement modèle : {e}")

# --- Création dynamique Pydantic ---
try:
    df_template = pd.read_csv('application_train.csv', nrows=1)
    fields = {}
    for col in df_template.columns:
        if col == 'SK_ID_CURR':
            fields[col] = (int, ...)
        elif df_template[col].dtype == np.int64:
            fields[col] = (Optional[int], None)
        elif df_template[col].dtype == np.float64:
            fields[col] = (Optional[float], None)
        elif df_template[col].dtype == 'object':
            fields[col] = (Optional[str], None)
    if 'TARGET' in fields:
        del fields['TARGET']
    ClientFeatures = create_model('ClientFeatures', **fields)
except Exception as e:
    raise RuntimeError(f"Erreur création Pydantic : {e}")

app = FastAPI(title="API Scoring Crédit", description="Prédiction défaut + SHAP")

# --- Utils ---
def clean_column_names(df):
    df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    return df

def transform_data(pipeline, df):
    X_raw = df.copy()
    preprocessor = Pipeline(pipeline.steps[:-1])
    X_trans = preprocessor.transform(X_raw)
    # Noms des features
    try:
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [f.split("__")[-1] for f in feature_names]
    except Exception:
        feature_names = [f"Feature_{i}" for i in range(X_trans.shape[1])]
    if issparse(X_trans):
        X_trans = X_trans.toarray()
    return X_trans, feature_names

# --- Endpoints ---
@app.get("/")
def home():
    return {"message": "API de scoring. Endpoints : /predict, /shap, /shap/global"}

@app.post("/predict")
async def predict_risk(client_data: ClientFeatures):
    try:
        df = pd.DataFrame([client_data.model_dump()])
        df = clean_column_names(df)
        client_id = df['SK_ID_CURR'].iloc[0]
        X = df.drop('SK_ID_CURR', axis=1, errors='ignore')
        prob = model_pipeline.predict_proba(X)[:,1][0]
        pred_class = int(prob >= BEST_THRESHOLD)
        return {
            "SK_ID_CURR": int(client_id),
            "probability": float(prob),
            "prediction": pred_class,
            "decision_message": "Refusé" if pred_class==1 else "Accepté"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction : {e}")

@app.post("/shap")
async def shap_local(client_data: ClientFeatures):
    try:
        df = pd.DataFrame([client_data.model_dump()])
        df = clean_column_names(df)
        X_trans, feature_names = transform_data(model_pipeline, df)
        explainer = shap.TreeExplainer(model_pipeline.steps[-1][1])
        shap_vals = explainer.shap_values(X_trans)

        if isinstance(shap_vals, list):
            vals = shap_vals[1][0].tolist() if len(shap_vals)>1 else shap_vals[0][0].tolist()
            base = float(explainer.expected_value[1]) if len(shap_vals)>1 else float(explainer.expected_value[0])
        else:
            vals = shap_vals[0].tolist()
            base = float(explainer.expected_value)

        return {"shap_values": vals, "base_value": base, "feature_names": feature_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur SHAP local : {e}")

@app.get("/shap/global")
async def shap_global():
    try:
        df_ref = pd.read_csv("client_sample_dashboard.csv").drop(columns=['SK_ID_CURR','TARGET'], errors='ignore')
        X_trans, feature_names = transform_data(model_pipeline, df_ref)
        explainer = shap.TreeExplainer(model_pipeline.steps[-1][1])
        shap_vals = explainer.shap_values(X_trans)

        if isinstance(shap_vals, list):
            shap_mean = np.abs(shap_vals[1]).mean(axis=0) if len(shap_vals)>1 else np.abs(shap_vals[0]).mean(axis=0)
        else:
            shap_mean = np.abs(shap_vals).mean(axis=0)

        importance = pd.DataFrame({"feature": feature_names, "importance": shap_mean}).sort_values("importance", ascending=False)
        return importance.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur SHAP global : {e}")
