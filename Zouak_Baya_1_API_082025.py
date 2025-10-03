from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
import shap

# Importe ta fonction utils 
from utils import clean_column_names

app = FastAPI()

MODEL_PATH = "modele_de_scoring.pkl"
DATA_REF_PATH = "client_sample_dashboard.csv"

# Chargement modèle, données de référence
model_pipeline = joblib.load(MODEL_PATH)
df_ref = pd.read_csv(DATA_REF_PATH)
df_ref = clean_column_names(df_ref)
if 'SK_ID_CURR' in df_ref.columns:
    df_ref = df_ref.drop(columns=['SK_ID_CURR'], errors='ignore')
if 'TARGET' in df_ref.columns:
    df_ref = df_ref.drop(columns=['TARGET'], errors='ignore')


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    # Transforme en DataFrame et nettoie colonnes avec ta fonction utils
    dff = pd.DataFrame([data])
    dff = clean_column_names(dff)
    if 'SK_ID_CURR' in dff.columns:
        dff = dff.drop(columns=['SK_ID_CURR'], errors='ignore')
    if 'TARGET' in dff.columns:
        dff = dff.drop(columns=['TARGET'], errors='ignore')

    # Calcul score / prédictions
    processed_data = model_pipeline[:-1].transform(dff)
    y_prob = model_pipeline.predict_proba(dff)[0][1]
    y_pred = model_pipeline.predict(dff)[0]

    # SHAP Explainer initialisé sur df_ref prétraité
    explainer = shap.TreeExplainer(model_pipeline.steps[-1][1], model_pipeline[:-1].transform(df_ref))

    # SHAP local (client)
    shap_local_array = explainer.shap_values(processed_data)
    if isinstance(shap_local_array, list):
        shap_local = shap_local_array[1][0] if len(shap_local_array) > 1 else shap_local_array[0][0]
        base_value = explainer.expected_value[1] if len(shap_local_array) > 1 else explainer.expected_value[0]
    else:
        shap_local = shap_local_array[0]
        base_value = explainer.expected_value if not isinstance(explainer.expected_value, (np.ndarray, list)) else explainer.expected_value[0]

    # SHAP global (moyenne importance)
    shap_global_array = explainer.shap_values(model_pipeline[:-1].transform(df_ref))
    if isinstance(shap_global_array, list):
        shap_global_values = np.abs(shap_global_array[1]).mean(axis=0) if len(shap_global_array) > 1 else np.abs(shap_global_array[0]).mean(axis=0)
    else:
        shap_global_values = np.abs(shap_global_array).mean(axis=0)

    threshold = 0.52
    message = 'Prêt Approuvé' if y_pred == 0 else 'Prêt Refusé'

    # Retourne tout (score + shap + features + base_value)
    return JSONResponse(content={
        "SK_ID_CURR": data.get("SK_ID_CURR"),
        "probability": float(y_prob),
        "prediction": int(y_pred),
        "decision_message": message,
        "shap_local": [float(x) for x in shap_local],
        "shap_global": [float(x) for x in shap_global_values],
        "feature_names": list(df_ref.columns),
        "base_value": float(base_value)
    })
