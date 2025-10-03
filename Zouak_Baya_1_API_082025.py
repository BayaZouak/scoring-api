import pandas as pd
import numpy as np
import joblib
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import create_model

from utils import preprocess_data 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import shap

def clean_column_names(df):
    df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    return df

numerical_pipeline_recreate = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline_recreate = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor_recreate = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline_recreate, []),
        ('cat', categorical_pipeline_recreate, [])
    ],
    remainder='passthrough'
)

BEST_THRESHOLD = 0.52

try:
    model_pipeline = joblib.load('modele_de_scoring.pkl')
    print("✅ Modèle chargé avec succès.")
except FileNotFoundError:
    raise RuntimeError("❌ Le fichier 'modele_de_scoring.pkl' est introuvable.")
except Exception as e:
    raise RuntimeError(f"❌ Erreur lors du chargement du modèle : {e}")

try:
    df_raw_template = pd.read_csv('application_train.csv', nrows=1)
    fields = {}
    for col in df_raw_template.columns:
        if col == 'SK_ID_CURR':
            fields[col] = (int, ...)
        elif df_raw_template[col].dtype == np.int64:
            fields[col] = (Optional[int], None)
        elif df_raw_template[col].dtype == np.float64:
            fields[col] = (Optional[float], None)
        elif df_raw_template[col].dtype == 'object':
            fields[col] = (Optional[str], None)
    if 'TARGET' in fields:
        del fields['TARGET']
    ClientFeatures = create_model('ClientFeatures', **fields)
except FileNotFoundError:
    raise RuntimeError("❌ Le fichier 'application_train.csv' est requis pour générer le modèle Pydantic.")
except Exception as e:
    raise RuntimeError(f"❌ Erreur lors de la génération de la classe ClientFeatures : {e}")

app = FastAPI(
    title="API de Scoring Prêt à Dépenser",
    description="API de prédiction de défaut de paiement pour les clients."
)

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de scoring de crédit. Utilisez /predict, /shap ou /shap/global."}

@app.post("/predict")
async def predict_risk(client_data: ClientFeatures):
    try:
        df_raw = pd.DataFrame([client_data.model_dump()])
        df_raw = clean_column_names(df_raw)
        client_id = df_raw['SK_ID_CURR'].iloc[0]
        df_predict = df_raw.drop('SK_ID_CURR', axis=1, errors='ignore')

        probability = model_pipeline.predict_proba(df_predict)[:, 1][0]
        prediction_class = int(probability >= BEST_THRESHOLD)

        return {
            "SK_ID_CURR": int(client_id),
            "probability": float(probability),
            "prediction": prediction_class,
            "decision_message": "Refusé (risque de défaut élevé)" if prediction_class == 1 else "Accepté (faible risque)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ La prédiction a échoué : {e}")

@app.post("/shap")
async def shap_values(client_data: ClientFeatures):
    try:
        df_raw = pd.DataFrame([client_data.model_dump()])
        df_raw = clean_column_names(df_raw)
        df_predict = df_raw.drop('SK_ID_CURR', axis=1, errors='ignore')

        preprocessor_pipeline = Pipeline(model_pipeline.steps[:-1])
        final_classifier = model_pipeline.steps[-1][1]
        X_client_processed = preprocessor_pipeline.transform(df_predict)

        explainer = shap.TreeExplainer(final_classifier)
        shap_vals = explainer.shap_values(X_client_processed)

        if isinstance(shap_vals, list):
            client_shap = shap_vals[1][0].tolist() if len(shap_vals) > 1 else shap_vals[0][0].tolist()
            base_value = float(explainer.expected_value[1]) if len(shap_vals) > 1 else float(explainer.expected_value[0])
        else:
            client_shap = shap_vals[0].tolist()
            base_value = float(explainer.expected_value)

        return {
            "shap_values": client_shap,
            "base_value": base_value,
            "feature_names": df_predict.columns.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Calcul SHAP local échoué : {e}")

@app.get("/shap/global")
async def shap_global():
    try:
        df_ref = pd.read_csv("client_sample_dashboard.csv").drop(columns=['SK_ID_CURR','TARGET'], errors='ignore')
        preprocessor_pipeline = Pipeline(model_pipeline.steps[:-1])
        final_classifier = model_pipeline.steps[-1][1]
        X_ref_processed = preprocessor_pipeline.transform(df_ref)

        explainer = shap.TreeExplainer(final_classifier)
        shap_vals = explainer.shap_values(X_ref_processed)

        if isinstance(shap_vals, list):
            shap_sum = np.abs(shap_vals[1]).mean(axis=0) if len(shap_vals) > 1 else np.abs(shap_vals[0]).mean(axis=0)
        else:
            shap_sum = np.abs(shap_vals).mean(axis=0)

        importance = pd.DataFrame({
            "feature": df_ref.columns.tolist(),
            "importance": shap_sum
        }).sort_values(by="importance", ascending=False)

        return importance.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Calcul SHAP global échoué : {e}")
