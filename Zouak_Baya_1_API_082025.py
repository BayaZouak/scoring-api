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


# =============================================================================
# PARTIE 1 : Préparation
# =============================================================================

def clean_column_names(df):
    """Nettoie les noms de colonnes (remplace les caractères spéciaux)."""
    df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    return df

# On redéclare les composants pour éviter les erreurs avec joblib
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
        ('cat', categorical_pipeline_recreate, []),
    ],
    remainder='passthrough'
)


# =============================================================================
# PARTIE 2 : Chargement du modèle
# =============================================================================

BEST_THRESHOLD = 0.52

try:
    model_pipeline = joblib.load('modele_de_scoring.pkl')
    print("✅ Modèle chargé avec succès.")
except FileNotFoundError:
    raise RuntimeError("❌ Le fichier 'modele_de_scoring.pkl' est introuvable.")
except Exception as e:
    raise RuntimeError(f"❌ Erreur lors du chargement du modèle : {e}")


# =============================================================================
# PARTIE 3 : Création dynamique du modèle Pydantic (colonnes optionnelles)
# =============================================================================

try:
    df_raw_template = pd.read_csv('client_sample_dashboard.csv', nrows=1)  # ⚠️ remplacé par client_sample_dashboard
    fields = {}

    for col in df_raw_template.columns:
        if col == 'SK_ID_CURR':
            fields[col] = (int, ...)  # obligatoire
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
    raise RuntimeError("❌ Le fichier 'client_sample_dashboard.csv' est requis pour générer le modèle Pydantic.")
except Exception as e:
    raise RuntimeError(f"❌ Erreur lors de la génération de la classe ClientFeatures : {e}")


# =============================================================================
# PARTIE 4 : Définition de l'API
# =============================================================================

app = FastAPI(
    title="API de Scoring Prêt à Dépenser",
    description="API de prédiction de défaut de paiement pour les clients."
)

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de scoring de crédit. Utilisez /predict pour envoyer une prédiction."}


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


# =============================================================================
# PARTIE 5 : Endpoints SHAP
# =============================================================================

@app.post("/shap")
async def shap_local(client_data: ClientFeatures):
    """
    Retourne les valeurs SHAP pour un client (explication locale).
    """
    try:
        df_raw = pd.DataFrame([client_data.model_dump()])
        df_raw = clean_column_names(df_raw)
        df_predict = df_raw.drop('SK_ID_CURR', axis=1, errors='ignore')

        # Calcul SHAP
        explainer = shap.Explainer(model_pipeline.named_steps['classifier'], model_pipeline.named_steps['preprocessor'].transform(df_predict))
        shap_values = explainer(df_predict)

        return {
            "shap_values": shap_values.values[0].tolist(),
            "base_value": float(shap_values.base_values[0]),
            "feature_names": list(df_predict.columns)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Erreur SHAP locale : {e}")


@app.get("/shap/global")
async def shap_global():
    """
    Retourne les valeurs SHAP globales (moyenne absolue sur un échantillon).
    """
    try:
        df_sample = pd.read_csv("client_sample_dashboard.csv").drop(columns=['SK_ID_CURR', 'TARGET'], errors="ignore")
        df_sample = clean_column_names(df_sample)

        # On prend un petit échantillon pour accélérer
        df_sample = df_sample.sample(n=500, random_state=42)

        explainer = shap.Explainer(model_pipeline.named_steps['classifier'], model_pipeline.named_steps['preprocessor'].transform(df_sample))
        shap_values = explainer(df_sample)

        return {
            "shap_values": shap_values.values.tolist(),
            "feature_names": list(df_sample.columns)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Erreur SHAP globale : {e}")
