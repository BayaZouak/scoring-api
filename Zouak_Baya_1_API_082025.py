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
        ('cat', categorical_pipeline_recreate, [])
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
    df_raw_template = pd.read_csv('application_train.csv', nrows=1)
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
    raise RuntimeError("❌ Le fichier 'application_train.csv' est requis pour générer le modèle Pydantic.")
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
# PARTIE 5 : Préparer l'explainer SHAP
# =============================================================================

try:
    # Charger l'échantillon de référence complet pour SHAP
    df_shap_ref = pd.read_csv('client_sample_dashboard.csv')
    df_shap_ref = df_shap_ref.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')

    # Remplir les valeurs manquantes avec la médiane (comme durant la modélisation)
    df_shap_ref_filled = df_shap_ref.fillna(df_shap_ref.median())

    # Séparer préprocesseur et modèle final
    preprocessor_pipeline = Pipeline(model_pipeline.steps[:-1])
    final_classifier = model_pipeline.steps[-1][1]

    # Appliquer le préprocesseur
    X_ref_processed = preprocessor_pipeline.transform(df_shap_ref_filled)

    # Créer l'explainer SHAP
    explainer = shap.TreeExplainer(final_classifier, X_ref_processed)

except FileNotFoundError:
    raise RuntimeError("❌ Le fichier 'client_sample_dashboard.csv' est requis pour SHAP.")
except Exception as e:
    raise RuntimeError(f"❌ Erreur lors de l'initialisation de SHAP : {e}")

# =============================================================================
# PARTIE 6 : Préparer les fonctions pour SHAP dans l'API
# =============================================================================

def prepare_client_for_shap(client_df):
    """
    Prépare un DataFrame client pour SHAP :
    - complète les valeurs manquantes avec la médiane de l'échantillon de référence
    - applique le préprocesseur
    """
    client_filled = client_df.fillna(df_shap_ref.median())
    X_client_processed = preprocessor_pipeline.transform(client_filled)
    return X_client_processed

@app.post("/shap")
async def shap_local(client_data: ClientFeatures, top_features: int = 20):
    """
    Retourne les valeurs SHAP locales pour un client donné.
    """
    try:
        df_client = pd.DataFrame([client_data.model_dump()])
        df_client = clean_column_names(df_client)
        client_id = df_client['SK_ID_CURR'].iloc[0]

        X_client_processed = prepare_client_for_shap(df_client)

        shap_values = explainer.shap_values(X_client_processed)
        if isinstance(shap_values, list):
            client_shap = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            base_value = explainer.expected_value[1] if len(shap_values) > 1 else explainer.expected_value[0]
        else:
            client_shap = shap_values[0]
            base_value = explainer.expected_value

        feature_names = df_shap_ref.columns.tolist()

        return {
            "SK_ID_CURR": int(client_id),
            "shap_values": client_shap.tolist(),
            "feature_names": feature_names,
            "base_value": float(base_value)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Échec du calcul SHAP : {e}")
