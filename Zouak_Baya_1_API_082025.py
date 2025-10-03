import pandas as pd
import numpy as np
import joblib
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import create_model

import shap
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


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
    return {"message": "Bienvenue sur l'API de scoring de crédit. Utilisez /predict, /shap_local et /shap_global."}


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
# PARTIE 5 : SHAP (local et global)
# =============================================================================

def get_transformed_data(df):
    """Passe les données par le preprocess du pipeline et récupère les features encodés."""
    preprocessor = model_pipeline[:-1]  # tout sauf le modèle final
    X_trans = preprocessor.transform(df)

    # Récupération des noms de features
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"Feature_{i}" for i in range(X_trans.shape[1])]

    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    return pd.DataFrame(X_trans, columns=feature_names)


# --- Pré-calcul SHAP global ---
try:
    df_global_ref = pd.read_csv("client_sample_dashboard.csv").drop(columns=["SK_ID_CURR","TARGET"], errors="ignore")
    df_global_ref = clean_column_names(df_global_ref)
    X_global = get_transformed_data(df_global_ref)

    explainer_global = shap.TreeExplainer(model_pipeline.steps[-1][1])
    shap_vals_global = explainer_global.shap_values(X_global)

    if isinstance(shap_vals_global, list):
        shap_mean_global = np.abs(shap_vals_global[1]).mean(axis=0) if len(shap_vals_global) > 1 else np.abs(shap_vals_global[0]).mean(axis=0)
    else:
        shap_mean_global = np.abs(shap_vals_global).mean(axis=0)

    shap_importance_global = pd.DataFrame({
        "feature": X_global.columns,
        "importance": shap_mean_global
    }).sort_values("importance", ascending=False).to_dict(orient="records")

    print("✅ SHAP global pré-calculé")

except Exception as e:
    raise RuntimeError(f"❌ Erreur lors de l'initialisation de SHAP global : {e}")


@app.get("/shap/global")
async def shap_global():
    return shap_importance_global


@app.post("/shap")
async def shap_local(client_data: ClientFeatures):
    try:
        df_raw = pd.DataFrame([client_data.model_dump()])
        df_raw = clean_column_names(df_raw)
        X_trans = get_transformed_data(df_raw.drop("SK_ID_CURR", axis=1, errors="ignore"))

        explainer = shap.TreeExplainer(model_pipeline.steps[-1][1])
        shap_vals = explainer.shap_values(X_trans)

        if isinstance(shap_vals, list):
            vals = shap_vals[1][0].tolist() if len(shap_vals) > 1 else shap_vals[0][0].tolist()
            base_value = float(explainer.expected_value[1]) if len(shap_vals) > 1 else float(explainer.expected_value[0])
        else:
            vals = shap_vals[0].tolist()
            base_value = float(explainer.expected_value)

        return {
            "shap_values": vals,
            "base_value": base_value,
            "feature_names": X_trans.columns.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Erreur SHAP local : {e}")
