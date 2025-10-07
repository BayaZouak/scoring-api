import math
import pandas as pd
import numpy as np
import joblib
import uvicorn
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from pydantic import create_model, BaseModel

from utils import preprocess_data 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import shap
from scipy.sparse import issparse


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
    # Utilisé uniquement pour construire la classe Pydantic 
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
# PARTIE 3 bis : Outils SHAP (background + noms de features)
# =============================================================================

class SHAPLocalRequest(BaseModel):
    """Payload minimal pour expliquer localement un client (similaire à /predict)."""
    data: dict  

def _get_preprocessor_and_classifier():
    """Sépare le préprocesseur et le classifieur du pipeline."""
    preprocessor_pipeline = Pipeline(model_pipeline.steps[:-1])
    final_classifier = model_pipeline.steps[-1][1]
    return preprocessor_pipeline, final_classifier

def _get_background_matrix_and_feature_names():
    """
    Charge un échantillon 'client_sample_dashboard.csv' comme fond SHAP
    (ne pas utiliser application_train pour SHAP).
    Retourne X_background (transformé) et les noms de features post-traitement.
    """
    try:
        df_ref = pd.read_csv('client_sample_dashboard.csv').drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
    except Exception as e:
        raise RuntimeError(f"❌ Impossible de charger 'client_sample_dashboard.csv' pour SHAP : {e}")

    preprocessor_pipeline, _ = _get_preprocessor_and_classifier()
    X_ref_processed = preprocessor_pipeline.transform(df_ref)

    # Récupération des noms post-traitement
    feature_names_processed = None
    try:
        feature_names_full = preprocessor_pipeline.get_feature_names_out().tolist()
        feature_names_processed = [name.split('__')[-1] for name in feature_names_full]
    except Exception:
        try:
            # Récupération manuelle si get_feature_names_out n'est pas dispo
            if isinstance(preprocessor_pipeline, ColumnTransformer):
                ct = preprocessor_pipeline
            else:
                ct = next(step[1] for step in preprocessor_pipeline.steps if isinstance(step[1], ColumnTransformer))

            feature_names_processed = []
            for name, transformer, features in ct.transformers_:
                if name == 'remainder':
                    if transformer == 'passthrough':
                        cols_used = set()
                        for _, _, used_features in ct.transformers_:
                            if isinstance(used_features, list):
                                cols_used.update(used_features)
                        remainder_cols = [col for col in df_ref.columns if col not in cols_used]
                        feature_names_processed.extend(remainder_cols)
                elif transformer != 'drop':
                    if hasattr(transformer, 'get_feature_names_out'):
                        names_out = transformer.get_feature_names_out(features)
                        feature_names_processed.extend([n.split('__')[-1] for n in names_out])
                    else:
                        if isinstance(features, list):
                            feature_names_processed.extend(features)
            if not feature_names_processed:
                feature_names_processed = [f"Feature_{i}" for i in range(X_ref_processed.shape[1])]
        except Exception:
            feature_names_processed = [f"Feature_{i}" for i in range(X_ref_processed.shape[1])]

    return X_ref_processed, feature_names_processed, df_ref.columns.tolist()

def _build_explainer(X_background):
    """Construit un explainer SHAP approprié (TreeExplainer puis fallback)."""
    _, final_classifier = _get_preprocessor_and_classifier()
    try:
        explainer = shap.TreeExplainer(final_classifier, X_background)
    except Exception:
        try:
            explainer = shap.LinearExplainer(final_classifier, X_background)
        except Exception:
            explainer = shap.Explainer(final_classifier, X_background)
    return explainer

# Prépare l'arrière-plan SHAP une seule fois
try:
    X_BG, FEATURE_NAMES_PROCESSED, FEATURE_NAMES_RAW = _get_background_matrix_and_feature_names()
    SHAP_EXPLAINER = _build_explainer(X_BG)
    print("✅ Explainer SHAP initialisé.")
except Exception as e:
    print(f"SHAP indisponible : {e}")
    X_BG, FEATURE_NAMES_PROCESSED, FEATURE_NAMES_RAW, SHAP_EXPLAINER = None, None, None, None


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
# PARTIE 5 : Endpoints SHAP — calcul + sanitisation JSON
# =============================================================================

@app.post("/shap/local")
async def shap_local(req: SHAPLocalRequest):
    """
    Retourne :
    - base_value
    - shap_values (1D)
    - data_processed (1D)
    - feature_names_processed
    en garantissant un JSON sans NaN/Inf.
    """
    if SHAP_EXPLAINER is None or X_BG is None or FEATURE_NAMES_PROCESSED is None:
        raise HTTPException(status_code=500, detail="❌ SHAP indisponible sur le serveur.")

    try:
        # Transform client
        df_client = pd.DataFrame([req.data]).drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
        preprocessor_pipeline, _ = _get_preprocessor_and_classifier()
        X_client_processed = preprocessor_pipeline.transform(df_client)

        # Compute SHAP
        shap_values_all = SHAP_EXPLAINER.shap_values(X_client_processed)

        if isinstance(shap_values_all, list):
            shap_values_vec = shap_values_all[1][0] if len(shap_values_all) > 1 else shap_values_all[0][0]
            base_val = SHAP_EXPLAINER.expected_value[1] if len(shap_values_all) > 1 else SHAP_EXPLAINER.expected_value[0]
        else:
            shap_values_vec = shap_values_all[0]
            base_val = SHAP_EXPLAINER.expected_value if not isinstance(SHAP_EXPLAINER.expected_value, (np.ndarray, list)) else SHAP_EXPLAINER.expected_value[0]

        if issparse(X_client_processed):
            data_vec = X_client_processed.toarray()[0]
        else:
            data_vec = np.array(X_client_processed)[0]

        # ---- SANITIZE: remplace NaN/Inf -> valeurs JSON-compatibles ----
        def _clean_number(x, fallback=0.0):
            try:
                if x is None:
                    return None
                xf = float(x)
                if math.isnan(xf) or math.isinf(xf):
                    return float(fallback)
                return xf
            except Exception:
                return float(fallback)

        base_val_clean = _clean_number(base_val, 0.0)
        shap_values_clean = [_clean_number(v, 0.0) for v in shap_values_vec]
        data_processed_clean = [_clean_number(v, 0.0) for v in data_vec]

        payload = {
            "base_value": base_val_clean,
            "shap_values": shap_values_clean,
            "data_processed": data_processed_clean,
            "feature_names_processed": FEATURE_NAMES_PROCESSED
        }

        # Encode JSON-safe (sans NaN/Inf)
        return jsonable_encoder(payload)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Échec SHAP local : {e}")

@app.get("/shap/global")
async def shap_global(top_n: int = Query(20, ge=1, le=200)):
    """
    Retourne l'importance globale (moyenne des |SHAP|) sur un échantillon du fond,
    avec nettoyage JSON-safe.
    """
    if SHAP_EXPLAINER is None or X_BG is None or FEATURE_NAMES_PROCESSED is None:
        raise HTTPException(status_code=500, detail="❌ SHAP indisponible sur le serveur.")
    try:
        n = min(500, X_BG.shape[0])
        rng = np.random.default_rng(42)
        idx = rng.choice(X_BG.shape[0], size=n, replace=False)
        X_sample = X_BG[idx]

        shap_values_all = SHAP_EXPLAINER.shap_values(X_sample)
        if isinstance(shap_values_all, list):
            shap_abs_mean = np.abs(shap_values_all[1]).mean(axis=0) if len(shap_values_all) > 1 else np.abs(shap_values_all[0]).mean(axis=0)
        else:
            shap_abs_mean = np.abs(shap_values_all).mean(axis=0)

        # tri + top_n
        order = np.argsort(-shap_abs_mean)[:min(top_n, len(shap_abs_mean))]
        features = [FEATURE_NAMES_PROCESSED[i] for i in order]

        def _safe(v):
            try:
                vf = float(v)
                return 0.0 if (math.isnan(vf) or math.isinf(vf)) else vf
            except Exception:
                return 0.0

        importances = [_safe(shap_abs_mean[i]) for i in order]

        return jsonable_encoder({"features": features, "importances": importances})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Échec SHAP global : {e}")


# Point d’entrée local (optionnel)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)