import pandas as pd
import numpy as np
import joblib
import uvicorn
import random
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import create_model
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import shap 

# =============================================================================
# PARTIE 1 : Préparation et Fonctions Utilitaires
# =============================================================================

def clean_column_names(df):
    """Nettoie les noms de colonnes (remplace les caractères spéciaux)."""
    df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    return df

# --- FONCTION D'EXTRACTION MANUELLE DES NOMS DE FEATURES (Correction de "Feature_0") ---
def get_feature_names_manually(preprocessor_pipeline, raw_feature_names, X_ref_processed):
    """Tente d'extraire les noms de features post-traitement à partir du ColumnTransformer."""
    feature_names_processed = []
    ct = None
    if isinstance(preprocessor_pipeline, ColumnTransformer):
        ct = preprocessor_pipeline
    else:
        for _, step in preprocessor_pipeline.steps:
            if isinstance(step, ColumnTransformer):
                ct = step
                break
    
    if ct is None:
        return [f"Feature_{i}" for i in range(X_ref_processed.shape[1])]

    for name, transformer, features in ct.transformers_:
        if name == 'remainder':
            if transformer == 'passthrough':
                cols_used = set()
                for _, _, used_features in ct.transformers_:
                    if isinstance(used_features, list):
                        cols_used.update(used_features)
                
                remainder_cols = [col for col in raw_feature_names if col not in cols_used and col != 'SK_ID_CURR']
                feature_names_processed.extend(remainder_cols)

        elif transformer != 'drop':
            if hasattr(transformer, 'get_feature_names_out'):
                names_out = transformer.get_feature_names_out(features)
                feature_names_processed.extend([n.split('__')[-1] for n in names_out])
            else:
                if isinstance(features, list):
                    feature_names_processed.extend(features)
            
    return feature_names_processed


# =============================================================================
# PARTIE 2 : Chargement du Modèle, Préparation des Données et Calcul Global
# =============================================================================

BEST_THRESHOLD = 0.52
MAX_SHAP_GLOBAL_SAMPLES = 500 # Limite de l'échantillon pour le calcul Global

try:
    model_pipeline = joblib.load('modele_de_scoring.pkl')
    final_classifier = model_pipeline.steps[-1][1]
    preprocessor = Pipeline(model_pipeline.steps[:-1])
    
    print("✅ Modèle chargé avec succès.")

    # --- Chargement des données de référence pour SHAP ---
    try:
        df_ref = pd.read_csv('client_sample_dashboard.csv').drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
        X_ref_processed_full = preprocessor.transform(df_ref)
        
        # Échantillonnage des données de référence pour l'Explainer SHAP
        if X_ref_processed_full.shape[0] > MAX_SHAP_GLOBAL_SAMPLES:
            sample_indices = random.sample(range(X_ref_processed_full.shape[0]), MAX_SHAP_GLOBAL_SAMPLES)
            X_ref_processed = X_ref_processed_full[sample_indices]
        else:
            X_ref_processed = X_ref_processed_full
            
        feature_names_raw = df_ref.columns.tolist() 
    except FileNotFoundError:
        raise RuntimeError("Un fichier de données de référence (client_sample_dashboard.csv) est requis pour SHAP.")

    # --- Détermination des noms des features post-traitement ---
    try:
        feature_names_full = preprocessor.get_feature_names_out().tolist()
        GLOBAL_FEATURE_NAMES = [name.split('__')[-1] for name in feature_names_full]
    except Exception:
        GLOBAL_FEATURE_NAMES = get_feature_names_manually(preprocessor, feature_names_raw, X_ref_processed)
    
    # --- Initialisation de l'Explainer SHAP ---
    explainer = shap.TreeExplainer(final_classifier, X_ref_processed)
    
    # --- Calcul du SHAP GLOBAL (Mise en cache) ---
    GLOBAL_SHAP_VALUES = explainer.shap_values(X_ref_processed)
    
    if isinstance(GLOBAL_SHAP_VALUES, list):
        # On prend les valeurs pour la classe 1 (défaut)
        GLOBAL_SHAP_SUM = np.abs(GLOBAL_SHAP_VALUES[1]).mean(axis=0) if len(GLOBAL_SHAP_VALUES) > 1 else np.abs(GLOBAL_SHAP_VALUES[0]).mean(axis=0)
    else:
        GLOBAL_SHAP_SUM = np.abs(GLOBAL_SHAP_VALUES).mean(axis=0)
    
    GLOBAL_SHAP_IMPORTANCE = pd.DataFrame({
        'Feature': GLOBAL_FEATURE_NAMES,
        'Importance': GLOBAL_SHAP_SUM
    }).sort_values(by='Importance', ascending=False).to_dict('records') # Format JSON simple

    print("✅ Calcul SHAP Global effectué et mis en cache.")
    
except FileNotFoundError:
    raise RuntimeError("❌ Le fichier 'modele_de_scoring.pkl' est introuvable.")
except Exception as e:
    raise RuntimeError(f"❌ Erreur lors du chargement ou initialisation : {e}")


# =============================================================================
# PARTIE 3 : Création dynamique du modèle Pydantic
# =============================================================================

# Le code ici suppose que vous avez un fichier application_train.csv pour le typage
try:
    df_raw_template = pd.read_csv('application_train.csv', nrows=1)
    fields = {}
    # ... (Logique de création du modèle Pydantic) ...
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


# =============================================================================
# PARTIE 4 : Définition de l'API FastAPI et Endpoints
# =============================================================================

app = FastAPI(
    title="API de Scoring Prêt à Dépenser",
    description="API de prédiction de défaut de paiement pour les clients."
)

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de scoring de crédit."}


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


@app.post("/explain")
async def explain_client(client_data: ClientFeatures):
    """Calcule les valeurs SHAP LOCALES pour un client donné."""
    try:
        df_raw = pd.DataFrame([client_data.model_dump()])
        df_raw = clean_column_names(df_raw)
        client_id = df_raw['SK_ID_CURR'].iloc[0]
        df_explain = df_raw.drop('SK_ID_CURR', axis=1, errors='ignore')
        
        X_client_processed = preprocessor.transform(df_explain)
        if hasattr(X_client_processed, 'toarray'):
            X_client_processed = X_client_processed.toarray()

        shap_values = explainer.shap_values(X_client_processed)
        
        if isinstance(shap_values, list):
            client_shap_values = shap_values[1][0].tolist() 
            base_value = explainer.expected_value[1]
        else: 
            client_shap_values = shap_values[0].tolist()
            base_value = explainer.expected_value

        return {
            "SK_ID_CURR": int(client_id),
            "shap_values": client_shap_values,
            "base_value": float(base_value),
            "client_data_processed": X_client_processed[0].tolist(),
            "feature_names_processed": GLOBAL_FEATURE_NAMES # Les noms de features sont ici!
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Le calcul SHAP a échoué : {e}")


@app.get("/explain_global", response_model=List[Dict[str, Any]])
async def explain_global():
    """Renvoie les importances SHAP GLOBALES (moyennes) mises en cache."""
    try:
        return GLOBAL_SHAP_IMPORTANCE
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Échec de la récupération de l'importance globale : {e}")
