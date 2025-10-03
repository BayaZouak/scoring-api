import pandas as pd
import numpy as np
import joblib
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import create_model

# IMPORTS EXISTANTS POUR LA RECONSTRUCTION DU PIPELINE
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


import utils 


import shap
from scipy.sparse import issparse
# ---------------------------------

# =============================================================================
# VOS FONCTIONS DE DÉCLARATION ET PRÉPARATION 
# =============================================================================

def clean_column_names(df):
    """Nettoie les noms de colonnes (inchangé)."""
    df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    return df

#  DÉCLARATIONS DE COMPOSANTS SKLEARN 
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
# PARTIE 2 : CHARGEMENT DU MODÈLE ET CALCUL SHAP GLOBAL 
# =============================================================================

BEST_THRESHOLD = 0.52
GLOBAL_IMPORTANCE = {} 

try:
    model_pipeline = joblib.load('modele_de_scoring.pkl')
    
    # Séparation du Pipeline pour l'explainer SHAP :
    preprocessor_pipeline = Pipeline(model_pipeline.steps[:-1])
    final_classifier = model_pipeline.steps[-1][1]

    # Données de référence pour l'explainer 
    df_ref = pd.read_csv('client_sample_dashboard.csv').drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
    X_ref_processed = preprocessor_pipeline.transform(df_ref)

    # Création de l'explainer SHAP (une seule fois au démarrage)
    explainer = shap.TreeExplainer(final_classifier, X_ref_processed)

    # Extraction des noms de features post-traitement pour les renvoyer
    try:
        feature_names_full = preprocessor_pipeline.get_feature_names_out().tolist()
        FEATURE_NAMES_PROCESSED = [name.split('__')[-1] for name in feature_names_full]
    except Exception:
        FEATURE_NAMES_PROCESSED = [f"Feature_{i}" for i in range(X_ref_processed.shape[1])]
    
    # --- CALCUL DE L'IMPORTANCE GLOBALE  ---
    # Calculer toutes les valeurs SHAP sur le jeu de référence.
    shap_values_ref = explainer.shap_values(X_ref_processed)
    
    # Prendre la classe cible (généralement 1 pour le défaut)
    if isinstance(shap_values_ref, list):
        shap_values_for_importance = shap_values_ref[1]
    else:
        shap_values_for_importance = shap_values_ref

    # Calculer la moyenne de l'amplitude (importance globale)
    global_importance_raw = np.mean(np.abs(shap_values_for_importance), axis=0)
    
    # Stocker le dictionnaire d'importance globale
    GLOBAL_IMPORTANCE = {
        name: float(importance)
        for name, importance in zip(FEATURE_NAMES_PROCESSED, global_importance_raw)
    }
    # ---------------------------------------------

    print("✅ Modèle, Explainer SHAP et Importance Globale chargés avec succès.")
except Exception as e:
    raise RuntimeError(f"❌ Erreur critique lors du chargement de modèle/SHAP : {e}")

# ... (Votre création du modèle Pydantic ClientFeatures inchangée) ...

# =============================================================================
# PARTIE 4 : Endpoints 
# =============================================================================

app = FastAPI(
    title="API de Scoring Prêt à Dépenser",
    description="API de prédiction de défaut de paiement pour les clients."
)

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de scoring de crédit. Utilisez /predict pour envoyer une prédiction."}

# importance globale
@app.get("/global_importance")
def get_global_importance():
    """Retourne l'importance globale des features (calculée avec SHAP)."""
    return GLOBAL_IMPORTANCE


@app.post("/predict")
async def predict_risk(client_data: ClientFeatures):
    try:
        df_raw = pd.DataFrame([client_data.model_dump()])
        df_raw = clean_column_names(df_raw)
        
        client_id = df_raw['SK_ID_CURR'].iloc[0]
        df_predict = df_raw.drop('SK_ID_CURR', axis=1, errors='ignore')

        # 1. Prédiction 
        probability = model_pipeline.predict_proba(df_predict)[:, 1][0]
        prediction_class = int(probability >= BEST_THRESHOLD)

        # 2. CALCUL SHAP LOCAL
        X_client_processed = preprocessor_pipeline.transform(df_predict) 
        shap_values = explainer.shap_values(X_client_processed)
        
        if isinstance(shap_values, list):
            client_shap_values = shap_values[1][0]
        else:
            client_shap_values = shap_values[0]

        client_shap_list = client_shap_values.tolist()
        
        # 3. Retour de la réponse enrichie
        return {
            "SK_ID_CURR": int(client_id),
            "probability": float(probability),
            "prediction": prediction_class,
            "decision_message": "Refusé (risque de défaut élevé)" if prediction_class == 1 else "Accepté (risque faible)",
            "shap_values": client_shap_list,             
            "shap_feature_names": FEATURE_NAMES_PROCESSED  
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de la requête: {e}")