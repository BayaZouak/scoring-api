import pandas as pd
import numpy as np
import joblib
import uvicorn
import shap
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import create_model
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.sparse import issparse

# Importation de l'utilitaire créé
from utils import get_processed_feature_names, preprocess_data

# =============================================================================
# PARTIE 1 : Préparation & Recréation de la structure du pipeline
# =============================================================================
def clean_column_names(df):
    """Nettoie les noms de colonnes (remplace les caractères spéciaux)."""
    df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    return df

numerical_pipeline_recreate = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_pipeline_recreate = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor_recreate = ColumnTransformer(
    transformers=[('num', numerical_pipeline_recreate, []), ('cat', categorical_pipeline_recreate, [])],
    remainder='passthrough'
)

# =============================================================================
# PARTIE 2 : Chargement, Explainer et CALCUL GLOBAL
# =============================================================================

BEST_THRESHOLD = 0.52
# Variables globales pour le SHAP global
GLOBAL_SHAP_VALUES = None
GLOBAL_SHAP_BASE_VALUE = None
GLOBAL_X_PROCESSED = None
feature_names_processed = []
explainer = None
preprocessor_pipeline = None
model_pipeline = None

try:
    # --- 1. Chargement du Pipeline ---
    model_pipeline = joblib.load('modele_de_scoring.pkl')
    preprocessor_pipeline = Pipeline(model_pipeline.steps[:-1])
    final_classifier = model_pipeline.steps[-1][1]
    
    # --- 2. Préparation pour le SHAP Global et les Noms de Features ---
    # Utilisation de client_sample_dashboard.csv pour l'échantillon SHAP Global
    df_sample_raw = pd.read_csv('client_sample_dashboard.csv') 
    df_sample_raw = df_sample_raw.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
    
    df_sample_fe = preprocess_data(df_sample_raw.copy()) 
    X_sample_processed = preprocessor_pipeline.transform(df_sample_fe)

    # --- 3. Calcul des Noms de Features (POST-PRÉ-TRAITEMENT) ---
    raw_feature_names = df_sample_fe.columns.tolist()
    feature_names_processed = get_processed_feature_names(preprocessor_pipeline, raw_feature_names, X_sample_processed) 
    
    # --- 4. Création et Calcul de l'Explainer SHAP Global ---
    explainer = shap.TreeExplainer(final_classifier)
    
    shap_values_global = explainer.shap_values(X_sample_processed)
    
    # Extraction des valeurs pour la classe 1 (Défaut)
    if isinstance(shap_values_global, list):
        GLOBAL_SHAP_VALUES = np.array(shap_values_global[1])
        GLOBAL_SHAP_BASE_VALUE = explainer.expected_value[1]
    else:
        GLOBAL_SHAP_VALUES = np.array(shap_values_global)
        GLOBAL_SHAP_BASE_VALUE = explainer.expected_value
    
    # Stocker les données de référence transformées pour le Dot Plot Streamlit
    GLOBAL_X_PROCESSED = X_sample_processed

    print("✅ Modèle, SHAP Explainer et données globales chargés/calculés avec succès.")
    
except FileNotFoundError as e:
    raise RuntimeError(f"❌ Un fichier requis est introuvable : {e}. Assurez-vous d'avoir 'modele_de_scoring.pkl', 'application_train.csv' et 'client_sample_dashboard.csv'.")
except Exception as e:
    raise RuntimeError(f"❌ Erreur critique lors du chargement/initialisation : {e}")


# =============================================================================
# PARTIE 3 : Création dynamique du modèle Pydantic
# =============================================================================

try:
    df_raw_template = pd.read_csv('application_train.csv', nrows=1)
    df_template_full = preprocess_data(df_raw_template.copy()) 
    
    fields = {}

    for col in df_template_full.columns:
        if col == 'SK_ID_CURR':
            fields[col] = (int, ...)
        elif df_template_full[col].dtype == np.int64:
            fields[col] = (Optional[int], None)
        elif df_template_full[col].dtype == np.float64:
            fields[col] = (Optional[float], None)
        elif df_template_full[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_template_full[col].dtype):
            fields[col] = (Optional[str], None)

    if 'TARGET' in fields:
        del fields['TARGET']

    ClientFeatures = create_model('ClientFeatures', **fields)

except FileNotFoundError:
    raise RuntimeError("❌ Le fichier 'application_train.csv' est requis pour générer le modèle Pydantic.")
except Exception as e:
    raise RuntimeError(f"❌ Erreur lors de la génération de la classe ClientFeatures : {e}")


# =============================================================================
# PARTIE 4 : Définition de l'API - ENDPOINTS
# =============================================================================

app = FastAPI(
    title="API de Scoring Prêt à Dépenser",
    description="API de prédiction de défaut de paiement et d'explicabilité (SHAP)."
)

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de scoring de crédit. Utilisez /predict pour la prédiction et /global_shap pour l'analyse globale."}

@app.get("/global_shap")
async def get_global_shap():
    """Endpoint pour récupérer les valeurs SHAP globales pour le dashboard."""
    if GLOBAL_SHAP_VALUES is None or GLOBAL_X_PROCESSED is None:
        raise HTTPException(status_code=500, detail="Les données SHAP globales n'ont pas été calculées au démarrage de l'API.")
    
    X_processed_list = GLOBAL_X_PROCESSED.toarray().tolist() if issparse(GLOBAL_X_PROCESSED) else GLOBAL_X_PROCESSED.tolist()

    return {
        "global_shap_values": GLOBAL_SHAP_VALUES.tolist(),
        "global_x_processed": X_processed_list,
        "shap_features": feature_names_processed 
    }

@app.post("/predict")
async def predict_risk(client_data: ClientFeatures):
    try:
        df_raw = pd.DataFrame([client_data.model_dump()])
        df_raw = clean_column_names(df_raw)
        client_id = df_raw['SK_ID_CURR'].iloc[0]
        df_predict = df_raw.drop('SK_ID_CURR', axis=1, errors='ignore')

        # --- 1. Feature Engineering ---
        df_predict = preprocess_data(df_predict) 

        # --- 2. Scoring (Prédiction) ---
        probability = model_pipeline.predict_proba(df_predict)[:, 1][0]
        prediction_class = int(probability >= BEST_THRESHOLD)
        
        # --- 3. Calcul SHAP Côté API ---
        X_client_processed = preprocessor_pipeline.transform(df_predict)
        shap_values = explainer.shap_values(X_client_processed)
        
        # Extraction des valeurs pour la classe 1 (Défaut)
        if isinstance(shap_values, list):
            client_shap_values = shap_values[1][0]
            base_value = explainer.expected_value[1]
        else:
            client_shap_values = shap_values[0]
            base_value = explainer.expected_value
            
        # Conversion en listes natives pour l'envoi JSON
        shap_list = client_shap_values.tolist()
        
        # --- 4. Envoi de la Réponse (avec les données SHAP) ---
        return {
            "SK_ID_CURR": int(client_id),
            "probability": float(probability),
            "prediction": prediction_class,
            "decision_message": "Refusé (risque de défaut élevé)" if prediction_class == 1 else "Accepté (faible risque)",
            "shap_values": shap_list,
            "base_value": float(base_value),
            "shap_features": feature_names_processed 
        }

    except Exception as e:
        print(f"Erreur dans /predict: {e}")
        raise HTTPException(status_code=500, detail=f"❌ La prédiction a échoué : {e}")