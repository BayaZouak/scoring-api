import pandas as pd
import numpy as np
import joblib
import uvicorn
import shap 
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import create_model

# Importation de l'utilitaire créé pour récupérer les noms
from utils import get_processed_feature_names 
# Importations des composants scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# =============================================================================
# PARTIE 1 : Préparation & Recréation de la structure du pipeline
# =============================================================================

def clean_column_names(df):
    """Nettoie les noms de colonnes (remplace les caractères spéciaux)."""
    df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    return df

# On redéclare les composants pour la compatibilité
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
# PARTIE 2 : Chargement du modèle ET de l'Explainer SHAP
# =============================================================================

BEST_THRESHOLD = 0.52

try:
    # --- 1. Chargement du Pipeline ---
    model_pipeline = joblib.load('modele_de_scoring.pkl')
    preprocessor_pipeline = Pipeline(model_pipeline.steps[:-1])
    final_classifier = model_pipeline.steps[-1][1]
    
    # --- 2. Création de l'Explainer SHAP ---
    explainer = shap.TreeExplainer(final_classifier) 
    
    # --- 3. Calcul des Noms de Features (POST-PRÉ-TRAITEMENT) ---
    # Récupérer les noms de colonnes brutes 
    raw_feature_names = pd.read_csv('application_train.csv', nrows=1).drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore').columns.tolist()
    feature_names_processed = get_processed_feature_names(preprocessor_pipeline, raw_feature_names)
    
    print("✅ Modèle, SHAP Explainer et noms de features chargés/calculés avec succès.")
    
except FileNotFoundError as e:
    raise RuntimeError(f"❌ Un fichier requis est introuvable : {e}")
except Exception as e:
    raise RuntimeError(f"❌ Erreur critique lors du chargement/initialisation : {e}")


# =============================================================================
# PARTIE 3 : Création dynamique du modèle Pydantic (inchangée)
# =============================================================================

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


# =============================================================================
# PARTIE 4 : Définition de l'API - 
# =============================================================================

app = FastAPI(
    title="API de Scoring Prêt à Dépenser",
    description="API de prédiction de défaut de paiement et d'explicabilité (SHAP)."
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

        # --- 1. Scoring (Prédiction) ---
        probability = model_pipeline.predict_proba(df_predict)[:, 1][0]
        prediction_class = int(probability >= BEST_THRESHOLD)
        
        # --- 2. Calcul SHAP Côté API ---
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
        
        # --- 3. Envoi de la Réponse (avec les données SHAP) ---
        return {
            "SK_ID_CURR": int(client_id),
            "probability": float(probability),
            "prediction": prediction_class,
            "decision_message": "Refusé (risque de défaut élevé)" if prediction_class == 1 else "Accepté (faible risque)",
            "shap_values": shap_list,
            "base_value": float(base_value),
            "shap_features": feature_names_processed # Envoyé pour l'affichage dans Streamlit
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ La prédiction a échoué : {e}")