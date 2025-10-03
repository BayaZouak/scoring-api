import pandas as pd
import numpy as np
import joblib
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import create_model

# Assurez-vous que le fichier 'utils.py' contenant 'preprocess_data' est bien présent
# Si 'preprocess_data' n'est pas utilisé ou est géré par le pipeline, cette ligne peut être commentée
from utils import preprocess_data 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import shap # Ajout de SHAP

# =============================================================================
# PARTIE 1 : Préparation
# =============================================================================

def clean_column_names(df):
    """Nettoie les noms de colonnes (remplace les caractères spéciaux)."""
    df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    return df

# On redéclare les composants pour éviter les erreurs avec joblib (bonne pratique)
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
# PARTIE 2 : Chargement du modèle et initialisation SHAP
# =============================================================================

BEST_THRESHOLD = 0.52

try:
    model_pipeline = joblib.load('modele_de_scoring.pkl')
    final_classifier = model_pipeline.steps[-1][1] # Récupérer le modèle final
    preprocessor = Pipeline(model_pipeline.steps[:-1]) # Récupérer le pré-traitement
    
    print("✅ Modèle chargé avec succès.")

    # --- Chargement des données de référence pour SHAP ---
    try:
        # Pour SHAP et l'extraction des noms de features
        df_ref = pd.read_csv('client_sample_dashboard.csv').drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
        X_ref_processed = preprocessor.transform(df_ref)
    except FileNotFoundError:
        raise RuntimeError("Un fichier de données de référence (client_sample_dashboard.csv) est requis pour SHAP.")

    # --- Détermination des noms des features post-traitement ---
    try:
        feature_names_full = preprocessor.get_feature_names_out().tolist()
        GLOBAL_FEATURE_NAMES = [name.split('__')[-1] for name in feature_names_full]
    except Exception:
        # Cas de repli si get_feature_names_out échoue
        GLOBAL_FEATURE_NAMES = [f"Feature_{i}" for i in range(X_ref_processed.shape[1])]
        print("Avertissement: Les noms de features post-traitement n'ont pu être extraits.")
    
    # --- Initialisation de l'Explainer SHAP ---
    # L'explainer est basé sur le modèle final et les données de référence prétraitées
    explainer = shap.TreeExplainer(final_classifier, X_ref_processed)
    
except FileNotFoundError:
    raise RuntimeError("❌ Le fichier 'modele_de_scoring.pkl' est introuvable.")
except Exception as e:
    raise RuntimeError(f"❌ Erreur lors du chargement ou initialisation : {e}")


# =============================================================================
# PARTIE 3 : Création dynamique du modèle Pydantic (colonnes optionnelles)
# =============================================================================

try:
    # Charge seulement la première ligne pour obtenir les types
    df_raw_template = pd.read_csv('application_train.csv', nrows=1)
    fields = {}

    for col in df_raw_template.columns:
        if col == 'SK_ID_CURR':
            fields[col] = (int, ...)  # SK_ID_CURR est obligatoire
        elif df_raw_template[col].dtype == np.int64:
            fields[col] = (Optional[int], None) # Optionnel = peut être absent ou None
        elif df_raw_template[col].dtype == np.float64:
            fields[col] = (Optional[float], None)
        elif df_raw_template[col].dtype == 'object':
            fields[col] = (Optional[str], None)

    if 'TARGET' in fields:
        del fields['TARGET']

    # Crée dynamiquement la classe Pydantic pour valider les données d'entrée
    ClientFeatures = create_model('ClientFeatures', **fields)

except FileNotFoundError:
    raise RuntimeError("❌ Le fichier 'application_train.csv' est requis pour générer le modèle Pydantic.")
except Exception as e:
    raise RuntimeError(f"❌ Erreur lors de la génération de la classe ClientFeatures : {e}")


# =============================================================================
# PARTIE 4 : Définition de l'API FastAPI
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
        # Convertir les données Pydantic en DataFrame
        df_raw = pd.DataFrame([client_data.model_dump()])
        df_raw = clean_column_names(df_raw)
        client_id = df_raw['SK_ID_CURR'].iloc[0]
        df_predict = df_raw.drop('SK_ID_CURR', axis=1, errors='ignore')

        # Prédiction (classe 1 = défaut)
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


@app.post("/explain") # Nouvel endpoint pour l'explicabilité
async def explain_client(client_data: ClientFeatures):
    try:
        df_raw = pd.DataFrame([client_data.model_dump()])
        df_raw = clean_column_names(df_raw)
        client_id = df_raw['SK_ID_CURR'].iloc[0]
        df_explain = df_raw.drop('SK_ID_CURR', axis=1, errors='ignore')
        
        # Pré-traitement des données du client
        X_client_processed = preprocessor.transform(df_explain)
        if hasattr(X_client_processed, 'toarray'):
            X_client_processed = X_client_processed.toarray()

        # Calcul des valeurs SHAP
        shap_values = explainer.shap_values(X_client_processed)
        
        # Gestion des modèles à deux classes (LightGBM/XGBoost)
        if isinstance(shap_values, list):
            client_shap_values = shap_values[1][0].tolist() # On prend les valeurs pour la classe 1 (défaut)
            base_value = explainer.expected_value[1]
        else: # Modèles à une seule sortie (régression ou autre)
            client_shap_values = shap_values[0].tolist()
            base_value = explainer.expected_value

        return {
            "SK_ID_CURR": int(client_id),
            "shap_values": client_shap_values,
            "base_value": float(base_value),
            "client_data_processed": X_client_processed[0].tolist(),
            "feature_names_processed": GLOBAL_FEATURE_NAMES
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Le calcul SHAP a échoué : {e}")
