import pandas as pd
import numpy as np
import joblib
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import create_model

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import shap 
# from utils import preprocess_data

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
    final_classifier = model_pipeline.steps[-1][1] 
    preprocessor = Pipeline(model_pipeline.steps[:-1])
    
    try:
        df_ref = pd.read_csv('client_sample_dashboard.csv').drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
        X_ref_processed = preprocessor.transform(df_ref)
        
    except FileNotFoundError:
        raise RuntimeError("Un fichier de données de référence (client_sample_dashboard.csv) est requis.")

    try:
        feature_names_full = preprocessor.get_feature_names_out().tolist()
        GLOBAL_FEATURE_NAMES = [name.split('__')[-1] for name in feature_names_full]
    except Exception:
        GLOBAL_FEATURE_NAMES = [f"Feature_{i}" for i in range(X_ref_processed.shape[1])]
        print("Avertissement: Les noms de features post-traitement n'ont pu être extraits.")

    explainer = shap.TreeExplainer(final_classifier, X_ref_processed)
    
except FileNotFoundError:
    raise RuntimeError("Un fichier requis (modèle ou données) est introuvable.")
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement: {e}")


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
    raise RuntimeError("Le fichier 'application_train.csv' est requis pour générer le modèle Pydantic.")
except Exception as e:
    raise RuntimeError(f"Erreur lors de la génération de la classe ClientFeatures : {e}")


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
        raise HTTPException(status_code=500, detail=f"La prédiction a échoué : {e}")


@app.post("/explain")
async def explain_client(client_data: ClientFeatures):
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
            "feature_names_processed": GLOBAL_FEATURE_NAMES
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Le calcul SHAP a échoué : {e}")