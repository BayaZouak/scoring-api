import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import create_model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import shap
from scipy.sparse import issparse

BEST_THRESHOLD = 0.52

# --- Charger pipeline complet ---
try:
    model_pipeline = joblib.load("modele_de_scoring.pkl")
    print("✅ Modèle chargé")
except Exception as e:
    raise RuntimeError(f"Erreur chargement modèle : {e}")

# --- Créer Pydantic dynamique ---
try:
    df_template = pd.read_csv("application_train.csv", nrows=1)
    fields = {}
    for col in df_template.columns:
        if col == "SK_ID_CURR":
            fields[col] = (int, ...)
        elif df_template[col].dtype == np.int64:
            fields[col] = (Optional[int], None)
        elif df_template[col].dtype == np.float64:
            fields[col] = (Optional[float], None)
        else:
            fields[col] = (Optional[str], None)
    if "TARGET" in fields:
        del fields["TARGET"]
    ClientFeatures = create_model("ClientFeatures", **fields)
except Exception as e:
    raise RuntimeError(f"Erreur création Pydantic : {e}")

app = FastAPI(title="API Scoring Crédit")

# --- Utils ---
def clean_column_names(df):
    df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    return df

def get_feature_names(preprocessor, raw_feature_names):
    """Extraire les noms de features post-prétraitement"""
    feature_names = []
    try:
        if isinstance(preprocessor, ColumnTransformer):
            ct = preprocessor
        else:
            ct = next(step[1] for step in preprocessor.steps if isinstance(step[1], ColumnTransformer))
        for name, transformer, cols in ct.transformers_:
            if transformer == 'drop':
                continue
            if transformer == 'passthrough':
                feature_names.extend(cols)
            elif hasattr(transformer, 'get_feature_names_out'):
                names_out = transformer.get_feature_names_out(cols)
                feature_names.extend([n.split("__")[-1] for n in names_out])
            else:
                feature_names.extend(cols if isinstance(cols, list) else [cols])
        return feature_names
    except Exception:
        return [f"Feature_{i}" for i in range(len(raw_feature_names))]

def transform_data(pipeline, df):
    """Transforme les données et retourne array + noms de features"""
    preprocessor = Pipeline(pipeline.steps[:-1])
    X_trans = preprocessor.transform(df)
    feature_names = get_feature_names(preprocessor, df.columns.tolist())
    if issparse(X_trans):
        X_trans = X_trans.toarray()
    return X_trans, feature_names, preprocessor

# --- Endpoints ---
@app.get("/")
def home():
    return {"message": "API de scoring. /predict, /shap, /shap/global"}

@app.post("/predict")
async def predict_risk(client_data: ClientFeatures):
    try:
        df = pd.DataFrame([client_data.model_dump()])
        df = clean_column_names(df)
        client_id = df["SK_ID_CURR"].iloc[0]
        X = df.drop("SK_ID_CURR", axis=1, errors="ignore")
        prob = model_pipeline.predict_proba(X)[:,1][0]
        pred_class = int(prob >= BEST_THRESHOLD)
        return {
            "SK_ID_CURR": int(client_id),
            "probability": float(prob),
            "prediction": pred_class,
            "decision_message": "Refusé" if pred_class == 1 else "Accepté"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction : {e}")

@app.post("/shap")
async def shap_local(client_data: ClientFeatures):
    try:
        df = pd.DataFrame([client_data.model_dump()])
        df = clean_column_names(df)
        X_trans, feature_names, preprocessor = transform_data(model_pipeline, df)
        explainer = shap.TreeExplainer(model_pipeline.steps[-1][1])
        shap_vals = explainer.shap_values(X_trans)

        if isinstance(shap_vals, list):
            vals = shap_vals[1][0].tolist() if len(shap_vals) > 1 else shap_vals[0][0].tolist()
            base = float(explainer.expected_value[1]) if len(shap_vals) > 1 else float(explainer.expected_value[0])
        else:
            vals = shap_vals[0].tolist()
            base = float(explainer.expected_value)

        return {"shap_values": vals, "base_value": base, "feature_names": feature_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur SHAP local : {e}")

@app.get("/shap/global")
async def shap_global():
    try:
        df_ref = pd.read_csv("client_sample_dashboard.csv").drop(columns=["SK_ID_CURR","TARGET"], errors="ignore")
        X_trans, feature_names, preprocessor = transform_data(model_pipeline, df_ref)
        explainer = shap.TreeExplainer(model_pipeline.steps[-1][1])

        # Sample pour ne pas saturer
        sample_indices = np.random.choice(X_trans.shape[0], size=min(500, X_trans.shape[0]), replace=False)
        X_sample = X_trans[sample_indices]

        shap_vals = explainer.shap_values(X_sample)

        if isinstance(shap_vals, list):
            shap_mean = np.abs(shap_vals[1]).mean(axis=0) if len(shap_vals) > 1 else np.abs(shap_vals[0]).mean(axis=0)
        else:
            shap_mean = np.abs(shap_vals).mean(axis=0)

        importance = pd.DataFrame({"feature": feature_names, "importance": shap_mean}).sort_values("importance", ascending=False)
        return importance.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur SHAP global : {e}")
