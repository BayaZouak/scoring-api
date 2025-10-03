import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import create_model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import shap

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

def get_feature_names_from_column_transformer(ct: ColumnTransformer):
    """Récupère tous les noms des features après transformation"""
    feature_names = []
    for name, trans, cols in ct.transformers_:
        if name == 'remainder' and trans == 'passthrough':
            feature_names.extend(cols)
        elif hasattr(trans, 'get_feature_names_out'):
            try:
                names = trans.get_feature_names_out(cols)
                # Nettoyer les noms (num__col → col)
                names = [n.split("__")[-1] for n in names]
                feature_names.extend(names)
            except:
                feature_names.extend(cols)
        else:
            feature_names.extend(cols)
    return feature_names

def transform_and_get_features(df: pd.DataFrame):
    """Transforme les données via la pipeline et récupère les noms exacts des features"""
    preprocessor = model_pipeline[:-1]
    X_trans = preprocessor.transform(df)
    feature_names = get_feature_names_from_column_transformer(preprocessor)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    X_trans_df = pd.DataFrame(X_trans, columns=feature_names)
    return X_trans_df, feature_names

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
        X_trans_df, feature_names = transform_and_get_features(df)
        explainer = shap.TreeExplainer(model_pipeline.steps[-1][1])
        shap_vals = explainer.shap_values(X_trans_df)

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
        X_trans_df, feature_names = transform_and_get_features(df_ref)
        explainer = shap.TreeExplainer(model_pipeline.steps[-1][1])
        shap_vals = explainer.shap_values(X_trans_df)

        if isinstance(shap_vals, list):
            shap_mean = np.abs(shap_vals[1]).mean(axis=0) if len(shap_vals)>1 else np.abs(shap_vals[0]).mean(axis=0)
        else:
            shap_mean = np.abs(shap_vals).mean(axis=0)

        importance = pd.DataFrame({"feature": feature_names, "importance": shap_mean}).sort_values("importance", ascending=False)
        return importance.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur SHAP global : {e}")
