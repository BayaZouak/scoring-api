from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import shap

app = Flask(__name__)

# Chargement du modèle et des données de référence
MODEL_PATH = "modele_de_scoring.pkl"
DATA_REF_PATH = "client_sample_dashboard.csv"
model_pipeline = joblib.load(MODEL_PATH)
df_ref = pd.read_csv(DATA_REF_PATH)
if 'SK_ID_CURR' in df_ref.columns:
    df_ref = df_ref.drop(columns=['SK_ID_CURR'], errors='ignore')
if 'TARGET' in df_ref.columns:
    df_ref = df_ref.drop(columns=['TARGET'], errors='ignore')

# pipeline pré-traitement
preprocessor_pipeline = model_pipeline.steps[:-1]
pipeline = model_pipeline

def get_feature_names_manually(pipeline, raw_feature_names):
    from sklearn.compose import ColumnTransformer
    feature_names_processed = []
    try:
        if isinstance(pipeline, ColumnTransformer):
            ct = pipeline
        else:
            ct = next(step[1] for step in pipeline if isinstance(step[1], ColumnTransformer))
        for name, transformer, features in ct.transformers_:
            if name == 'remainder' and transformer == 'passthrough':
                cols_used = set()
                for _, _, used_features in ct.transformers_:
                    if isinstance(used_features, list):
                        cols_used.update(used_features)
                remainder_cols = [col for col in raw_feature_names if col not in cols_used]
                feature_names_processed.extend(remainder_cols)
            elif transformer != 'drop':
                if hasattr(transformer, 'get_feature_names_out'):
                    names_out = transformer.get_feature_names_out(features)
                    feature_names_processed.extend([n.split('__')[-1] for n in names_out])
                else:
                    if isinstance(features, list):
                        feature_names_processed.extend(features)
        return feature_names_processed
    except Exception:
        return [f"Feature_{i}" for i in range(df_ref.shape[1])]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Conversion en DataFrame
    dff = pd.DataFrame([data])
    # Cas d'absence
    if 'SK_ID_CURR' in dff.columns: dff = dff.drop(columns=['SK_ID_CURR'], errors='ignore')
    if 'TARGET' in dff.columns: dff = dff.drop(columns=['TARGET'], errors='ignore')
    # Prétraitement
    preprocessor = joblib.load(MODEL_PATH).steps[:-1]
    processed_data = pipeline[:-1].transform(dff)
    y_prob = pipeline.predict_proba(dff)[0][1]
    y_pred = pipeline.predict(dff)[0]

    # Récupération des noms de features (après pré-traitement)
    feature_names_raw = df_ref.columns.tolist()
    try:
        feature_names_full = pipeline[:-1].get_feature_names_out().tolist()
        feature_names = [name.split('__')[-1] for name in feature_names_full]
    except Exception:
        feature_names = get_feature_names_manually(pipeline[:-1], feature_names_raw)

    # SHAP local
    explainer = shap.TreeExplainer(pipeline.steps[-1][1], pipeline[:-1].transform(df_ref))
    shap_local_array = explainer.shap_values(processed_data)
    # compatibilité : SHAP peut renvoyer une liste si modèle binaire
    if isinstance(shap_local_array, list):
        shap_local = shap_local_array[1][0] if len(shap_local_array) > 1 else shap_local_array[0][0]
        base_value = explainer.expected_value[1] if len(shap_local_array) > 1 else explainer.expected_value[0]
    else:
        shap_local = shap_local_array[0]
        base_value = explainer.expected_value if not isinstance(explainer.expected_value, (np.ndarray, list)) else explainer.expected_value[0]

    # SHAP global
    shap_global_array = explainer.shap_values(pipeline[:-1].transform(df_ref))
    # Si model binaire, prendre la classe 1
    if isinstance(shap_global_array, list):
        shap_global_values = np.abs(shap_global_array[1]).mean(axis=0) if len(shap_global_array) > 1 else np.abs(shap_global_array[0]).mean(axis=0)
    else:
        shap_global_values = np.abs(shap_global_array).mean(axis=0)

    # Décision message
    threshold = 0.52
    message = 'Prêt Approuvé' if y_pred == 0 else 'Prêt Refusé'

    return jsonify({
        "SK_ID_CURR": data.get("SK_ID_CURR"),
        "probability": float(y_prob),
        "prediction": int(y_pred),
        "decision_message": message,
        "shap_local": [float(x) for x in shap_local],
        "shap_global": [float(x) for x in shap_global_values],
        "feature_names": feature_names,
        "base_value": float(base_value)
    })

if __name__ == '__main__':
    app.run(debug=True)
