import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import create_model
import numpy as np

# ---- PARTIE 1: Fonctions de Feature Engineering ----

def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def application_train_test(df_input, nan_as_category=False):
    df = df_input.copy()
    
    if 'CODE_GENDER' in df.columns:
        df = df[df['CODE_GENDER'] != 'XNA']
    
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        if bin_feature in df.columns:
            df[bin_feature], _ = pd.factorize(df[bin_feature])
    
    df, _ = one_hot_encoder(df, nan_as_category)
    
    if 'DAYS_EMPLOYED' in df.columns:
        df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = np.nan
    
    if 'DAYS_EMPLOYED' in df.columns and 'DAYS_BIRTH' in df.columns:
        df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    if 'AMT_INCOME_TOTAL' in df.columns and 'AMT_CREDIT' in df.columns:
        df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    if 'AMT_INCOME_TOTAL' in df.columns and 'CNT_FAM_MEMBERS' in df.columns:
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    if 'AMT_ANNUITY' in df.columns and 'AMT_CREDIT' in df.columns:
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    return df

def normalize_columns(df):
    df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
    return df

# ---- PARTIE 2: Chargement du modèle et préparation des colonnes ----

df_train = pd.read_csv('application_train.csv')

# Normalisation des colonnes comme dans le notebook
df_train = normalize_columns(df_train)

df_processed_train = application_train_test(df_train)

columns_to_drop = df_train.columns[df_train.isna().mean() > 0.5]
df_processed_train = df_processed_train.drop(columns=columns_to_drop, errors='ignore')

common_cols = [col for col in df_processed_train.columns if col not in ['TARGET', 'SK_ID_CURR']]

# Calcul des médianes pour imputation
median_values = df_processed_train[common_cols].median()

try:
    model = joblib.load('modele_de_scoring.pkl')
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None

SEUIL_METIER = 0.53

# ---- PARTIE 3: Création dynamique de la classe Pydantic ----

type_mapping = {
    'object': str,
    'int64': int,
    'float64': float,
    'int32': int,
    'float32': float
}

cols_to_use = [col for col in df_train.columns if col not in columns_to_drop and col != 'TARGET']
fields = {col: (type_mapping.get(str(df_train[col].dtype), str), ...) for col in cols_to_use}
ClientData = create_model('ClientData', **fields)

# ---- PARTIE 4: Définition de l'API FastAPI ----

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de scoring de crédit."}

@app.post("/predict")
def predict_credit_risk(data: ClientData):
    if model is None:
        return {"error": "Modèle non chargé. Vérifiez le fichier 'modele_de_scoring.pkl'."}

    try:
        df_raw = pd.DataFrame([data.model_dump()])

        # Normaliser les colonnes des données entrantes
        df_raw = normalize_columns(df_raw)

        # Appliquer le feature engineering
        df_transformed = application_train_test(df_raw)

        # Ajouter les colonnes manquantes avec valeur 0
        for col in common_cols:
            if col not in df_transformed.columns:
                df_transformed[col] = 0

        # S'assurer que les colonnes sont dans le bon ordre
        df_final = df_transformed[common_cols]

        # Imputation des valeurs manquantes avec la médiane
        df_final = df_final.fillna(median_values)

        # Prédiction
        probability = model.predict_proba(df_final.values)[:, 1][0]
        prediction = "Accepté" if probability < SEUIL_METIER else "Refusé"

        return {
            "prediction": prediction,
            "probability": float(probability),
            "seuil_metier": SEUIL_METIER
        }

    except Exception as e:
        return {
            "error": str(e),
            "message": "Erreur lors du traitement de la prédiction."
        }
