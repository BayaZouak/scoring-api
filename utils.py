import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import issparse

def preprocess_data(df):
    """Fonction de feature engineering."""
    # 1. Gestion des valeurs extrêmes
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    
    # 2. Création de features (Feature Engineering)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    return df

def get_processed_feature_names(preprocessor_pipeline, raw_feature_names, X_ref_processed=None):
    """
    Tente de récupérer les noms des features après le ColumnTransformer.
    Utilise get_feature_names_out (moderne) ou un fallback.
    """
    
    # Extraire le ColumnTransformer s'il est encapsulé dans un Pipeline
    if isinstance(preprocessor_pipeline, Pipeline):
        try:
            ct = next(step[1] for step in preprocessor_pipeline.steps if isinstance(step[1], ColumnTransformer))
        except StopIteration:
            ct = preprocessor_pipeline.steps[-1][1] if isinstance(preprocessor_pipeline.steps[-1][1], ColumnTransformer) else None
    elif isinstance(preprocessor_pipeline, ColumnTransformer):
        ct = preprocessor_pipeline
    else:
        ct = None

    if ct:
        try:
            # Tenter la méthode standard (Scikit-learn >= 1.0)
            if hasattr(ct, 'get_feature_names_out'):
                # On utilise la méthode officielle avec les noms bruts pour l'entrée
                feature_names_full = ct.get_feature_names_out(raw_feature_names).tolist()
                # Nettoyer les préfixes (ex: 'num__', 'cat__')
                return [name.split('__')[-1] for name in feature_names_full]
        
        except Exception as e:
             print(f"Échec de l'extraction des noms de features par get_feature_names_out. Erreur: {e}")
    
    # Solution de secours 
    if X_ref_processed is not None:
        num_cols = X_ref_processed.shape[1]
    else:
        # Fallback générique si l'API ne fournit pas les noms, on suppose 200 
        num_cols = 200 
        
    print(f"ATTENTION: Utilisation des noms génériques: Feature_0 à Feature_{num_cols-1}")
    return [f"Feature_{i}" for i in range(num_cols)]