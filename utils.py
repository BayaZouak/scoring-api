import numpy as np
import pandas as pd

def preprocess_data(df):
    """Fonction de feature engineering."""
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    return df

def get_processed_feature_names(preprocessor_pipeline, raw_feature_names, X_ref_processed=None):
    """
    Tente de récupérer les noms des features après le ColumnTransformer.
    Utilise get_feature_names_out (moderne) ou une logique manuelle.
    """
    feature_names_processed = []

    try:
        # Tenter la méthode standard (Scikit-learn >= 1.0)
        if hasattr(preprocessor_pipeline, 'get_feature_names_out'):
            feature_names_full = preprocessor_pipeline.get_feature_names_out().tolist()
            # Nettoyer les préfixes du ColumnTransformer (ex: 'num__', 'cat__')
            return [name.split('__')[-1] for name in feature_names_full]
        
        # Tenter la méthode manuelle (Similaire à votre ancien code Streamlit)
        if isinstance(preprocessor_pipeline, Pipeline):
            ct = next(step[1] for step in preprocessor_pipeline.steps if isinstance(step[1], ColumnTransformer))
        elif isinstance(preprocessor_pipeline, ColumnTransformer):
            ct = preprocessor_pipeline
        else:
            raise AttributeError("Le pré-processeur n'est ni un Pipeline ni un ColumnTransformer.")
        
        # Logique pour traverser les transformateurs
        for name, transformer, features in ct.transformers_:
            if name == 'remainder':
                if transformer == 'passthrough' and X_ref_processed is not None:
                    # Si 'passthrough', on doit déduire les colonnes restantes
                    if issparse(X_ref_processed):
                        num_cols = X_ref_processed.shape[1]
                    else:
                        num_cols = X_ref_processed.shape[1]
                    # Si cette partie est trop complexe, on peut la simplifier en forçant l'utilisation de get_feature_names_out
                    # Mais pour l'heure, simplifions en ne traitant que les features transformées.
                    
                    # 💡 Pour le "passthrough", il faut le jeu de données pour déduire ce qui n'a pas été transformé.
                    # Puisqu'on ne veut pas charger de données, concentrons-nous sur les transformateurs nommés :
                    pass # On va ignorer le remainder si on n'a pas de X_ref pour le moment

            elif transformer != 'drop':
                # Pour les transformateurs comme OneHotEncoder, récupérer les noms
                if hasattr(transformer, 'get_feature_names_out'):
                    names_out = transformer.get_feature_names_out(features)
                    feature_names_processed.extend([n.split('__')[-1] for n in names_out])
                else:
                    # Pour les simples scalers ou imputers, les noms de features restent les mêmes
                    if isinstance(features, list):
                        feature_names_processed.extend(features)
        
        return feature_names_processed

    except Exception as e:
        print(f"Échec de l'extraction des noms de features. Erreur: {e}")
        # Solution de secours : retourner une liste de noms génériques
        if X_ref_processed is not None and issparse(X_ref_processed):
            num_cols = X_ref_processed.shape[1]
        elif X_ref_processed is not None:
            num_cols = X_ref_processed.shape[1]
        else:
            num_cols = 200 # Valeur par défaut
            
        return [f"Feature_{i}" for i in range(num_cols)]