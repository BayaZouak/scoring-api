import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from typing import Optional
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer 

BASE_API_URL = "https://scoring-api-latest.onrender.com" 
API_PREDICT_URL = f"{BASE_API_URL}/predict" 
API_EXPLAIN_URL = f"{BASE_API_URL}/explain" 
BEST_THRESHOLD = 0.52 
st.set_page_config(layout="wide", page_title="Dashboard Scoring Crédit")

@st.cache_data
def load_data():
    try:
        df_data = pd.read_csv('client_sample_dashboard.csv') 
        client_ids = df_data['SK_ID_CURR'].unique().tolist()
        sample_population_stats = {}
        cols_to_ignore = ['SK_ID_CURR', 'TARGET'] 
        
        for col in df_data.columns:
            if col in cols_to_ignore:
                continue
            
            dtype = df_data[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype) and df_data[col].nunique() > 10:
                sample_population_stats[col] = {'type': 'num'}
            elif pd.api.types.is_object_dtype(dtype) or df_data[col].nunique() <= 10:
                sample_population_stats[col] = {'type': 'cat'}
            
        full_population_stats = sample_population_stats
        
        return df_data, client_ids, full_population_stats
        
    except FileNotFoundError as e:
        st.error(f"Un fichier de données est manquant. Erreur: {e}")
        return pd.DataFrame(), [], {}

@st.cache_resource
def load_model_and_explainer():
    
    try:
        model_pipeline = joblib.load('modele_de_scoring.pkl')
        df_ref = pd.read_csv('client_sample_dashboard.csv').drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')

        preprocessor_pipeline = Pipeline(model_pipeline.steps[:-1])
        final_classifier = model_pipeline.steps[-1][1]
        
        X_ref_processed = preprocessor_pipeline.transform(df_ref)
        
        feature_names_raw = df_ref.columns.tolist() 

        try:
            feature_names_full = preprocessor_pipeline.get_feature_names_out().tolist()
            feature_names_processed = [name.split('__')[-1] for name in feature_names_full]
        except Exception:
             feature_names_processed = [f"Feature_{i}" for i in range(X_ref_processed.shape[1])]
        
        explainer = shap.TreeExplainer(final_classifier, X_ref_processed)
        
        return model_pipeline, explainer, preprocessor_pipeline, X_ref_processed, feature_names_processed, feature_names_raw
        
    except Exception as e:
        st.error(f"Erreur critique lors du chargement ou initialisation. Détail: {e}")
        return None, None, None, None, None, None

def create_gauge_chart(probability, threshold):
    confidence_score = (1 - probability) * 100
    confidence_threshold = (1 - threshold) * 100 
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Score de Confiance (100 = Risque Faible)", 'font': {'size': 18}},
        number={'suffix': "%", 'font': {'size': 48}},
        gauge = {
            'shape': "angular",
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, confidence_threshold], 'color': "red"}, 
                {'range': [confidence_threshold, 100], 'color': "green"} 
            ],
            'bar': {'color': 'black', 'thickness': 0.15}, 
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': confidence_threshold
            }}
    ))
    
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10)) 
    return fig

def get_api_result(client_features, endpoint="predict"):
    
    if endpoint == "predict":
        url = API_PREDICT_URL
    elif endpoint == "explain":
        url = API_EXPLAIN_URL
    else:
        st.error(f"Endpoint inconnu: {endpoint}")
        return None
        
    payload = {k: None if (pd.isna(v) or v == "") else v for k, v in client_features.items()}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion ou API indisponible. Détail: {e}")
        return None

df_data, client_ids, full_population_stats = load_data() 
model_pipeline, explainer, preprocessor_pipeline, X_ref_processed, feature_names_processed, feature_names_raw = load_model_and_explainer()

st.markdown("<style>.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center;'>
        <h1>Dashboard d'Analyse de Crédit</h1>
        <p>Outil d'aide à la décision pour l'octroi de prêts.</p>
    </div>
    """, 
    unsafe_allow_html=True
)

client_id = st.sidebar.selectbox(
    "Choisissez le SK_ID_CURR :",
    client_ids
)

client_data_raw = df_data[df_data['SK_ID_CURR'] == client_id].iloc[0].to_dict()
data_to_send = {'SK_ID_CURR': client_id}
edited_data = {}

if st.sidebar.button("Calculer le Score (API)", key="calculate_score_quick"):
    data_to_send.update({k: v for k, v in client_data_raw.items() if k not in ['SK_ID_CURR', 'TARGET']})
    
    api_result = get_api_result(data_to_send, endpoint="predict")
    
    if api_result:
        shap_result = get_api_result(data_to_send, endpoint="explain")
        
        if shap_result:
            st.session_state['api_result'] = api_result
            st.session_state['shap_result'] = shap_result
            st.session_state['current_client_data'] = data_to_send
            st.toast(f"Score pour le client {client_id} calculé!", icon='🚀')
            st.rerun()

with st.sidebar.form(key=f"form_{client_id}"):
    # Code de votre formulaire ici
    submit_button_mod = st.form_submit_button(label="🔄 Recalculer le Score (Après Modification)")

if submit_button_mod:
    data_to_send.update(edited_data)
    
    api_result = get_api_result(data_to_send, endpoint="predict")
    
    if api_result:
        shap_result = get_api_result(data_to_send, endpoint="explain")
        
        if shap_result:
            st.session_state['api_result'] = api_result
            st.session_state['shap_result'] = shap_result
            st.session_state['current_client_data'] = data_to_send
            st.toast(f"Score pour le client {client_id} (modifié) mis à jour!", icon='🔄')
            st.rerun()
        
if 'api_result' in st.session_state and st.session_state['api_result']['SK_ID_CURR'] == client_id:
    prob = st.session_state['api_result']['probability']
    decision = st.session_state['api_result']['prediction']
    message = st.session_state['api_result']['decision_message']
    current_data = st.session_state['current_client_data']
    
    # Votre code pour les sections 'Score' et 'Informations client'
    
    tab_explicability, tab_comparison = st.tabs(["Explication des Facteurs (SHAP)", "Comparaison aux Autres Clients"])

    with tab_explicability:
        
        col_radio, col_slider = st.columns([2, 1])
        
        with col_radio:
             explanation_type = st.radio(
                 "Type d'Analyse :",
                 ('Locale (Client)', 'Globale (Modèle)'),
                 horizontal=True,
                 key='exp_type'
             )
        
        with col_slider:
             if feature_names_processed is not None:
                 max_features_display = min(20, len(feature_names_processed)) 
                 num_features_to_display = st.slider(
                     "Nombre de variables à afficher :",
                     min_value=5,
                     max_value=max_features_display,
                     value=min(10, max_features_display),
                     step=1,
                     key='num_feat'
                 )
             else:
                 st.warning("Variables SHAP non disponibles.")
                 num_features_to_display = 10 
        
        if 'shap_result' in st.session_state:
            try:
                if explanation_type == 'Locale (Client)':
                    st.markdown("#### Explication Locale : Facteurs influençant le score du client sélectionné")
                    
                    shap_result = st.session_state['shap_result']
                    
                    e = shap.Explanation(
                        np.array(shap_result['shap_values']), 
                        shap_result['base_value'], 
                        data=np.array(shap_result['client_data_processed']), 
                        feature_names=shap_result['feature_names_processed']
                    )
                    
                    plt.rcParams.update({'figure.max_open_warning': 0})
                    fig_height = max(5, num_features_to_display * 0.5) 
                    fig, ax = plt.subplots(figsize=(15, fig_height))
                    shap.plots.waterfall(e, max_display=num_features_to_display, show=False)
                    st.pyplot(fig, use_container_width=True)
                    
                    st.caption(f"Le rouge pousse vers le défaut, le bleu diminue le risque. Affiche les {num_features_to_display} facteurs les plus importants.")

                elif explanation_type == 'Globale (Modèle)':
                    st.markdown("#### Explication Globale : Importance moyenne des variables pour le modèle")
                    
                    @st.cache_data
                    def get_global_shap_values(_explainer, X_ref_processed):
                        sample_indices = np.random.choice(X_ref_processed.shape[0], size=min(500, X_ref_processed.shape[0]), replace=False)
                        X_sample_for_global = X_ref_processed[sample_indices]
                        return _explainer.shap_values(X_sample_for_global)
                    
                    if explainer and X_ref_processed is not None:
                        global_shap_values = get_global_shap_values(explainer, X_ref_processed)
                        
                        if isinstance(global_shap_values, list):
                            shap_sum = np.abs(global_shap_values[1]).mean(axis=0) if len(global_shap_values) > 1 else np.abs(global_shap_values[0]).mean(axis=0) 
                        else:
                            shap_sum = np.abs(global_shap_values).mean(axis=0)
                        
                        importance_df = pd.DataFrame({
                            'Feature': feature_names_processed, 
                            'Importance': shap_sum
                        }).sort_values(by='Importance', ascending=False).head(num_features_to_display)

                        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', 
                                     title=f"Top {num_features_to_display} des Variables les Plus Importantes",
                                     color='Importance',
                                     color_continuous_scale=px.colors.sequential.Blues) 
                        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=max(500, num_features_to_display * 40))
                        
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True}) 
                        st.caption(f"Affiche les {num_features_to_display} variables qui ont, en moyenne, le plus grand impact sur la décision du modèle.")
                    else:
                        st.warning("Impossible de calculer le SHAP global.")

            except Exception as e:
                st.error(f"Échec de l'Explication SHAP. Détail: {e}")
        else:
            st.info("Veuillez calculer le score pour ce client pour afficher les graphiques SHAP.")

    with tab_comparison:
        # Votre code de l'onglet comparaison
        pass

else:
    st.info("Sélectionnez un client et cliquez sur **'Calculer le Score (API)'** dans la barre latérale pour démarrer l'analyse.")