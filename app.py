import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import joblib
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from typing import Optional

# --- Configuration Globale ---
API_URL = "https://scoring-api-latest.onrender.com/predict"
BEST_THRESHOLD = 0.52 
st.set_page_config(layout="wide", page_title="Dashboard Scoring Crédit")

# --- Fonctions de Chargement ---

@st.cache_data
def load_data():
    """Charge l'échantillon client et les statistiques de la population."""
    try:

        df_sample = pd.read_csv('client_sample_dashboard.csv') 
        client_ids = df_sample['SK_ID_CURR'].unique().tolist()
        
        with open('comparison_stats.json', 'r') as f:
            full_population_stats = json.load(f)
            
        return df_sample, client_ids, full_population_stats
        
    except FileNotFoundError as e:
        st.error(f"❌ Un fichier de données est manquant. Erreur: {e}")
        return pd.DataFrame(), [], {}

@st.cache_resource
def load_model_and_explainer():
    """Charge le modèle et initialise l'explainer SHAP (pour l'explication locale)."""
    try:
        model_pipeline = joblib.load('modele_de_scoring.pkl')
        # Charger les données de référence pour l'explainer
        df_ref = pd.read_csv('client_sample_dashboard.csv').drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')

        # 1. Extraire les composants du pipeline
        preprocessor_pipeline = Pipeline(model_pipeline.steps[:-1])
        final_classifier = model_pipeline.steps[-1][1]
        
        # 2. Prétraiter les données de fond
        X_ref_processed = preprocessor_pipeline.transform(df_ref)
        
        # 3. Initialiser l'explainer SHAP
        explainer = shap.TreeExplainer(final_classifier, X_ref_processed)
        
        st.success("✅ Modèle et Explainer SHAP chargés.")
        return model_pipeline, explainer, preprocessor_pipeline
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement ou initialisation SHAP. Vérifiez le modèle .pkl. Détail: {e}")
        return None, None, None

# --- Chargement ---
df_data, client_ids, full_population_stats = load_data() 
model_pipeline, explainer, preprocessor_pipeline = load_model_and_explainer()


# --- Fonction d'Appel de l'API ---
def get_prediction_from_api(client_features):
    """Appelle l'API Render pour obtenir le score."""
    # Convertir les champs vides de Streamlit en None pour correspondre à FastAPI/Pydantic
    payload = {k: None if (pd.isna(v) or v == "") else v for k, v in client_features.items()}
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erreur de connexion ou API indisponible. Vérifiez l'URL: {API_URL}. Détail: {e}")
        return None

# =============================================================================
# MISE EN PAGE STREAMLIT
# =============================================================================

st.title("💳 Dashboard d'Analyse de Crédit (Explicabilité Client)")

# --- Sélection Client et Modification ---
st.sidebar.header("🔍 Sélection et Modification Client")

client_id = st.sidebar.selectbox(
    "Sélectionnez le SK_ID_CURR du client :",
    client_ids
)

client_data_raw = df_data[df_data['SK_ID_CURR'] == client_id].iloc[0].to_dict()
data_to_send = {'SK_ID_CURR': client_id}
edited_data = {}

# --- Formulaire de Modification  ---
st.sidebar.markdown("### 📝 Modification des Données (API rafraîchie)")

with st.sidebar.form(key=f"form_{client_id}"):
    for feature, value in client_data_raw.items():
        if feature not in ['SK_ID_CURR', 'TARGET']:
            
            # Utilisation d'un widget générique pour les noms BRUTS
            input_val = st.text_input(f"{feature}", value=str(value) if pd.notna(value) else "", key=f"input_{feature}_{client_id}")

            # Tentative de conversion (pour s'assurer que les nombres sont envoyés comme des nombres)
            try:
                if input_val == "":
                    edited_data[feature] = np.nan
                elif '.' in input_val or 'e' in input_val.lower():
                    edited_data[feature] = float(input_val)
                else:
                    edited_data[feature] = int(input_val)
            except ValueError:
                edited_data[feature] = input_val
            
    submit_button = st.form_submit_button(label="📊 Calculer le Score (API)")

if submit_button:
    data_to_send.update(edited_data)
    api_result = get_prediction_from_api(data_to_send)
    
    if api_result:
        st.session_state['api_result'] = api_result
        st.session_state['current_client_data'] = data_to_send
        st.toast(f"Score pour le client {client_id} mis à jour!", icon='🚀')
        st.experimental_rerun()
        
        
# --- Affichage Principal ---
if 'api_result' in st.session_state and st.session_state['api_result']['SK_ID_CURR'] == client_id:
    result = st.session_state['api_result']
    prob = result['probability']
    decision = result['prediction']
    message = result['decision_message']
    current_data = st.session_state['current_client_data']

    st.header(f"Client: {client_id} | Statut: {message}")
    st.markdown("---")

    col_score, col_jauge, col_decision = st.columns([1, 2, 1])

    with col_score:
        st.metric(label="Probabilité de Défaut", value=f"{prob*100:.2f}%")
        st.info(f"Seuil Métier : {BEST_THRESHOLD*100:.2f}%")

    with col_jauge:
        score_color = "#FF4B4B" if prob >= BEST_THRESHOLD else "#008000"
        st.markdown(f"""
            <div style="background-color: #f0f2f6; border-radius: 5px; padding: 10px; text-align: center;">
                <h4 style="color: black;">Visualisation du Risque</h4>
                <div style="width: 100%; height: 20px; background-color: #ddd; border-radius: 10px; overflow: hidden;">
                    <div style="width: {prob*100}%; height: 100%; background-color: {score_color};"></div>
                </div>
                <p style="color: black;">{prob*100:.2f}% risque de défaut</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_decision:
        color = "red" if decision == 1 else "green"
        st.markdown(f"**Décision Finale :** <span style='color:{color}; font-size: 1.5em;'>{message}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # =============================================================================
    # 2. Explicabilité (SHAP Waterfall)
    # =============================================================================
    st.subheader("2. Explication Locale du Score (Facteurs de la Décision)")
    
    if explainer and preprocessor_pipeline:
        try:
            data_to_explain = st.session_state['current_client_data']
            df_client = pd.DataFrame([data_to_explain]).drop(columns=['SK_ID_CURR'], errors='ignore')
            
            X_client_processed = preprocessor_pipeline.transform(df_client) 
            shap_values = explainer.shap_values(X_client_processed)
            
            # Créer l'objet Explanation
            e = shap.Explanation(
                shap_values[1][0], 
                explainer.expected_value[1], 
                feature_names=df_client.columns.tolist() 
            )
            
            # Afficher le graphique
            plt.rcParams.update({'figure.max_open_warning': 0})
            fig, ax = plt.subplots(figsize=(12, 7)) 
            shap.plots.waterfall(e, max_display=10, show=False)
            st.pyplot(fig, use_container_width=True)
            
            st.caption("Le rouge augmente le risque (défaut), le bleu diminue le risque. Les noms de colonnes sont bruts.")

        except Exception as e:
            st.error(f"❌ Échec du calcul SHAP. Détail: {e}")

    st.markdown("---")
    
    # =============================================================================
    # 3. Comparaison et Positionnement
    # =============================================================================
    st.subheader("3. Comparaison aux Autres Clients")
    
    col_feat_1, col_feat_2 = st.columns(2)

    with col_feat_1:
        st.markdown("**Analyse Univariée : Client vs Population (Moyenne)**")
        
        
        features_to_compare = [col for col in full_population_stats.keys()]
        selected_feature = st.selectbox(
            "Choisissez la caractéristique à comparer :",
            features_to_compare,
            key='feature_uni'
        )
        
        # --- Graphique Univarié ---
        stats_ref = full_population_stats.get(selected_feature)
        client_val = current_data.get(selected_feature)

        if stats_ref and stats_ref['type'] == 'num' and pd.notna(client_val):
            
            # Affichage des métriques clés
            st.metric(label="Valeur Client", value=f"{client_val:,.2f}")
            st.metric(label="Moyenne Population", value=f"{stats_ref['mean']:,.2f}")
            
            
            st.warning("Pour un Box Plot précis, toutes les données sont nécessaires. Affichage des métriques clés ci-dessus.")
            
        elif stats_ref and stats_ref['type'] == 'cat':
             client_cat = current_data.get(selected_feature)
             st.markdown(f"**Valeur Client :** `{client_cat}`")
             st.info("Cette variable est catégorielle.")
             
        else:
            st.warning("Données indisponibles ou non numériques pour la comparaison.")


    with col_feat_2:
        st.markdown("**Analyse Bivariée : Échantillon Client**")
        
        feat_x = st.selectbox("Axe X :", features_to_compare, index=0, key='feat_x')
        feat_y = st.selectbox("Axe Y :", features_to_compare, index=1, key='feat_y')
        
        # Utilise l'échantillon léger df_data pour le scatter plot
        fig_biv = px.scatter(df_data, x=feat_x, y=feat_y, color='TARGET', 
                             title=f"Relation entre {feat_x} et {feat_y} (Échantillon)",
                             color_continuous_scale=px.colors.sequential.Sunset,
                             hover_data=['SK_ID_CURR'])
        
        # Mettre en évidence le client sélectionné
        client_x = current_data.get(feat_x)
        client_y = current_data.get(feat_y)
        
        if client_x is not None and client_y is not None:
            fig_biv.add_scatter(x=[client_x], y=[client_y], mode='markers', name='Client Actuel (Modifié)', 
                                marker=dict(color='red', size=15, symbol='star', line=dict(width=2, color='DarkRed')))

        st.plotly_chart(fig_biv, use_container_width=True)

else:
    st.info("Sélectionnez un client et cliquez sur 'Calculer le Score' dans la barre latérale pour démarrer l'analyse.")