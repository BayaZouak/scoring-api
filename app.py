import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import shap # On garde SHAP pour le dessin des plots
import matplotlib.pyplot as plt

# --- Configuration Globale ---
# ⚠️ Mettez ici l'URL de votre API déployée
API_URL = "https://scoring-api-latest.onrender.com/predict" 
BEST_THRESHOLD = 0.52 
st.set_page_config(layout="wide", page_title="Dashboard Scoring Crédit")

# --- Fonctions de Chargement des Données (Locales) ---

@st.cache_data
def load_data():
    """Charge les données client et calcule les métadonnées pour la comparaison."""
    try:
        
        df_data = pd.read_csv('client_sample_dashboard.csv') 
        client_ids = df_data['SK_ID_CURR'].unique().tolist()
        
        # Calculer les métadonnées pour l'onglet de comparaison
        sample_population_stats = {}
        cols_to_ignore = ['SK_ID_CURR', 'TARGET'] 
        for col in df_data.columns:
            if col in cols_to_ignore: continue
            dtype = df_data[col].dtype
            if pd.api.types.is_numeric_dtype(dtype) and df_data[col].nunique() > 10:
                sample_population_stats[col] = {'type': 'num'}
            elif pd.api.types.is_object_dtype(dtype) or df_data[col].nunique() <= 10:
                sample_population_stats[col] = {'type': 'cat'}
        full_population_stats = sample_population_stats
        
        return df_data, client_ids, full_population_stats
        
    except FileNotFoundError as e:
        st.error(f"❌ Le fichier 'client_sample_dashboard.csv' est manquant. Erreur: {e}")
        return pd.DataFrame(), [], {}

# --- Chargement initial des données ---
df_data, client_ids, full_population_stats = load_data() 
feature_names_raw = df_data.columns.drop(['SK_ID_CURR', 'TARGET'], errors='ignore').tolist()

# --- Fonction de Jauge Plotly ---
def create_gauge_chart(probability, threshold):
    """Crée le graphique de jauge pour le score de confiance."""
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

# --- Fonction d'Appel de l'API ---
def get_prediction_from_api(client_features):
    """Appelle l'API et récupère le score, la décision et les valeurs SHAP."""
    # Convertit NaN/None en None pour l'envoi JSON
    payload = {k: None if (pd.isna(v) or v == "") else v for k, v in client_features.items()}
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erreur de connexion ou API indisponible. Détail: {e}")
        return None

# --- Fonctions de Visualisation de Comparaison ---

def plot_feature_distribution(df, client_value, feature_name, feature_type):
    """Affiche la distribution d'une variable pour la comparaison client/population."""
    fig = None
    
    # 1. Variables Numériques
    if feature_type == 'num':
        # Histogramme
        fig = px.histogram(df, x=feature_name, title=f"Distribution de {feature_name}")
        fig.add_vline(x=client_value, line_color="red", line_dash="dash", annotation_text="Client")
        fig.update_layout(showlegend=False, height=300)
    
    # 2. Variables Catégorielles
    elif feature_type == 'cat':
        # Diagramme en barres
        temp_df = df[feature_name].value_counts(normalize=True).reset_index()
        temp_df.columns = [feature_name, 'Proportion']
        fig = px.bar(temp_df, x=feature_name, y='Proportion', title=f"Proportion de {feature_name}")
        
        # Surligner la catégorie du client
        if pd.notna(client_value):
            for i, row in temp_df.iterrows():
                if str(row[feature_name]) == str(client_value):
                    fig.data[0].marker.color = ['red' if str(x) == str(client_value) else 'skyblue' for x in temp_df[feature_name]]
                    break
        fig.update_layout(height=300)
    
    return fig

# =============================================================================
# MISE EN PAGE STREAMLIT (Corps du Dashboard)
# =============================================================================

st.markdown("<style>.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# --- Barre latérale : Sélection et Formulaire ---
try:
    st.sidebar.image('logo_entreprise.png', use_container_width=True) 
except FileNotFoundError:
    st.sidebar.warning("⚠️ Logo non trouvé. Placez 'logo_entreprise.png' pour l'affichage.")
st.sidebar.markdown("---")
st.sidebar.header("🔍 Sélection Client")

client_id = st.sidebar.selectbox(
    "Choisissez le SK_ID_CURR :",
    client_ids
)
client_data_raw = df_data[df_data['SK_ID_CURR'] == client_id].iloc[0].to_dict()
data_to_send = {'SK_ID_CURR': client_id}
edited_data = {}

# Bouton de Score Rapide
if st.sidebar.button("Calculer le Score (API)", key="calculate_score_quick"):
    data_to_send.update({k: v for k, v in client_data_raw.items() if k not in ['SK_ID_CURR', 'TARGET']})
    api_result = get_prediction_from_api(data_to_send)
    if api_result:
        st.session_state['api_result'] = api_result
        st.session_state['current_client_data'] = data_to_send
        st.toast(f"Score pour le client {client_id} calculé!", icon='🚀')
        st.rerun()

# Formulaire de Modification 
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Modification des Données")

with st.sidebar.form(key=f"form_{client_id}"):
    st.markdown("Modifiez les variables pour simuler un nouveau score :")
    for feature in feature_names_raw:
        value = client_data_raw.get(feature)
        col_label, col_input = st.columns([1.5, 2])
        with col_input:
            input_val = st.text_input(f"{feature}", value=str(value) if pd.notna(value) else "", label_visibility="collapsed")
        with col_label:
            st.caption(f"{feature}")
        try:
            # Tente de convertir en float/int selon le besoin
            if input_val == "": edited_data[feature] = np.nan
            elif feature_names_raw and pd.api.types.is_numeric_dtype(df_data[feature].dtype):
                edited_data[feature] = float(input_val) if '.' in input_val else int(input_val)
            else:
                edited_data[feature] = input_val
        except ValueError:
            edited_data[feature] = input_val
            
    submit_button_mod = st.form_submit_button(label="🔄 Recalculer le Score (Après Modification)")

if submit_button_mod:
    data_to_send.update(edited_data)
    api_result = get_prediction_from_api(data_to_send)
    if api_result:
        st.session_state['api_result'] = api_result
        st.session_state['current_client_data'] = data_to_send
        st.toast(f"Score pour le client {client_id} (modifié) mis à jour!", icon='🔄')
        st.rerun()
        
# --- Affichage Principal ---
st.title("👨‍💼 Dashboard d'Aide à la Décision Crédit")

if 'api_result' in st.session_state and st.session_state['api_result']['SK_ID_CURR'] == client_id:
    result = st.session_state['api_result']
    prob = result['probability']
    decision = result['prediction']
    message = result['decision_message']
    current_data = st.session_state['current_client_data']

    st.markdown("---")
    
    # 1. Score et Jauge
    st.subheader("Score de Probabilité de Défaut et Confiance")
    col_score, col_jauge, col_decision = st.columns([1, 2, 1])
    with col_score:
        st.metric(label="Probabilité de Défaut", value=f"{prob*100:.2f}%")
        st.info(f"Seuil Métier : {BEST_THRESHOLD*100:.2f}%")
    with col_jauge:
        gauge_fig = create_gauge_chart(prob, BEST_THRESHOLD)
        st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': True}) 
    with col_decision:
        color = "red" if decision == 1 else "green"
        st.markdown(f"**Décision Finale :** <span style='color:{color}; font-size: 1.5em;'>{message}</span>", unsafe_allow_html=True)
        st.markdown(f"**Score de Confiance :** <span style='font-size: 1.5em;'>{(1-prob)*100:.2f}%</span>", unsafe_allow_html=True)
        
    st.markdown("---")

    # 2 & 3. Explicabilité et Comparaison
    tab_explicability, tab_comparison = st.tabs(["Explication des Facteurs (SHAP)", "Comparaison aux Autres Clients"])

    # --- CONTENU DE L'ONGLET 1 : EXPLICATION SHAP ---
    with tab_explicability:
        
        # --- Extraction des données SHAP de l'API ---
        try:
            client_shap_values = np.array(result['shap_values'])
            base_value = result['base_value']
            feature_names_processed_api = result['shap_features'] 
            shap_data_available = True
            
        except (KeyError, TypeError, Exception) as e:
            st.warning(f"⚠️ Les données SHAP n'ont pas été trouvées dans la réponse de l'API. Détail: {e}")
            shap_data_available = False
        
        if shap_data_available:
            
            col_radio, col_slider = st.columns([2, 1])
            
            with col_radio:
                st.radio(
                    "Type d'Analyse :",
                    ('Locale (Client)'), 
                    horizontal=True,
                    key='exp_type_api'
                )
                st.caption("L'analyse locale est effectuée par l'API.")
            
            with col_slider:
                max_features_display = min(20, len(feature_names_processed_api) if feature_names_processed_api else 10) 
                num_features_to_display = st.slider(
                    "Nombre de variables à afficher :",
                    min_value=5,
                    max_value=max_features_display,
                    value=min(10, max_features_display),
                    step=1,
                    key='num_feat_api'
                )
            
            st.markdown("#### Explication Locale : Facteurs influençant le score du client sélectionné")

            # --- Affichage du SHAP Waterfall Plot ---
            plt.rcParams.update({'figure.max_open_warning': 0})
            
            # Création de l'objet Explanation SHAP pour le plot
            e = shap.Explanation(
                client_shap_values, 
                base_value, 
                data=client_shap_values,
                feature_names=feature_names_processed_api
            )
            
            fig_height = max(5, num_features_to_display * 0.5) 
            fig, ax = plt.subplots(figsize=(15, fig_height))
            shap.plots.waterfall(e, max_display=num_features_to_display, show=False)
            st.pyplot(fig, use_container_width=True)
            
            st.caption("Le rouge pousse vers le défaut, le bleu diminue le risque. Valeurs calculées par l'API.")

        else:
            st.warning("Impossible de générer les graphiques SHAP.")

    # --- CONTENU DE L'ONGLET 2 : COMPARAISON ---
    with tab_comparison:
        st.subheader("Comparaison et Positionnement Client")
        st.info("Visualisez comment le client se positionne par rapport à la population de référence.")
        
        # Sélecteur de variable pour la comparaison
        comparison_feature = st.selectbox(
            "Choisissez une variable pour la comparaison :",
            options=feature_names_raw,
            key='comparison_feature'
        )
        
        if comparison_feature in full_population_stats:
            
            client_value = current_data.get(comparison_feature)
            feature_type = full_population_stats[comparison_feature]['type']
            
            st.markdown(f"#### Analyse de : **{comparison_feature}**")
            
            col_stat, col_plot = st.columns([1, 2])
            
            with col_stat:
                st.markdown("**Statistiques du Client :**")
                st.markdown(f"Valeur actuelle : **{client_value if pd.notna(client_value) else 'N/A'}**")
                
                # Ajout de statistiques de la population
                if feature_type == 'num':
                    st.markdown("**Statistiques de la Population :**")
                    st.metric(label="Moyenne", value=f"{df_data[comparison_feature].mean():.2f}")
                    st.metric(label="Médiane", value=f"{df_data[comparison_feature].median():.2f}")

            with col_plot:
                fig = plot_feature_distribution(df_data, client_value, comparison_feature, feature_type)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Impossible de générer le graphique pour cette variable.")

else:
    st.info("Sélectionnez un client et cliquez sur **'Calculer le Score (API)'** dans la barre latérale pour démarrer l'analyse.")