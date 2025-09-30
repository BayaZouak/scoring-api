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
        df_data = pd.read_csv('client_sample_dashboard.csv') 
        client_ids = df_data['SK_ID_CURR'].unique().tolist()
        
        with open('comparison_stats.json', 'r') as f:
            full_population_stats = json.load(f)
            
        return df_data, client_ids, full_population_stats
        
    except FileNotFoundError as e:
        st.error(f"❌ Un fichier de données est manquant. Erreur: {e}")
        return pd.DataFrame(), [], {}

@st.cache_resource
def load_model_and_explainer():
    """Charge le modèle et initialise l'explainer SHAP (pour l'explication locale et globale)."""
    try:
        model_pipeline = joblib.load('modele_de_scoring.pkl')
        df_ref = pd.read_csv('client_sample_dashboard.csv').drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')

        preprocessor_pipeline = Pipeline(model_pipeline.steps[:-1])
        final_classifier = model_pipeline.steps[-1][1]
        
        X_ref_processed = preprocessor_pipeline.transform(df_ref)
        
        explainer = shap.TreeExplainer(final_classifier, X_ref_processed)
        
        return model_pipeline, explainer, preprocessor_pipeline, X_ref_processed
        
    except Exception as e:
        st.error(f"❌ Erreur critique lors du chargement ou initialisation SHAP. Détail: {e}")
        return None, None, None, None

# --- Chargement ---
df_data, client_ids, full_population_stats = load_data() 
model_pipeline, explainer, preprocessor_pipeline, X_ref_processed = load_model_and_explainer()


# --- Fonction d'Appel de l'API ---
def get_prediction_from_api(client_features):
    """Appelle l'API Render pour obtenir le score."""
    payload = {k: None if (pd.isna(v) or v == "") else v for k, v in client_features.items()}
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erreur de connexion ou API indisponible. Détail: {e}")
        return None

# =============================================================================
# MISE EN PAGE STREAMLIT
# =============================================================================

st.title("💳 Dashboard d'Analyse de Crédit")
st.markdown("Outil d'aide à la décision pour l'octroi de prêts. Les données affichées reflètent les informations clients envoyées à l'API de scoring.")


# --- Sélection Client et Modification ---
st.sidebar.header("🔍 Sélection et Modification Client")

client_id = st.sidebar.selectbox(
    "Sélectionnez le SK_ID_CURR du client :",
    client_ids
)

client_data_raw = df_data[df_data['SK_ID_CURR'] == client_id].iloc[0].to_dict()
data_to_send = {'SK_ID_CURR': client_id}
edited_data = {}

# --- Formulaire de Modification  ---
st.sidebar.markdown("### 📝 Modification des Données (API rafraîchie)")

with st.sidebar.form(key=f"form_{client_id}"):
    for feature, value in client_data_raw.items():
        if feature not in ['SK_ID_CURR', 'TARGET']:
            
            input_val = st.text_input(f"{feature}", value=str(value) if pd.notna(value) else "", key=f"input_{feature}_{client_id}")

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
        st.rerun()
        
        
# --- Affichage Principal ---
if 'api_result' in st.session_state and st.session_state['api_result']['SK_ID_CURR'] == client_id:
    result = st.session_state['api_result']
    prob = result['probability']
    decision = result['prediction']
    message = result['decision_message']
    current_data = st.session_state['current_client_data']

    st.header(f"Client: {client_id} | Statut: {message}")
    st.markdown("---")

    # --- 1. Score et Jauge ---
    st.subheader("1. Score de Probabilité de Défaut")

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
    # 2. Explicabilité (Locale ou Globale)
    # =============================================================================
    st.subheader("2. Explication des Facteurs de Décision")
    
    col_radio, col_slider = st.columns([2, 1])
    
    with col_radio:
        explanation_type = st.radio(
            "Type d'Analyse :",
            ('Locale (Client)', 'Globale (Modèle)'),
            horizontal=True
        )
    
    with col_slider:
        # Ajout du slider pour le nombre de variables
        max_features = df_data.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore').shape[1]
        num_features_to_display = st.slider(
            "Nombre de variables à afficher :",
            min_value=5,
            max_value=min(20, max_features),
            value=10,
            step=1
        )
    
    if explainer and preprocessor_pipeline and X_ref_processed is not None:
        try:
            if explanation_type == 'Locale (Client)':
                st.markdown("#### Explication Locale : Facteurs influençant le score du client sélectionné")
                
                # --- Logique SHAP Locale ---
                data_to_explain = st.session_state['current_client_data']
                df_client = pd.DataFrame([data_to_explain]).drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
                
                X_client_processed = preprocessor_pipeline.transform(df_client) 
                
                shap_values = explainer.shap_values(X_client_processed)
                
                # Gestion de l'index SHAP
                if isinstance(shap_values, list):
                    try:
                        client_shap_values = shap_values[1][0] 
                        base_value = explainer.expected_value[1]
                    except IndexError:
                        client_shap_values = shap_values[0] 
                        base_value = explainer.expected_value
                else:
                    client_shap_values = shap_values[0] 
                    if isinstance(explainer.expected_value, np.ndarray) or isinstance(explainer.expected_value, list):
                        base_value = explainer.expected_value[0]
                    else:
                        base_value = explainer.expected_value


                # Créer l'objet Explanation 
                e = shap.Explanation(
                    client_shap_values, 
                    base_value, 
                    feature_names=df_client.columns.tolist() 
                )
                
                # Afficher le graphique Waterfall avec le nombre de variables sélectionné
                plt.rcParams.update({'figure.max_open_warning': 0})
                fig, ax = plt.subplots(figsize=(12, 7)) 
                shap.plots.waterfall(e, max_display=num_features_to_display, show=False)
                st.pyplot(fig, use_container_width=True)
                
                st.caption(f"Le rouge pousse vers le défaut, le bleu diminue le risque. Affiche les {num_features_to_display} facteurs les plus importants pour ce client.")

            elif explanation_type == 'Globale (Modèle)':
                st.markdown("#### Explication Globale : Importance moyenne des variables pour le modèle")
                
                # Correction Streamlit Cache pour explainer non hashable
                @st.cache_data
                def get_global_shap_values(_explainer, X_ref_processed):
                    return _explainer.shap_values(X_ref_processed)
                
                global_shap_values = get_global_shap_values(explainer, X_ref_processed)
                
                # Calculer la valeur absolue moyenne
                if isinstance(global_shap_values, list):
                    try:
                        shap_sum = np.abs(global_shap_values[1]).mean(axis=0)
                    except IndexError:
                        shap_sum = np.abs(global_shap_values[0]).mean(axis=0)
                else:
                    shap_sum = np.abs(global_shap_values).mean(axis=0)
                
                
                feature_names = df_data.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore').columns.tolist()
                
                # Limite le DataFrame d'importance au nombre choisi par le slider
                importance_df = pd.DataFrame({
                    'Feature': feature_names, 
                    'Importance': shap_sum
                }).sort_values(by='Importance', ascending=False).head(num_features_to_display)

                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', 
                             title=f"Top {num_features_to_display} des Variables les Plus Importantes (Moyenne Absolue des Valeurs SHAP)",
                             color='Importance',
                             color_continuous_scale=px.colors.sequential.OrRd)
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Affiche les {num_features_to_display} variables qui ont, en moyenne, le plus grand impact sur la décision du modèle.")

        except Exception as e:
            st.error(f"❌ Échec de l'Explication SHAP. Détail: {e}")

    st.markdown("---")
    
    # =============================================================================
    # 3. Comparaison et Positionnement
    # =============================================================================
    st.subheader("3. Comparaison aux Autres Clients (Échantillon de Référence)")
    
    col_feat_1, col_feat_2 = st.columns(2)

    with col_feat_1:
        st.markdown("#### Analyse Univariée (Distribution)")
        
        features_to_compare = [col for col in full_population_stats.keys() if full_population_stats[col]['type'] == 'num']
        selected_feature = st.selectbox(
            "Choisissez la caractéristique numérique à comparer :",
            features_to_compare,
            key='feature_uni'
        )
        
        client_val = current_data.get(selected_feature)

        if pd.notna(client_val):
            
            fig_dist = px.histogram(df_data, x=selected_feature, color='TARGET', 
                                    opacity=0.6, marginal="box", 
                                    title=f"Distribution de '{selected_feature}' dans l'Échantillon")

            fig_dist.add_vline(x=client_val, line_width=3, line_dash="dash", line_color="red", 
                               annotation_text="Client Actuel", annotation_position="top right")

            st.plotly_chart(fig_dist, use_container_width=True)
            
            st.metric(label="Valeur Client Actuelle", value=f"{client_val:,.2f}")
            
        else:
            st.warning("Variable non numérique ou valeur manquante pour la comparaison.")


    with col_feat_2:
        st.markdown("#### Analyse Bivariée (Positionnement)")
        
        num_features = [col for col in df_data.columns if df_data[col].dtype in [np.float64, np.int64] and col not in ['SK_ID_CURR', 'TARGET']]

        feat_x = st.selectbox("Axe X :", num_features, index=0, key='feat_x')
        feat_y = st.selectbox("Axe Y :", num_features, index=1, key='feat_y')
        
        fig_biv = px.scatter(df_data, x=feat_x, y=feat_y, color='TARGET', 
                              title=f"Relation entre {feat_x} et {feat_y} (Échantillon)",
                              color_continuous_scale=px.colors.sequential.Sunset,
                              hover_data=['SK_ID_CURR'])
        
        client_x = current_data.get(feat_x)
        client_y = current_data.get(feat_y)
        
        if client_x is not None and client_y is not None:
            fig_biv.add_scatter(x=[client_x], y=[client_y], mode='markers', name='Client Actuel', 
                                 marker=dict(color='red', size=15, symbol='star', line=dict(width=2, color='DarkRed')))

        st.plotly_chart(fig_biv, use_container_width=True)

else:
    st.info("Sélectionnez un client et cliquez sur 'Calculer le Score' dans la barre latérale pour démarrer l'analyse.")