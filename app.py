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
# Importation nécessaire pour inspecter la structure du pipeline
from sklearn.compose import ColumnTransformer 

# --- Configuration Globale ---
API_URL = "https://scoring-api-latest.onrender.com/predict"
BEST_THRESHOLD = 0.52 
st.set_page_config(layout="wide", page_title="Dashboard Scoring Crédit")

# --- Fonctions de Chargement ---

@st.cache_data
def load_data():
    try:
        # 1. Chargement des données de l'échantillon
        df_data = pd.read_csv('client_sample_dashboard.csv') 
        client_ids = df_data['SK_ID_CURR'].unique().tolist()

        # 2. Calculer les métadonnées de l'échantillon
        sample_population_stats = {}
        
        cols_to_ignore = ['SK_ID_CURR', 'TARGET'] 
        
        for col in df_data.columns:
            if col in cols_to_ignore:
                continue
            
            dtype = df_data[col].dtype
            
            # Détermination du type :
            # Si c'est numérique ET qu'il y a plus de 10 valeurs uniques -> Numérique
            if pd.api.types.is_numeric_dtype(dtype) and df_data[col].nunique() > 10:
                sample_population_stats[col] = {'type': 'num'}
            # Sinon (chaîne de caractères ou moins de 10 valeurs uniques) -> Catégoriel
            elif pd.api.types.is_object_dtype(dtype) or df_data[col].nunique() <= 10:
                sample_population_stats[col] = {'type': 'cat'}
            
        full_population_stats = sample_population_stats
        
        # 3. Retourner le DataFrame, la liste des IDs et les stats calculées
        return df_data, client_ids, full_population_stats
        
    except FileNotFoundError as e:
        st.error(f"❌ Un fichier de données est manquant. Erreur: {e}")
        return pd.DataFrame(), [], {}

# ====== ⬇️⬇️⬇️ PARTIE SHAP MODIFIÉE : plus de calcul local, seulement des appels API ======

def api_shap_local(client_payload: dict):
    """
    Appelle l'endpoint /shap/local de l'API pour récupérer :
    - base_value
    - shap_values
    - data_processed
    - feature_names_processed
    """
    SHAP_URL = API_URL.replace("/predict", "/shap/local")
    try:
        resp = requests.post(SHAP_URL, json={"data": client_payload})
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erreur SHAP local (API) : {e}")
        return None

def api_shap_global(top_n: int):
    """
    Appelle l'endpoint /shap/global de l'API pour récupérer :
    - features
    - importances
    """
    SHAPG_URL = API_URL.replace("/predict", "/shap/global")
    try:
        resp = requests.get(SHAPG_URL, params={"top_n": top_n})
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erreur SHAP global (API) : {e}")
        return None

# ====== ⬆️⬆️⬆️ FIN PARTIE SHAP MODIFIÉE ======


# (L’ancienne fonction load_model_and_explainer n’est plus utilisée pour SHAP local/global.
#  On la laisse vide/sûre pour ne rien casser ailleurs.)
@st.cache_resource
def load_model_and_explainer():
    try:
        # Nous ne chargeons plus le modèle ni SHAP côté Streamlit
        # pour respecter la contrainte : SHAP calculé côté API.
        # On renvoie des placeholders compatibles avec le code existant.
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"❌ Erreur critique lors de l'initialisation. Détail: {e}")
        return None, None, None, None, None, None

# --- Fonction de Jauge Plotly ---

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


# --- Fonction d'Appel de l'API ---
def get_prediction_from_api(client_features):
    # Remplace les NaN/None par None pour l'API
    payload = {k: None if (pd.isna(v) or v == "") else v for k, v in client_features.items()}
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erreur de connexion ou API indisponible. Détail: {e}")
        return None

# --- Chargement ---
df_data, client_ids, full_population_stats = load_data() 
model_pipeline, explainer, preprocessor_pipeline, X_ref_processed, feature_names_processed, feature_names_raw = load_model_and_explainer()

# =============================================================================
# MISE EN PAGE STREAMLIT
# =============================================================================

st.markdown("<style>.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# Centrage du titre
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>Dashboard d'Analyse de Crédit</h1>
        <p>Outil d'aide à la décision pour l'octroi de prêts.</p>
    </div>
    """, 
    unsafe_allow_html=True
)


# --- Barre Latérale ---

try:
    st.sidebar.image('logo_entreprise.png', use_container_width=True) 
except FileNotFoundError:
    st.sidebar.warning("⚠️ Logo non trouvé.")
st.sidebar.markdown("---")

st.sidebar.header("🔍 Sélection Client")

client_id = st.sidebar.selectbox(
    "Choisissez le SK_ID_CURR :",
    client_ids
)

client_data_raw = df_data[df_data['SK_ID_CURR'] == client_id].iloc[0].to_dict()
data_to_send = {'SK_ID_CURR': client_id}
edited_data = {}

# --- Bouton de Score Rapide ---
if st.sidebar.button("Calculer le Score (API)", key="calculate_score_quick"):
    data_to_send.update({k: v for k, v in client_data_raw.items() if k not in ['SK_ID_CURR', 'TARGET']})
    api_result = get_prediction_from_api(data_to_send)
    
    if api_result:
        st.session_state['api_result'] = api_result
        st.session_state['current_client_data'] = data_to_send
        st.toast(f"Score pour le client {client_id} calculé!", icon='🚀')
        st.rerun()

# --- Formulaire de Modification ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Modification des Données")

with st.sidebar.form(key=f"form_{client_id}"):
    st.markdown("Modifiez les variables ci-dessous pour simuler un nouveau score :")
    
    for feature, value in client_data_raw.items():
        if feature not in ['SK_ID_CURR', 'TARGET']:
            
            col_label, col_input = st.columns([1.5, 2])
            with col_input:
                input_val = st.text_input(
                    f"{feature}", 
                    value=str(value) if pd.notna(value) else "", 
                    key=f"input_{feature}_{client_id}", 
                    label_visibility="collapsed"
                )
            
            with col_label:
                st.caption(f"{feature}")

            try:
                if input_val == "":
                    edited_data[feature] = np.nan
                elif '.' in input_val or 'e' in input_val.lower():
                    edited_data[feature] = float(input_val)
                else:
                    edited_data[feature] = int(input_val)
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
if 'api_result' in st.session_state and st.session_state['api_result']['SK_ID_CURR'] == client_id:
    result = st.session_state['api_result']
    prob = result['probability']
    decision = result['prediction']
    message = result['decision_message']
    current_data = st.session_state['current_client_data']

    st.markdown("---")
    
    # =============================================================================
    # 1. Score et Jauge
    # =============================================================================
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

    # --- DÉTAILS DU CLIENT ---
    st.markdown("---")
    st.subheader("Informations client")
    
    df_details = pd.Series(
        {k: v for k, v in current_data.items() if k not in ['SK_ID_CURR', 'TARGET']}
    ).rename('Valeur Client').to_frame()
    
    with st.expander("Cliquez pour voir toutes les variables et leurs valeurs", expanded=False):
        st.dataframe(df_details, height=300, use_container_width=True)
    
    st.markdown("---")

    # =============================================================================
    # 2 & 3. Explicabilité et Comparaison
    # =============================================================================
    tab_explicability, tab_comparison = st.tabs(["Explication des Facteurs (SHAP)", "Comparaison aux Autres Clients"])

    # --- CONTENU DE L'ONGLET 1 : EXPLICATION SHAP ---
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
            # On garde le slider tel quel
            num_features_to_display = st.slider(
                "Nombre de variables à afficher :",
                min_value=5,
                max_value=20,
                value=10,
                step=1,
                key='num_feat'
            )
        
        # ====== ⬇️⬇️⬇️ PARTIE SHAP MODIFIÉE : appels à l’API ======
        try:
            if explanation_type == 'Locale (Client)':
                st.markdown("#### Explication Locale : Facteurs influençant le score du client sélectionné")

                # Construire le payload identique au /predict
                data_to_explain = st.session_state['current_client_data']
                shap_resp = api_shap_local(data_to_explain)

                if shap_resp:
                    # Reconstituer l'Explanation pour le waterfall
                    base_value = shap_resp["base_value"]
                    shap_values = np.array(shap_resp["shap_values"])
                    data_processed = np.array(shap_resp["data_processed"])
                    feature_names_processed = shap_resp["feature_names_processed"]

                    e = shap.Explanation(
                        shap_values,
                        base_value,
                        data=data_processed,
                        feature_names=feature_names_processed
                    )

                    plt.rcParams.update({'figure.max_open_warning': 0})
                    fig_height = max(5, num_features_to_display * 0.5)
                    fig, ax = plt.subplots(figsize=(15, fig_height))
                    shap.plots.waterfall(e, max_display=num_features_to_display, show=False)
                    st.pyplot(fig, use_container_width=True)

                    st.caption(f"Le rouge pousse vers le défaut, le bleu diminue le risque. Affiche les **{num_features_to_display} facteurs les plus importants** (noms des variables après pré-traitement).")

            elif explanation_type == 'Globale (Modèle)':
                st.markdown("#### Explication Globale : Importance moyenne des variables pour le modèle")

                global_resp = api_shap_global(num_features_to_display)
                if global_resp:
                    importance_df = pd.DataFrame({
                        'Feature': global_resp['features'],
                        'Importance': global_resp['importances']
                    })

                    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', 
                                 title=f"Top {num_features_to_display} des Variables les Plus Importantes (Moyenne Absolue des Valeurs SHAP)",
                                 color='Importance',
                                 color_continuous_scale=px.colors.sequential.Blues)
                    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=max(500, num_features_to_display * 40))

                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
                    st.caption(f"Affiche les **{num_features_to_display} variables** qui ont, en moyenne, le plus grand impact sur la décision du modèle.")
        except Exception as e:
            st.error(f"❌ Échec de l'Explication SHAP. Détail: {e}")
        # ====== ⬆️⬆️⬆️ FIN PARTIE SHAP MODIFIÉE ======

    # --- CONTENU DE L'ONGLET 2 : COMPARAISON ---
    with tab_comparison:
        st.subheader("Comparaison et Positionnement Client")
        
        # 1. Sélection et Analyse Univariée
        st.markdown("---")
        st.markdown("### Analyse Univariée (Distribution)")

        # Regrouper toutes les variables à comparer (basé sur les métadonnées)
        features_all = list(full_population_stats.keys()) 
        
        col_uni_feat, col_uni_exp = st.columns([2.5, 1])
        
        with col_uni_feat:
            # Une seule liste déroulante pour toutes les variables
            selected_feature = st.selectbox(
                "Choisissez la caractéristique à comparer :",
                features_all,
                key='feature_uni_all'
            )
            
        with col_uni_exp:
            show_explanation_uni = st.checkbox("Afficher l'explication", value=False)
            
        # --- LOGIQUE D'AFFICHAGE AUTOMATIQUE DU GRAPHIQUE ---
        
        if selected_feature and selected_feature in current_data:
            if 'TARGET' not in df_data.columns or df_data['TARGET'].isnull().all():
                 st.error("La colonne 'TARGET' est manquante ou vide dans les données de l'échantillon.")
            else:
                client_val = current_data.get(selected_feature)
                
                # Déterminer le type grâce aux stats précalculées
                variable_type = full_population_stats.get(selected_feature, {}).get('type')

                # Traitement Numérique
                if variable_type == 'num':
                    
                    if pd.notna(client_val) and pd.api.types.is_numeric_dtype(df_data[selected_feature]):

                        st.markdown(f"**Valeur Actuelle :** <span style='font-size: 1.2em; font-weight: bold;'>{client_val:,.2f}</span>", unsafe_allow_html=True)
                        st.caption("Affichage de la distribution (Histogramme/Boxplot) pour cette variable numérique.")
                        df_data['TARGET_Label'] = df_data['TARGET'].astype(str).replace({
                            '0': 'Approuvé (0)', 
                            '1': 'Défaut (1)'
                        })
                        
                        # Graphique Numérique (Box-Histogramme)
                        fig_dist = px.histogram(df_data, x=selected_feature, color='TARGET_Label', 
                                                opacity=0.6, marginal="box", 
                                                title=f"Distribution de '{selected_feature}' dans l'Échantillon ",
                                                height=400,
                                                color_discrete_map={'Approuvé (0)': 'green', 'Défaut (1)': 'red'}) 

                        # Ligne verticale pour la valeur du client
                        fig_dist.add_shape(type="line", x0=client_val, y0=0, x1=client_val, y1=1, 
                                           yref='paper',
                                           line=dict(color="red", width=3, dash="dash"))
                        
                        fig_dist.add_annotation(x=client_val, y=0.95, yref="paper", 
                                                text="Client Actuel", showarrow=True, arrowhead=2, 
                                                font=dict(color="red", size=14))

                        st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': True})
                        
                        if show_explanation_uni:
                            st.info(f"""
                            **Interprétation (Analyse Numérique) :**
                            Ce graphique compare la valeur du client (**ligne rouge en tirets**) à la distribution de tous les clients.
                            L'histogramme montre la fréquence de la variable pour les clients approuvés (**0**) et ceux en défaut (**1**).
                            Regardez si la position du client est dans une zone où la majorité des clients sont en **Défaut (rouge)** ou **Approuvés (vert/bleu)**.
                            """)
                    else:
                        st.warning(f"La variable '{selected_feature}' n'est pas traitable comme numérique ou a une valeur manquante.")


                # Traitement Catégoriel
                elif variable_type == 'cat':
                    
                    if pd.notna(client_val):
                        st.markdown(f"**Catégorie Actuelle :** <span style='font-size: 1.2em; font-weight: bold;'>{client_val}</span>", unsafe_allow_html=True)
                        st.caption("Affichage du comptage (Graphique à barres) pour cette variable catégorielle.")
                        
                        # Préparation des données pour le bar chart (comptage par catégorie)
                        df_counts = df_data.groupby(selected_feature)['TARGET'].value_counts(normalize=False).rename('Count').reset_index()
                        df_counts['TARGET_Label'] = df_counts['TARGET'].astype(str).replace({
                            '0': 'Approuvé (0)', 
                            '1': 'Défaut (1)'
                        })

                        fig_cat = px.bar(df_counts, x=selected_feature, y='Count', color='TARGET_Label', 
                                         title=f"Distribution de '{selected_feature}' dans l'Échantillon (Comptage)",
                                         height=450,
                                         color_discrete_map={'Approuvé (0)': 'green', 'Défaut (1)': 'red'})
                        
                        st.plotly_chart(fig_cat, use_container_width=True, config={'displayModeBar': True})
                        
                        if show_explanation_uni:
                            st.info(f"""
                            **Interprétation (Analyse Catégorielle) :**
                            Ce graphique à barres montre la répartition des clients (Approuvé vs Défaut) pour chaque catégorie de la variable **'{selected_feature}'**.
                            La catégorie actuelle du client est **'{client_val}'**. 
                            Si la partie **Défaut (rouge)** est dominante dans cette catégorie, cela indique que les clients ayant cette caractéristique ont une forte propension au défaut.
                            """)
                    else:
                        st.warning(f"La variable '{selected_feature}' a une valeur manquante pour ce client.")
                
                else:
                    st.warning("Type de variable non reconnu pour l'affichage.")
        
        st.markdown("---")
        
        # 2. Analyse Bivariée
        st.markdown("### Analyse Bivariée (Positionnement)")
        
        # Récupération des features numériques
        num_features = [col for col in df_data.columns if df_data[col].dtype in [np.float64, np.int64] and col not in ['SK_ID_CURR', 'TARGET']]

        col_biv_feat_x, col_biv_feat_y, col_biv_exp = st.columns([1, 1, 1])

        with col_biv_feat_x:
            feat_x = st.selectbox("Axe X :", num_features, index=0, key='feat_x_tab')
        with col_biv_feat_y:
            feat_y = st.selectbox("Axe Y :", num_features, index=1, key='feat_y_tab')
        with col_biv_exp:
            show_explanation_biv = st.checkbox("Afficher l'explication (Biv.)", value=False)
        
        # Création du Scatter Plot
        fig_biv = px.scatter(df_data, x=feat_x, y=feat_y, color='TARGET', 
                             title=f"Relation entre {feat_x} et {feat_y} (Échantillon)",
                             color_continuous_scale=px.colors.sequential.Inferno,
                             hover_data=['SK_ID_CURR'])
        
        client_x = current_data.get(feat_x)
        client_y = current_data.get(feat_y)
        
        if client_x is not None and client_y is not None:
            # Ajout du point du client actuel (étoile rouge)
            fig_biv.add_scatter(x=[client_x], y=[client_y], mode='markers', name='Client Actuel', 
                                marker=dict(color='red', size=15, symbol='star', line=dict(width=2, color='DarkRed')))

        fig_biv.update_layout(height=500)
        st.plotly_chart(fig_biv, use_container_width=True, config={'displayModeBar': True})
        
        if show_explanation_biv:
            st.info(f"""
            **Interprétation (Analyse Bivariée) :**
            Ce nuage de points compare la position du client actuel (**étoile rouge**) par rapport à l'échantillon complet.
            La **couleur des points** (voir l'échelle à droite) indique le statut de défaut (`TARGET`): **sombre/violet** pour les défauts (TARGET=1) et **clair/jaune** pour les approuvés (TARGET=0). 
            Si l'**étoile rouge** se situe dans une zone majoritairement **sombre**, cela signale un risque plus élevé basé sur la combinaison de ces deux variables.
            """)

else:
    st.info("Sélectionnez un client et cliquez sur **'Calculer le Score (API)'** dans la barre latérale pour démarrer l'analyse.")
