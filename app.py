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
            if pd.api.types.is_numeric_dtype(dtype) and df_data[col].nunique() > 10:
                sample_population_stats[col] = {'type': 'num'}
            elif pd.api.types.is_object_dtype(dtype) or df_data[col].nunique() <= 10:
                sample_population_stats[col] = {'type': 'cat'}
            
        full_population_stats = sample_population_stats
        
        # 3. Retourner le DataFrame, la liste des IDs et les stats calculées
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

# =============================================================================
# MISE EN PAGE STREAMLIT (Corps du Dashboard)
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
    
    for feature in feature_names_raw:
        value = client_data_raw.get(feature)
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
            # On vérifie si la colonne est numérique dans les données originales pour l'inférence
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

    # --- CONTENU DE L'ONGLET 1 : EXPLICATION SHAP (API) ---
    with tab_explicability:
        
        # --- Extraction des données SHAP de l'API ---
        try:
            client_shap_values = np.array(result['shap_values'])
            base_value = result['base_value']
            feature_names_processed_api = result['shap_features'] 
            # Les données utilisées pour le waterfall plot sont les SHAP values elles-mêmes, pas les features prétraitées
            # Cependant, on a besoin d'une "data" (l'input) pour shap.Explanation, mais pour un waterfall plot,
            # on utilise souvent simplement le vecteur de SHAP values comme 'data' si l'input pré-traité n'est pas disponible.
            # L'API ne nous a pas donné l'input pré-traité, nous utiliserons donc un array de 0 (car le waterfall plot ne s'en sert pas vraiment).
            client_input_data = np.zeros_like(client_shap_values) 
            shap_data_available = True
            
        except (KeyError, TypeError, Exception) as e:
            st.warning(f"⚠️ Les données SHAP n'ont pas été trouvées dans la réponse de l'API. Détail: {e}")
            shap_data_available = False
        
        if shap_data_available and feature_names_processed_api:
            
            col_radio, col_slider = st.columns([2, 1])
            
            with col_radio:
                st.radio(
                    "Type d'Analyse :",
                    ('Locale (Client)'), 
                    horizontal=True,
                    key='exp_type_api'
                )
                st.caption("L'analyse locale est effectuée par l'API. L'analyse globale n'est pas disponible sans l'explainer local.")
            
            with col_slider:
                max_features_display = min(20, len(feature_names_processed_api)) 
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
                data=client_input_data, # Utilisation de l'array de zéros
                feature_names=feature_names_processed_api
            )
            
            fig_height = max(5, num_features_to_display * 0.5) 
            fig, ax = plt.subplots(figsize=(15, fig_height))
            shap.plots.waterfall(e, max_display=num_features_to_display, show=False)
            st.pyplot(fig, use_container_width=True)
            
            st.caption("Le rouge pousse vers le défaut, le bleu diminue le risque. Valeurs calculées par l'API.")

        else:
            st.warning("Impossible de générer les graphiques SHAP.")

    # --- CONTENU DE L'ONGLET 2 : COMPARAISON (Récupéré de votre ancienne version) ---
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