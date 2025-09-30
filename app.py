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
        df_data = pd.read_csv('client_sample_dashboard.csv') 
        client_ids = df_data['SK_ID_CURR'].unique().tolist()
        
        # Le fichier de comparaison est conservé, car il ne concerne pas les noms SHAP
        with open('comparison_stats.json', 'r') as f:
            full_population_stats = json.load(f)
            
        return df_data, client_ids, full_population_stats
        
    except FileNotFoundError as e:
        st.error(f"❌ Un fichier de données est manquant. Erreur: {e}")
        return pd.DataFrame(), [], {}

@st.cache_resource
def load_model_and_explainer():
    
    # --- FONCTION D'EXTRACTION MANUELLE DES NOMS DE FEATURES ---
    # Si get_feature_names_out() échoue, on tente de reconstruire les noms.
    def get_feature_names_manually(preprocessor_pipeline, raw_feature_names):
        feature_names_processed = []
        try:
            # 1. Obtenir le ColumnTransformer (en supposant qu'il soit la seule étape)
            # Sinon, il faut adapter le nom de l'étape : e.g., preprocessor_pipeline.named_steps['column_transformer_step']
            if isinstance(preprocessor_pipeline, ColumnTransformer):
                ct = preprocessor_pipeline
            else:
                # Tenter de trouver le ColumnTransformer dans le pipeline
                ct = next(step[1] for step in preprocessor_pipeline.steps if isinstance(step[1], ColumnTransformer))

            # 2. Parcourir les transformateurs
            for name, transformer, features in ct.transformers_:
                
                # Le 'remainder' renvoie les noms bruts des colonnes non transformées
                if name == 'remainder':
                    # Dans les versions récentes, 'remainder' retourne 'passthrough' et on gère les colonnes restantes
                    if transformer == 'passthrough':
                        # Trouver les noms de colonnes non utilisées par d'autres transformateurs
                        cols_used = set()
                        for _, _, used_features in ct.transformers_:
                            if isinstance(used_features, str):
                                cols_used.add(used_features)
                            elif isinstance(used_features, list):
                                cols_used.update(used_features)
                        
                        remainder_cols = [col for col in raw_feature_names if col not in cols_used]
                        feature_names_processed.extend(remainder_cols)
                    else:
                        # Si le remainder fait une transformation (rare), on gère ici si besoin
                        pass 
                
                # Pour les transformateurs spécifiques (ex: num, cat)
                elif transformer != 'drop':
                    # Les transformateurs ayant 'get_feature_names_out' sont généralement les encodeurs (OneHot)
                    if hasattr(transformer, 'get_feature_names_out'):
                        # Utilisation de la méthode spécifique du transformateur (plus fiable que le CT)
                        names_out = transformer.get_feature_names_out(features)
                        # On applique votre nettoyage (retirer le préfixe)
                        feature_names_processed.extend([n.split('__')[-1] for n in names_out])
                    else:
                        # Pour les Standard Scaler, Imputer, etc., les noms ne changent pas
                        if isinstance(features, str):
                             feature_names_processed.append(features)
                        elif isinstance(features, list):
                            feature_names_processed.extend(features)
                        
            st.sidebar.success("✅ Noms de features extraits manuellement!")
            return feature_names_processed

        except Exception as e:
            st.sidebar.error(f"❌ Échec de l'extraction manuelle. Retour aux noms génériques. Détail: {e}")
            # Retourne une liste de noms génériques si l'extraction échoue
            return [f"Feature_{i}" for i in range(X_ref_processed.shape[1])]

    try:
        model_pipeline = joblib.load('modele_de_scoring.pkl')
        # Chargement des données de référence (sans ID/cible)
        df_ref = pd.read_csv('client_sample_dashboard.csv').drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')

        preprocessor_pipeline = Pipeline(model_pipeline.steps[:-1])
        final_classifier = model_pipeline.steps[-1][1]
        
        # Transformation pour obtenir la dimension correcte
        X_ref_processed = preprocessor_pipeline.transform(df_ref)
        
        # Noms des features brutes (pour l'extraction manuelle)
        feature_names_raw = df_ref.columns.tolist() 

        # --- DÉTERMINATION DES NOMS DES FEATURES POST-TRAITEMENT ---
        try:
            # 1. Tenter la méthode standard de scikit-learn (si la version a été fixée)
            feature_names_full = preprocessor_pipeline.get_feature_names_out().tolist()
            feature_names_processed = [name.split('__')[-1] for name in feature_names_full]
            st.sidebar.success("✅ Noms de features récupérés via get_feature_names_out()!")
        except Exception:
            # 2. Si échec (votre cas), utiliser la fonction d'extraction manuelle
            feature_names_processed = get_feature_names_manually(preprocessor_pipeline, feature_names_raw)
        
        # ----------------------------------------------------------------------
        
        # Création de l'explainer après avoir défini feature_names_processed
        explainer = shap.TreeExplainer(final_classifier, X_ref_processed)
        
        # On retourne X_ref_processed et les noms transformés (nettoyés ou par défaut)
        return model_pipeline, explainer, preprocessor_pipeline, X_ref_processed, feature_names_processed, feature_names_raw
        
    except Exception as e:
        st.error(f"❌ Erreur critique lors du chargement ou initialisation. Détail: {e}")
        return None, None, None, None, None, None

# --- Fonction de Jauge Plotly (Aucun changement nécessaire) ---

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


# --- Fonction d'Appel de l'API (Aucun changement nécessaire) ---
def get_prediction_from_api(client_features):
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

# --- En-tête (Centrage du Titre) ---
st.markdown("<style>.block-container {padding-top: 1rem;}</style>", unsafe_allow_html=True)

# Centrage du titre sans colonnes
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>Dashboard d'Analyse de Crédit</h1>
        <p>Outil d'aide à la décision pour l'octroi de prêts.</p>
    </div>
    """, 
    unsafe_allow_html=True
)


# --- Barre Latérale  ---

# Affichage du logo dans la barre latérale
try:
    # Utilisez 'logo_entreprise.png' si vous l'avez, sinon retirez cette ligne.
    st.sidebar.image(
        'logo_entreprise.png', 
        use_container_width=True
    ) 
except FileNotFoundError:
    st.sidebar.warning("⚠️ Logo non trouvé.")
st.sidebar.markdown("---")

st.sidebar.header("🔍 Sélection Client")

client_id = st.sidebar.selectbox(
    "1. Choisissez le SK_ID_CURR :",
    client_ids
)

client_data_raw = df_data[df_data['SK_ID_CURR'] == client_id].iloc[0].to_dict()
data_to_send = {'SK_ID_CURR': client_id}
edited_data = {}

# --- Bouton de Score Rapide ---
if st.sidebar.button("Calculer le Score (API)", key="calculate_score_quick"):
    # Envoie toutes les données brutes (sauf SK_ID_CURR et TARGET)
    data_to_send.update({k: v for k, v in client_data_raw.items() if k not in ['SK_ID_CURR', 'TARGET']})
    api_result = get_prediction_from_api(data_to_send)
    
    if api_result:
        st.session_state['api_result'] = api_result
        st.session_state['current_client_data'] = data_to_send
        st.toast(f"Score pour le client {client_id} calculé!", icon='🚀')
        st.rerun()

# --- Formulaire de Modification  ---
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

            # Tentative de conversion de type pour l'API
            try:
                if input_val == "":
                    edited_data[feature] = np.nan
                elif '.' in input_val or 'e' in input_val.lower():
                    edited_data[feature] = float(input_val)
                else:
                    edited_data[feature] = int(input_val)
            except ValueError:
                # Si la conversion échoue (ex: texte dans un champ numérique), on garde la chaîne
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
# Affiche la page principale uniquement si un score a été calculé et correspond au client actuel
if 'api_result' in st.session_state and st.session_state['api_result']['SK_ID_CURR'] == client_id:
    result = st.session_state['api_result']
    prob = result['probability']
    decision = result['prediction']
    message = result['decision_message']
    current_data = st.session_state['current_client_data']

    st.markdown("---")
    
    # =============================================================================
    # 1. Score et Jauge (SECTION FIXE)
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

    # --- DÉTAILS DU CLIENT DÉFILABLES ---
    st.markdown("---")
    st.subheader("Informations client")
    
    df_details = pd.Series(
        {k: v for k, v in current_data.items() if k not in ['SK_ID_CURR', 'TARGET']}
    ).rename('Valeur Client').to_frame()
    
    with st.expander("Cliquez pour voir toutes les variables et leurs valeurs", expanded=False):
        st.dataframe(df_details, height=300, use_container_width=True)
    
    st.markdown("---")

    # =============================================================================
    # 2 & 3. Explicabilité et Comparaison (ONGLETS INTERACTIFS)
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
        
        if explainer and preprocessor_pipeline and X_ref_processed is not None and feature_names_processed is not None:
            try:
                # --- EXPLICATION LOCALE ---
                if explanation_type == 'Locale (Client)':
                    st.markdown("#### Explication Locale : Facteurs influençant le score du client sélectionné")
                    
                    data_to_explain = st.session_state['current_client_data']
                    df_client = pd.DataFrame([data_to_explain]).drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
                    
                    # Transformation des données pour le SHAP local
                    X_client_processed = preprocessor_pipeline.transform(df_client) 
                    
                    shap_values = explainer.shap_values(X_client_processed)
                    
                    if isinstance(shap_values, list):
                        if len(shap_values) > 1:
                            client_shap_values = shap_values[1][0] 
                            base_value = explainer.expected_value[1]
                        else:
                            client_shap_values = shap_values[0][0]
                            base_value = explainer.expected_value[0]
                    else:
                        client_shap_values = shap_values[0] 
                        base_value = explainer.expected_value if not isinstance(explainer.expected_value, (np.ndarray, list)) else explainer.expected_value[0]
                    
                    # Convertit en array si c'est une matrice creuse pour SHAP.Explanation
                    if issparse(X_client_processed):
                        client_data = X_client_processed.toarray()[0]
                    else:
                        client_data = X_client_processed[0]
                        
                    # Utilisation des noms de features nettoyés/reconstruits
                    e = shap.Explanation(
                        client_shap_values, 
                        base_value, 
                        data=client_data, 
                        feature_names=feature_names_processed
                    )
                    
                    plt.rcParams.update({'figure.max_open_warning': 0})
                    
                    fig_height = max(5, num_features_to_display * 0.5) 
                    fig, ax = plt.subplots(figsize=(15, fig_height))
                    
                    shap.plots.waterfall(e, max_display=num_features_to_display, show=False)
                    
                    st.pyplot(fig, use_container_width=True)
                    
                    st.caption(f"Le rouge pousse vers le défaut, le bleu diminue le risque. Affiche les **{num_features_to_display} facteurs les plus importants** (noms des variables après pré-traitement).")

                # --- EXPLICATION GLOBALE ---
                elif explanation_type == 'Globale (Modèle)':
                    st.markdown("#### Explication Globale : Importance moyenne des variables pour le modèle")
                    
                    @st.cache_data
                    def get_global_shap_values(_explainer, X_ref_processed):
                        sample_indices = np.random.choice(X_ref_processed.shape[0], size=min(500, X_ref_processed.shape[0]), replace=False)
                        X_sample_for_global = X_ref_processed[sample_indices]
                        return _explainer.shap_values(X_sample_for_global)
                    
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
                                 title=f"Top {num_features_to_display} des Variables les Plus Importantes (Moyenne Absolue des Valeurs SHAP)",
                                 color='Importance',
                                 color_continuous_scale=px.colors.sequential.Blues) 
                    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=max(500, num_features_to_display * 40))
                    
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True}) 
                    st.caption(f"Affiche les **{num_features_to_display} variables** qui ont, en moyenne, le plus grand impact sur la décision du modèle.")

            except Exception as e:
                st.error(f"❌ Échec de l'Explication SHAP. Détail: {e}")
        else:
             st.warning("Impossible de générer les graphiques SHAP. Vérifiez que le modèle et les données de référence sont chargés correctement.")

    # --- CONTENU DE L'ONGLET 2 : COMPARAISON (Aucun changement nécessaire) ---
    with tab_comparison:
        st.subheader("Comparaison et Positionnement Client (Échantillon de Référence)")
        
        col_feat_1, col_feat_2 = st.columns(2)

        with col_feat_1:
            st.markdown("#### Analyse Univariée (Distribution)")
            
            features_to_compare = [col for col in full_population_stats.keys() if full_population_stats[col]['type'] == 'num']
            selected_feature = st.selectbox(
                "Choisissez la caractéristique numérique à comparer :",
                features_to_compare,
                key='feature_uni_tab'
            )
            
            client_val = current_data.get(selected_feature)

            if pd.notna(client_val):
                
                fig_dist = px.histogram(df_data, x=selected_feature, color='TARGET', 
                                        opacity=0.6, marginal="box", 
                                        title=f"Distribution de '{selected_feature}' dans l'Échantillon")

                fig_dist.add_vline(x=client_val, line_width=3, line_dash="dash", line_color="red", 
                                   annotation_text="Client Actuel", annotation_position="top right")

                st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': True})
                
                st.metric(label="Valeur Client Actuelle", value=f"{client_val:,.2f}")
                
            else:
                st.warning("Variable non numérique ou valeur manquante pour la comparaison.")


        with col_feat_2:
            st.markdown("#### Analyse Bivariée (Positionnement)")
            
            num_features = [col for col in df_data.columns if df_data[col].dtype in [np.float64, np.int64] and col not in ['SK_ID_CURR', 'TARGET']]

            feat_x = st.selectbox("Axe X :", num_features, index=0, key='feat_x_tab')
            feat_y = st.selectbox("Axe Y :", num_features, index=1, key='feat_y_tab')
            
            fig_biv = px.scatter(df_data, x=feat_x, y=feat_y, color='TARGET', 
                                  title=f"Relation entre {feat_x} et {feat_y} (Échantillon)",
                                  color_continuous_scale=px.colors.sequential.Sunset,
                                  hover_data=['SK_ID_CURR'])
            
            client_x = current_data.get(feat_x)
            client_y = current_data.get(feat_y)
            
            if client_x is not None and client_y is not None:
                fig_biv.add_scatter(x=[client_x], y=[client_y], mode='markers', name='Client Actuel', 
                                     marker=dict(color='red', size=15, symbol='star', line=dict(width=2, color='DarkRed')))

            st.plotly_chart(fig_biv, use_container_width=True, config={'displayModeBar': True})

else:
    st.info("Sélectionnez un client et cliquez sur **'Calculer le Score (API)'** dans la barre latérale pour démarrer l'analyse.")