"""
Application Streamlit — Prédiction de l'Autonomie des Femmes en RDC
Modèle : GradientBoostingClassifier (pipeline avec MinMaxScaler)
Données : Enquête Démographique et de Santé (EDS) — République Démocratique du Congo
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Autonomie des Femmes · RDC",
    page_icon="🇨🇩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLES CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Palette */
:root {
  --primary:   #534AB7;
  --secondary: #1D9E75;
  --danger:    #D85A30;
  --warning:   #F4A261;
  --light:     #F7F6FF;
  --dark:      #1E1B4B;
  --card-bg:   #FFFFFF;
  --border:    #E2E0F0;
}

/* Global */
.main { background: #F4F3FB; }
.block-container { padding: 1.5rem 2rem 2rem; }

/* Hero banner */
.hero-banner {
  background: linear-gradient(135deg, #534AB7 0%, #1D9E75 100%);
  border-radius: 16px;
  padding: 2rem 2.5rem;
  color: white;
  margin-bottom: 1.5rem;
  box-shadow: 0 6px 24px rgba(83,74,183,0.25);
}
.hero-banner h1 { font-size: 2rem; font-weight: 800; margin:0; letter-spacing:-0.5px; }
.hero-banner p  { font-size: 1rem; opacity: 0.88; margin-top: 0.4rem; }
.hero-badge {
  display:inline-block; background:rgba(255,255,255,0.2);
  border:1px solid rgba(255,255,255,0.35);
  border-radius:50px; padding:0.25rem 0.85rem;
  font-size:0.78rem; font-weight:600; letter-spacing:0.3px; margin-top:0.8rem;
}

/* Metric cards */
.metric-card {
  background: var(--card-bg);
  border-radius: 14px;
  padding: 1.2rem 1.4rem;
  border: 1px solid var(--border);
  box-shadow: 0 2px 10px rgba(83,74,183,0.07);
  text-align: center;
}
.metric-card .value {
  font-size: 2.1rem; font-weight: 800;
  background: linear-gradient(135deg, #534AB7, #1D9E75);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-card .label { font-size: 0.78rem; color: #666; margin-top:0.2rem; font-weight:500; }

/* Result cards */
.result-autonome {
  background: linear-gradient(135deg, #1D9E75 0%, #0F7A5A 100%);
  border-radius: 16px; padding: 1.5rem 2rem; color: white;
  box-shadow: 0 6px 20px rgba(29,158,117,0.30);
  text-align: center;
}
.result-non-autonome {
  background: linear-gradient(135deg, #D85A30 0%, #B0391A 100%);
  border-radius: 16px; padding: 1.5rem 2rem; color: white;
  box-shadow: 0 6px 20px rgba(216,90,48,0.30);
  text-align: center;
}
.result-card h2 { font-size: 1.6rem; font-weight: 800; margin:0; }
.result-card p  { font-size: 1rem; opacity:0.90; margin-top:0.4rem; }
.result-card .prob-label { font-size:0.8rem; opacity:0.75; margin-top:1rem; }
.result-card .prob-val   { font-size:2.8rem; font-weight:900; }

/* Section titles */
.section-title {
  font-size: 1.05rem; font-weight: 700; color: #534AB7;
  border-left: 4px solid #534AB7; padding-left: 0.7rem;
  margin: 1.2rem 0 0.7rem;
}

/* Feature importance bars */
.fi-bar-pos { background: linear-gradient(90deg,#1D9E75,#2CC990); border-radius:6px; height:12px; }
.fi-bar-neg { background: linear-gradient(90deg,#D85A30,#F06A40); border-radius:6px; height:12px; }

/* Info box */
.info-box {
  background: #EEF0FF; border-radius:10px; padding:0.8rem 1rem;
  border-left: 4px solid #534AB7; font-size:0.85rem; color:#333;
  margin: 0.8rem 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: #FAFAFA;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .css-ng1t4o { padding-top:1rem; }

/* Selectbox / slider labels */
label { font-weight: 600 !important; font-size: 0.85rem !important; color:#333 !important; }

/* Spinner */
.stSpinner > div { border-top-color: #534AB7 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:#F4F3FB; border-radius:10px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"] {
  border-radius:7px; font-weight:600; font-size:0.87rem;
  padding:0.4rem 1rem; color:#666;
}
.stTabs [aria-selected="true"] {
  background: white; color: #534AB7;
  box-shadow: 0 2px 8px rgba(83,74,183,0.12);
}

/* Button */
.stButton > button {
  background: linear-gradient(135deg, #534AB7, #1D9E75);
  color: white; border:none; border-radius:10px;
  font-weight:700; font-size:1rem; padding:0.6rem 2rem;
  box-shadow: 0 4px 14px rgba(83,74,183,0.30);
  transition: transform 0.15s, box-shadow 0.15s;
  width:100%;
}
.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(83,74,183,0.40);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_bundle():
    return joblib.load("model_final_congo_v2.pkl")

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("final_congo_women.csv")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (réplication exacte du pipeline d'entraînement)
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.preprocessing import MinMaxScaler
    df = df.copy()

    # Famille 1 — interactions socioéconomiques
    df['Edu_Employed']         = df['education'] * df['employment']
    df['Edu_Wealth']           = df['education'] * df['household_wealth']
    df['Employed_Wealth']      = df['employment'] * df['household_wealth']
    df['Muslim_Polygamous']    = df['religion_muslim'] * df['household_structure_polygamous']
    christian = (df['religion_catholic'] + df['religion_protestant'] +
                 df['religion_non_denominationalcharismaticapostolic']).clip(0, 1)
    df['Christian_Monogamous'] = christian * df['household_structure_monogamous']
    df['Partner_Polygamous']   = df['marital_status'] * df['household_structure_polygamous']
    df['TopSES']               = ((df['education'] == 3).astype(int) *
                                   (df['household_wealth'] >= 3).astype(int))
    df['BottomSES']            = ((df['education'] == 0).astype(int) *
                                   (df['household_wealth'] <= 1).astype(int))
    df['NonDenom_Polygamous']  = (df['religion_non_denominationalcharismaticapostolic'] *
                                   df['household_structure_polygamous'])
    df['RuralWealthy']         = ((df['place_of_residence'] == 0).astype(int) *
                                   (df['household_wealth'] >= 3).astype(int))

    # Famille 2 — zones régionales
    df['Zone_Tension']    = (df['region_of_residence_sankuru'] +
                              df['region_of_residence_maniema'] +
                              df['region_of_residence_nord_ubangi'] +
                              df['region_of_residence_sud_ubangi']).clip(0, 1)
    df['Kasai_Belt']      = (df['region_of_residence_kasaï'] +
                              df['region_of_residence_kasaï_central'] +
                              df['region_of_residence_kasaï_oriental'] +
                              df['region_of_residence_lomami']).clip(0, 1)
    df['East_Positive']   = (df['region_of_residence_ituri'] +
                              df['region_of_residence_sud_kivu'] +
                              df['region_of_residence_nord_kivu']).clip(0, 1)
    df['West_Progressive']= (df['region_of_residence_equateur'] +
                              df['region_of_residence_kwilu'] +
                              df['region_of_residence_kongo_central'] +
                              df['region_of_residence_mai_ndombe']).clip(0, 1)
    df['Kinshasa']         = df['region_of_residence_kinshasa']
    df['Tension_Polygamous']= df['Zone_Tension'] * df['household_structure_polygamous']
    df['Geographic_Risk']  = (df['Zone_Tension'] * 0.279 * (-1) +
                               df['Kasai_Belt'] * 0.143 * (-1) +
                               df['East_Positive'] * 0.150 +
                               df['West_Progressive'] * 0.121)

    # Famille 3 — parité
    df['No_Children']     = (1 - df['number_of_children_living_1_2'] -
                              df['number_of_children_living_3_4'] -
                              df['number_of_children_living_5+']).clip(0, 1)
    df['High_Parity']     = df['number_of_children_living_5+']
    df['Parity_Polygamous']= (df['number_of_children_living_5+'] *
                               df['household_structure_polygamous'])
    df['Parity_Educated'] = (df['number_of_children_living_5+'] *
                              (df['education'] >= 2).astype(int))
    df['OlderHighParity'] = ((df['age'] >= 4).astype(int) *
                              df['number_of_children_living_5+'])

    # Famille 4 — âge
    age_stage_map = {0:0, 1:0, 2:1, 3:1, 4:1, 5:2, 6:2}
    df['Life_Stage']       = df['age'].map(age_stage_map)
    df['Age_x_Education']  = df['age'] * df['education']
    df['Early_Union']      = (df['age'] == 0).astype(int)
    df['SeniorMonogamous'] = ((df['age'] >= 5).astype(int) *
                               df['household_structure_monogamous'])

    # Famille 5 — scores composites
    sc = MinMaxScaler()
    edu_n    = sc.fit_transform(df[['education']]).flatten()
    wealth_n = sc.fit_transform(df[['household_wealth']]).flatten()
    df['Empowerment_Index']    = edu_n * 0.40 + df['employment'].values * 0.35 + wealth_n * 0.25
    df['Traditional_Context']  = (df['religion_muslim'] * 2.0 +
                                   df['household_structure_polygamous'] * 1.5 +
                                   (1 - df['place_of_residence']) * 1.0 +
                                   (3 - df['education']) / 3.0 +
                                   df['number_of_children_living_5+'] * 1.0)
    df['Empowerment_vs_Context']= df['Empowerment_Index'] / (df['Traditional_Context'] + 1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENCODAGE RAW → NUMÉRIQUE
# ─────────────────────────────────────────────────────────────────────────────
REGIONS = [
    'bas_uele','equateur','haut_lomami','haut_uele','haut_katanga',
    'ituri','kasaï','kasaï_central','kasaï_oriental','kinshasa',
    'kongo_central','kwango','kwilu','lomami','lualaba','mai_ndombe',
    'maniema','mongala','nord_ubangi','nord_kivu','sankuru',
    'sud_ubangi','sud_kivu','tanganyika','tshopo','tshuapa'
]
REGIONS_LABELS = {
    'bas_uele':'Bas-Uele','equateur':'Équateur','haut_lomami':'Haut-Lomami',
    'haut_uele':'Haut-Uele','haut_katanga':'Haut-Katanga','ituri':'Ituri',
    'kasaï':'Kasaï','kasaï_central':'Kasaï Central','kasaï_oriental':'Kasaï Oriental',
    'kinshasa':'Kinshasa','kongo_central':'Kongo Central','kwango':'Kwango',
    'kwilu':'Kwilu','lomami':'Lomami','lualaba':'Lualaba','mai_ndombe':'Maï-Ndombe',
    'maniema':'Maniema','mongala':'Mongala','nord_ubangi':'Nord-Ubangi',
    'nord_kivu':'Nord-Kivu','sankuru':'Sankuru','sud_ubangi':'Sud-Ubangi',
    'sud_kivu':'Sud-Kivu','tanganyika':'Tanganyika','tshopo':'Tshopo','tshuapa':'Tshuapa'
}
RELIGIONS = ['animisttraditional_religion','catholic','jehovah_witness',
             'kimbanguiste','muslim','no_religion',
             'non_denominationalcharismaticapostolic','other','protestant']
RELIGIONS_LABELS = {
    'animisttraditional_religion':'Animiste / Religion traditionnelle',
    'catholic':'Catholique','jehovah_witness':'Témoin de Jéhovah',
    'kimbanguiste':'Kimbanguiste','muslim':'Musulman(e)',
    'no_religion':'Sans religion',
    'non_denominationalcharismaticapostolic':'Non-dénominationnel / Charismatique',
    'other':'Autre','protestant':'Protestant(e)'
}

def encode_input(vals: dict) -> pd.DataFrame:
    """Transforme les sélections UI en DataFrame prêt pour l'ingénierie des features."""
    row = {}

    # Variables continues/ordinales
    row['employment']        = 1 if vals['employment'] == 'Oui' else 0
    row['household_wealth']  = vals['household_wealth']   # 0-4
    row['education']         = vals['education']           # 0-3
    row['age']               = vals['age']                 # 0-6
    row['place_of_residence']= 1 if vals['place_of_residence'] == 'Urbain' else 0
    row['marital_status']    = 1 if vals['marital_status'] == 'Union libre / Concubinage' else 0

    # Religion (one-hot)
    for r in RELIGIONS:
        row[f'religion_{r}'] = 1 if vals['religion'] == r else 0

    # Household structure (one-hot)
    row['household_structure_monogamous'] = 1 if vals['household_structure'] == 'Monogame' else 0
    row['household_structure_polygamous'] = 1 if vals['household_structure'] == 'Polygame' else 0

    # Region of residence (one-hot)
    for reg in REGIONS:
        row[f'region_of_residence_{reg}'] = 1 if vals['region'] == reg else 0

    # Number of children (one-hot)
    row['number_of_children_living_0']   = 1 if vals['n_children'] == '0' else 0
    row['number_of_children_living_1_2'] = 1 if vals['n_children'] == '1-2' else 0
    row['number_of_children_living_3_4'] = 1 if vals['n_children'] == '3-4' else 0
    row['number_of_children_living_5+']  = 1 if vals['n_children'] == '5+' else 0

    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────────────────────
# GAUGE SVG
# ─────────────────────────────────────────────────────────────────────────────
def gauge_html(prob: float, label: str, color: str) -> str:
    """Génère un arc de jauge SVG animé."""
    pct = int(prob * 100)
    # Arc angles
    start_deg = 180
    sweep = 180 * prob
    end_deg = start_deg + sweep
    r = 80
    cx, cy = 100, 100
    to_rad = lambda d: d * np.pi / 180
    x1 = cx + r * np.cos(to_rad(start_deg))
    y1 = cy + r * np.sin(to_rad(start_deg))
    x2 = cx + r * np.cos(to_rad(end_deg))
    y2 = cy + r * np.sin(to_rad(end_deg))
    large = 1 if sweep > 180 else 0

    return f"""
    <div style="text-align:center;">
      <svg width="200" height="120" viewBox="0 0 200 110">
        <defs>
          <linearGradient id="gaugeBg" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:#EEECFF;stop-opacity:1" />
            <stop offset="100%" style="stop-color:#E0EDFF;stop-opacity:1" />
          </linearGradient>
          <linearGradient id="gaugeFg" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:{color};stop-opacity:1" />
            <stop offset="100%" style="stop-color:{color}AA;stop-opacity:1" />
          </linearGradient>
        </defs>
        <!-- Background arc -->
        <path d="M {cx-r},{cy} A {r},{r} 0 0,1 {cx+r},{cy}" 
              fill="none" stroke="url(#gaugeBg)" stroke-width="18" stroke-linecap="round"/>
        <!-- Foreground arc -->
        <path d="M {x1:.2f},{y1:.2f} A {r},{r} 0 {large},1 {x2:.2f},{y2:.2f}"
              fill="none" stroke="url(#gaugeFg)" stroke-width="18" stroke-linecap="round"/>
        <!-- Center text -->
        <text x="{cx}" y="{cy-2}" text-anchor="middle"
              font-size="24" font-weight="900" fill="{color}">{pct}%</text>
        <text x="{cx}" y="{cy+16}" text-anchor="middle"
              font-size="10" fill="#888" font-weight="600">Probabilité</text>
        <!-- Tick marks -->
        <text x="18" y="{cy+4}" text-anchor="middle" font-size="9" fill="#aaa">0%</text>
        <text x="182" y="{cy+4}" text-anchor="middle" font-size="9" fill="#aaa">100%</text>
        <text x="{cx}" y="26" text-anchor="middle" font-size="9" fill="#aaa">50%</text>
      </svg>
      <div style="font-size:0.85rem;font-weight:700;color:{color};margin-top:-4px;">{label}</div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHIQUE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_feature_importance(_model, feature_names):
    clf = _model.named_steps['clf']
    imp = clf.feature_importances_
    fi = pd.Series(imp, index=feature_names).sort_values(ascending=False)
    return fi.head(15)


def plot_importance(fi: pd.Series):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    colors = ['#534AB7' if i < 3 else '#1D9E75' if i < 8 else '#AFA9EC'
              for i in range(len(fi))]
    bars = ax.barh(fi.index[::-1], fi.values[::-1], color=colors[::-1],
                   height=0.65, edgecolor='white')
    for bar, val in zip(bars, fi.values[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8, color='#444')
    ax.set_xlabel("Importance (Gain moyen)", fontsize=9, color='#666')
    ax.set_title("Top 15 Features — Importance du Modèle", fontweight='bold',
                 fontsize=10, pad=8)
    ax.tick_params(labelsize=8)
    ax.spines[['top','right','bottom']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    patches = [
        mpatches.Patch(color='#534AB7', label='Top 3'),
        mpatches.Patch(color='#1D9E75', label='Top 4-8'),
        mpatches.Patch(color='#AFA9EC', label='Autres'),
    ]
    ax.legend(handles=patches, fontsize=8, loc='lower right', framealpha=0.5)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TAUX PAR RÉGION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def regional_rates(df_path: str):
    df = pd.read_csv(df_path)
    reg_cols = [c for c in df.columns if c.startswith('region_of_residence_')]
    rates = {}
    for col in reg_cols:
        sub = df[df[col] == 1]
        if len(sub) > 0:
            w = sub['Survey_Weight']
            r = np.average(sub['Women autonomy'], weights=w)
            name = col.replace('region_of_residence_', '')
            rates[name] = r
    return pd.Series(rates).sort_values(ascending=False)


def plot_regional(rates: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    grate = 0.4754
    colors = ['#1D9E75' if v >= grate else '#D85A30' for v in rates.values]
    labels = [REGIONS_LABELS.get(k, k) for k in rates.index]
    ax.barh(labels[::-1], rates.values[::-1], color=colors[::-1],
            height=0.65, edgecolor='white')
    ax.axvline(grate, color='#534AB7', linestyle='--', linewidth=1.5,
               label=f'Moy. pondérée ({grate:.1%})')
    ax.set_xlabel("Taux d'autonomie pondéré", fontsize=9, color='#666')
    ax.set_title("Taux d'Autonomie par Région — RDC", fontweight='bold',
                 fontsize=10, pad=8)
    ax.set_xlim(0, 0.85)
    for i, (v, l) in enumerate(zip(rates.values[::-1], labels[::-1])):
        ax.text(v + 0.008, i, f'{v:.0%}', va='center', fontsize=7.5, color='#444')
    ax.tick_params(labelsize=8)
    ax.spines[['top','right','bottom']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Chargement du modèle…"):
    try:
        bundle = load_bundle()
        model  = bundle['model']
        feat_names = bundle['feature_names']
        global_rate = bundle['global_rate']
        MODEL_OK = True
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        MODEL_OK = False
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🇨🇩 Autonomie des Femmes — RDC</h1>
  <p>Outil d'aide à la décision basé sur l'apprentissage automatique.<br>
     Prédit la probabilité qu'une femme jouisse d'une autonomie décisionnelle
     à partir de ses caractéristiques socio-démographiques.</p>
  <span class="hero-badge">GradientBoostingClassifier · MinMaxScaler · EDS-RDC · 16 938 observations</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────
tab_pred, tab_explore, tab_info = st.tabs([
    "🔮 Prédiction Individuelle",
    "📊 Analyse & Statistiques",
    "ℹ️ À Propos du Modèle"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRÉDICTION
# ═════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown('<div class="section-title">Profil de la femme</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    📝 Renseignez les caractéristiques socio-démographiques ci-dessous, puis cliquez sur
    <strong>Prédire l'autonomie</strong> pour obtenir le résultat.
    </div>""", unsafe_allow_html=True)

    # ── Formulaire ────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 👤 Profil personnel")
        age_val = st.select_slider(
            "Tranche d'âge",
            options=['15-19','20-24','25-29','30-34','35-39','40-44','45-49'],
            value='25-29'
        )
        age_map = {'15-19':0,'20-24':1,'25-29':2,'30-34':3,'35-39':4,'40-44':5,'45-49':6}

        employment_val = st.radio(
            "Emploi rémunéré",
            options=['Non','Oui'],
            horizontal=True
        )

        education_val = st.select_slider(
            "Niveau d'éducation",
            options=['Aucune','Primaire','Secondaire','Supérieur'],
            value='Secondaire'
        )
        edu_map = {'Aucune':0,'Primaire':1,'Secondaire':2,'Supérieur':3}

        marital_val = st.radio(
            "Statut matrimonial",
            options=['Marié(e)','Union libre / Concubinage'],
            horizontal=False
        )

    with col2:
        st.markdown("##### 🏠 Ménage & Religion")
        wealth_val = st.select_slider(
            "Richesse du ménage",
            options=['Très pauvre','Pauvre','Moyen','Aisé','Très aisé'],
            value='Moyen'
        )
        wealth_map = {'Très pauvre':0,'Pauvre':1,'Moyen':2,'Aisé':3,'Très aisé':4}

        structure_val = st.radio(
            "Structure du ménage",
            options=['Monogame','Polygame'],
            horizontal=True
        )

        religion_options = {
            'Catholique':                'catholic',
            'Protestant(e)':             'protestant',
            'Non-dénominationnel':       'non_denominationalcharismaticapostolic',
            'Kimbanguiste':              'kimbanguiste',
            'Musulman(e)':               'muslim',
            'Animiste / Trad.':          'animisttraditional_religion',
            'Témoin de Jéhovah':         'jehovah_witness',
            'Sans religion':             'no_religion',
            'Autre':                     'other',
        }
        religion_label = st.selectbox("Religion", list(religion_options.keys()))

        n_children_val = st.radio(
            "Nombre d'enfants vivants",
            options=['0','1-2','3-4','5+'],
            horizontal=True
        )

    with col3:
        st.markdown("##### 📍 Localisation")
        residence_val = st.radio(
            "Milieu de résidence",
            options=['Rural','Urbain'],
            horizontal=True
        )

        region_label = st.selectbox(
            "Province / Région",
            options=sorted(REGIONS_LABELS.values()),
            index=sorted(REGIONS_LABELS.values()).index('Kinshasa')
        )
        inv_labels = {v: k for k, v in REGIONS_LABELS.items()}
        region_key = inv_labels[region_label]

    # ── Bouton prédiction ─────────────────────────────────────────────────────
    st.markdown("---")
    btn_col, _ = st.columns([1, 2])
    with btn_col:
        predict_clicked = st.button("🔮 Prédire l'autonomie", use_container_width=True)

    if predict_clicked:
        with st.spinner("Calcul en cours…"):
            vals = {
                'employment':        employment_val,
                'household_wealth':  wealth_map[wealth_val],
                'education':         edu_map[education_val],
                'age':               age_map[age_val],
                'place_of_residence':residence_val,
                'marital_status':    marital_val,
                'religion':          religion_options[religion_label],
                'household_structure': structure_val,
                'region':            region_key,
                'n_children':        n_children_val,
            }

            raw_df = encode_input(vals)
            fe_df  = engineer_features_v2(raw_df)

            # Aligner les colonnes sur le modèle
            X_pred = fe_df[feat_names]

            proba_arr = model.predict_proba(X_pred)[0]
            pred      = model.predict(X_pred)[0]
            prob_pos  = float(proba_arr[1])
            prob_neg  = float(proba_arr[0])

        # ── RÉSULTATS ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📋 Résultat de la Prédiction")

        res_col1, res_col2, res_col3 = st.columns([1.4, 1.4, 1.2])

        with res_col1:
            if pred == 1:
                st.markdown(f"""
                <div class="result-autonome result-card">
                  <div style="font-size:2.4rem;">✅</div>
                  <h2>Autonome</h2>
                  <p>Le profil indique une probabilité <strong>élevée</strong><br>
                     d'autonomie décisionnelle.</p>
                  <div class="prob-label">Probabilité d'autonomie</div>
                  <div class="prob-val">{prob_pos:.0%}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-non-autonome result-card">
                  <div style="font-size:2.4rem;">⚠️</div>
                  <h2>Non Autonome</h2>
                  <p>Le profil indique une probabilité <strong>faible</strong><br>
                     d'autonomie décisionnelle.</p>
                  <div class="prob-label">Probabilité d'autonomie</div>
                  <div class="prob-val">{prob_pos:.0%}</div>
                </div>""", unsafe_allow_html=True)

        with res_col2:
            color_gauge = '#1D9E75' if pred == 1 else '#D85A30'
            label_gauge = 'Autonome' if pred == 1 else 'Non autonome'
            st.markdown(gauge_html(prob_pos, label_gauge, color_gauge),
                        unsafe_allow_html=True)

            # Barre de comparaison
            st.markdown(f"""
            <div style="margin-top:1rem;">
              <div style="font-size:0.8rem;color:#666;font-weight:600;margin-bottom:4px;">
                vs Taux national ({global_rate:.1%})
              </div>
              <div style="background:#EEE;border-radius:8px;height:10px;overflow:hidden;">
                <div style="
                  width:{prob_pos*100:.1f}%;
                  height:10px;
                  background:{'#1D9E75' if prob_pos >= global_rate else '#D85A30'};
                  border-radius:8px;
                "></div>
              </div>
              <div style="display:flex;justify-content:space-between;
                          font-size:0.72rem;color:#999;margin-top:2px;">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>""", unsafe_allow_html=True)

        with res_col3:
            st.markdown('<div class="section-title">Récapitulatif</div>',
                        unsafe_allow_html=True)
            summary_items = [
                ("🎂 Âge", age_val),
                ("📚 Éducation", education_val),
                ("💼 Emploi", employment_val),
                ("💰 Richesse", wealth_val),
                ("🏘️ Milieu", residence_val),
                ("🛐 Religion", religion_label),
                ("👨‍👩‍👧 Enfants", n_children_val),
                ("📍 Région", region_label[:18] + ("…" if len(region_label) > 18 else "")),
            ]
            for icon_label, val in summary_items:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                            padding:0.22rem 0;border-bottom:1px solid #F0EEF8;
                            font-size:0.82rem;">
                  <span style="color:#666;">{icon_label}</span>
                  <span style="font-weight:600;color:#333;">{val}</span>
                </div>""", unsafe_allow_html=True)

            # Delta vs national
            delta = prob_pos - global_rate
            delta_color = '#1D9E75' if delta >= 0 else '#D85A30'
            delta_sign  = '+' if delta >= 0 else ''
            st.markdown(f"""
            <div style="margin-top:0.8rem;background:{'#E8F8F1' if delta>=0 else '#FEF0EB'};
                        border-radius:8px;padding:0.5rem 0.8rem;text-align:center;">
              <span style="font-size:0.78rem;color:#666;">Écart vs moyenne nationale</span><br>
              <span style="font-size:1.4rem;font-weight:800;color:{delta_color};">
                {delta_sign}{delta:.1%}
              </span>
            </div>""", unsafe_allow_html=True)

        # ── Facteurs clés ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-title">📌 Facteurs explicatifs clés</div>',
                    unsafe_allow_html=True)

        fi = get_feature_importance(model, feat_names)
        # Filtrer les features pertinentes pour ce profil (valeur != 0)
        row_vals = X_pred.iloc[0]
        relevant = [f for f in fi.index if row_vals.get(f, 0) != 0]
        if not relevant:
            relevant = fi.index[:5].tolist()

        top_fi = fi[fi.index.isin(relevant)].head(8)

        fi_labels = {
            'Empowerment_vs_Context': 'Émancipation vs Contexte traditionnel',
            'Empowerment_Index':      'Index d\'émancipation globale',
            'Geographic_Risk':        'Risque géographique (région)',
            'Traditional_Context':    'Score contexte traditionnel',
            'Age_x_Education':        'Interaction Âge × Éducation',
            'Edu_Wealth':             'Interaction Éducation × Richesse',
            'Edu_Employed':           'Interaction Éducation × Emploi',
            'Life_Stage':             'Phase de vie (jeune/prime/senior)',
            'household_wealth':       'Richesse du ménage',
            'education':              'Niveau d\'éducation',
            'age':                    'Âge',
            'Zone_Tension':           'Zone régionale à forte tension',
            'Kasai_Belt':             'Ceinture Kasaï (conservatrice)',
            'East_Positive':          'Provinces orientales (+)',
            'Muslim_Polygamous':      'Islam + polygamie (restrictif)',
            'SeniorMonogamous':       'Femme senior monogame (+)',
            'TopSES':                 'Capital socioéco. maximal (+)',
            'BottomSES':              'Vulnérabilité socioéco. max (-)',
        }

        fi_cols = st.columns(min(len(top_fi), 4))
        for i, (fname, fval) in enumerate(top_fi.items()):
            lbl = fi_labels.get(fname, fname.replace('_', ' '))
            raw_v = float(row_vals.get(fname, 0))
            direction = "➕" if raw_v > 0 else "🔲"
            pct_bar = int(fval / fi.iloc[0] * 100)
            col_idx = i % len(fi_cols)
            with fi_cols[col_idx]:
                st.markdown(f"""
                <div style="background:white;border-radius:10px;padding:0.7rem 0.9rem;
                            border:1px solid #E2E0F0;margin-bottom:0.5rem;
                            box-shadow:0 1px 6px rgba(83,74,183,0.06);">
                  <div style="font-size:0.72rem;font-weight:700;color:#534AB7;
                              text-overflow:ellipsis;overflow:hidden;white-space:nowrap;">
                    {direction} {lbl[:30]}{"…" if len(lbl)>30 else ""}
                  </div>
                  <div style="background:#F0EEFF;border-radius:5px;height:7px;
                              margin:0.4rem 0;overflow:hidden;">
                    <div style="width:{pct_bar}%;height:7px;
                                background:linear-gradient(90deg,#534AB7,#1D9E75);
                                border-radius:5px;"></div>
                  </div>
                  <div style="font-size:0.78rem;color:#888;">
                    Valeur : <strong>{raw_v:.2f}</strong> · Importance : <strong>{fval:.4f}</strong>
                  </div>
                </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORATION
# ═════════════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown('<div class="section-title">Statistiques globales du modèle</div>',
                unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    metrics = [
        ("69.0%",  "AUC-ROC", "Performance discriminante"),
        ("65.3%",  "Précision", "Accuracy test (20%)"),
        ("59.3%",  "F1-Score", "Classe positive"),
        ("47.5%",  "Taux national", "Autonomie pondérée · EDS"),
    ]
    for col, (val, name, desc) in zip([mc1,mc2,mc3,mc4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="value">{val}</div>
              <div style="font-size:0.9rem;font-weight:700;color:#333;margin-top:2px;">{name}</div>
              <div class="label">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown('<div class="section-title">Importance des Features</div>',
                    unsafe_allow_html=True)
        fi = get_feature_importance(model, feat_names)
        fig_fi = plot_importance(fi)
        st.pyplot(fig_fi, use_container_width=True)

    with exp_col2:
        st.markdown('<div class="section-title">Taux d\'Autonomie par Région</div>',
                    unsafe_allow_html=True)
        rates = regional_rates("final_congo_women.csv")
        fig_reg = plot_regional(rates)
        st.pyplot(fig_reg, use_container_width=True)

    # ── Tableau de progression ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Progression du Pipeline ML</div>',
                unsafe_allow_html=True)

    prog_data = pd.DataFrame([
        {'Étape':'1. Baseline','Modèle':'GradientBoosting (défaut)','AUC CV':'~0.665','AUC Test':'~0.668','Features':'47'},
        {'Étape':'2. + FE v1 (intuitif)','Modèle':'GradientBoosting','AUC CV':'~0.677','AUC Test':'—','Features':'64'},
        {'Étape':'3. + FE v2 (guidé EDA)','Modèle':'GradientBoosting','AUC CV':'~0.684','AUC Test':'~0.686','Features':'76'},
        {'Étape':'4. + Optimisation HP','Modèle':'GradientBoosting (tuné)','AUC CV':'~0.690','AUC Test':'0.690','Features':'76'},
    ])
    st.dataframe(prog_data, use_container_width=True, hide_index=True,
                 column_config={
                     'Étape': st.column_config.TextColumn(width='medium'),
                     'AUC CV': st.column_config.TextColumn(width='small'),
                     'AUC Test': st.column_config.TextColumn(width='small'),
                 })

    # ── Insights EDA clés ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Insights EDA Clés</div>',
                unsafe_allow_html=True)

    insights = [
        ("🗺️ Région = Prédicteur #1", "V de Cramér = 0.305 — 3× plus fort que la variable suivante."),
        ("📈 Âge → Monotone", "De 32.6% (15-19 ans) à 54.4% (45-49 ans) sans exception."),
        ("💎 Richesse & Éducation", "Profil TopSES : 59.6% (+12.1pt) vs BottomSES : 35.3% (−12.2pt)."),
        ("🕌 Islam + Polygamie", "Muslim : −21.5pt ; Polygame : −9.4pt vs moyenne nationale (47.5%)."),
        ("🔴 Sankuru outlier", "≈8.3% pondéré — écart de −39.3pt, profil culturel distinct."),
        ("⚖️ Lieu × Richesse", "V de Cramér = 0.661 — quasi-redondance gérée par RuralWealthy."),
    ]
    ins_cols = st.columns(3)
    for i, (title, detail) in enumerate(insights):
        with ins_cols[i % 3]:
            st.markdown(f"""
            <div style="background:white;border-radius:10px;padding:0.9rem 1rem;
                        border:1px solid #E2E0F0;margin-bottom:0.6rem;
                        box-shadow:0 1px 6px rgba(83,74,183,0.05);">
              <div style="font-weight:700;font-size:0.88rem;color:#534AB7;">{title}</div>
              <div style="font-size:0.81rem;color:#555;margin-top:0.3rem;">{detail}</div>
            </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — À PROPOS
# ═════════════════════════════════════════════════════════════════════════════
with tab_info:
    about_col1, about_col2 = st.columns([1.2, 0.8])

    with about_col1:
        st.markdown("""
        ### 🧠 Architecture du Modèle

        | Composant | Détail |
        |-----------|--------|
        | **Algorithme** | Gradient Boosting Classifier (sklearn) |
        | **Pré-traitement** | MinMaxScaler (normalisation [0,1]) |
        | **n_estimators** | 300 arbres |
        | **max_depth** | 4 |
        | **learning_rate** | 0.05 |
        | **subsample** | 0.8 |
        | **min_samples_leaf** | 10 |
        | **sample_weight** | Survey_Weight (EDS-pondéré) |
        | **Validation croisée** | StratifiedGroupKFold(5) — PSU clusters |

        ---

        ### 📦 Ingénierie des Features (v2)

        Le pipeline enrichit les 47 variables originales avec **29 features construites**
        organisées en 5 familles :

        - **Famille 1 — Interactions socioéconomiques** (10 features)
        - **Famille 2 — Zones régionales** (7 features)
        - **Famille 3 — Parité** (5 features)
        - **Famille 4 — Âge** (4 features)
        - **Famille 5 — Scores composites** (3 features)

        ---

        ### 📊 Données & Contexte

        - **Source** : Enquête Démographique et de Santé (EDS) — République Démocratique du Congo
        - **Observations** : 16 938 femmes âgées de 15 à 49 ans
        - **Variable cible** : *Women autonomy* — décision autonome concernant la santé,
          les achats ou les visites familiales (binaire : Oui / Non)
        - **Taux national pondéré** : 47.5% d'autonomie
        - **Déséquilibre de classe** : 52.5% Non / 47.5% Oui — modéré

        ---

        ### ⚠️ Limitations & Usage Éthique

        > Ce modèle est un **outil d'aide à la recherche** et à la politique publique.
        > Il ne doit pas être utilisé pour prendre des décisions individuelles
        > concernant des personnes réelles. Les prédictions reflètent des
        > **probabilités statistiques au niveau populationnel**, pas des
        > déterminations individuelles.

        Les facteurs structurels (région, religion, structure du ménage) reflètent
        des inégalités systémiques qui nécessitent des **interventions politiques**,
        pas des jugements individuels.
        """)

    with about_col2:
        st.markdown("### 🔑 Dictionnaire des Variables")

        var_dict = {
            "Emploi rémunéré": "Oui / Non — travail contre rémunération",
            "Niveau d'éducation": "Aucune (0) → Supérieur (3)",
            "Richesse du ménage": "Quintiles 0 (plus pauvre) → 4 (plus riche)",
            "Âge": "Tranches quinquennales 15-49 ans",
            "Milieu de résidence": "Rural (0) / Urbain (1)",
            "Statut matrimonial": "Marié(e) (0) / Union libre (1)",
            "Structure du ménage": "Monogame / Polygame",
            "Religion": "9 catégories (OHE)",
            "Province / Région": "26 provinces (OHE)",
            "Nombre d'enfants": "0 / 1-2 / 3-4 / 5+ (OHE)",
        }
        for var, desc in var_dict.items():
            st.markdown(f"""
            <div style="background:#FAFAFA;border-radius:8px;padding:0.5rem 0.7rem;
                        border-left:3px solid #534AB7;margin-bottom:0.4rem;
                        font-size:0.82rem;">
              <strong style="color:#534AB7;">{var}</strong><br>
              <span style="color:#555;">{desc}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        ---
        ### 📈 Métriques de Performance

        ```
        Classe          Précision  Rappel   F1
        ─────────────────────────────────────
        Pas autonome     0.70      0.70    0.70
        Autonome         0.59      0.59    0.59
        ─────────────────────────────────────
        Accuracy                          0.65
        AUC-ROC                           0.69
        ```
        """)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.78rem;color:#aaa;padding:0.5rem 0;">
  🇨🇩 Autonomie des Femmes · RDC &nbsp;|&nbsp;
  GradientBoostingClassifier · sklearn &nbsp;|&nbsp;
  Source : EDS-RDC &nbsp;|&nbsp;
  Usage académique et recherche uniquement
</div>""", unsafe_allow_html=True)
