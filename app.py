import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Claim Decisioning Engine · Royal Sundaram",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand tokens ───────────────────────────────────────────────────────────────
RS_YELLOW  = "#F5C200"
RS_BLUE    = "#003087"
RS_NAVY    = "#001A4E"
RS_GOLD    = "#D4A800"
RS_WHITE   = "#FFFFFF"
RS_OFFWHITE= "#F8F9FC"
RS_GRAY    = "#6B7280"
RS_LGRAY   = "#E5E7EB"
CRITICAL   = "#DC2626"
HIGH       = "#EA580C"
MEDIUM     = "#D97706"
FASTTRACK  = "#16A34A"

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Reset & base ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {RS_OFFWHITE};
    color: #111827;
  }}

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding: 0 2rem 2rem 2rem; max-width: 1400px; }}

  /* ── Top banner ── */
  .top-banner {{
    background: linear-gradient(135deg, {RS_NAVY} 0%, {RS_BLUE} 60%, #0050CC 100%);
    border-bottom: 4px solid {RS_YELLOW};
    padding: 1.4rem 2.5rem 1.2rem 2.5rem;
    margin: -1rem -2rem 2rem -2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
  }}
  .banner-left {{ display: flex; align-items: center; gap: 1.5rem; }}
  .banner-logo {{
    background: {RS_YELLOW};
    border-radius: 10px;
    padding: 8px 14px;
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: {RS_NAVY};
    letter-spacing: 0.5px;
    white-space: nowrap;
  }}
  .banner-title {{
    color: {RS_WHITE};
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    line-height: 1.3;
  }}
  .banner-sub {{
    color: rgba(255,255,255,0.65);
    font-size: 0.78rem;
    font-weight: 400;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 2px;
  }}
  .banner-right {{
    text-align: right;
    color: rgba(255,255,255,0.7);
    font-size: 0.72rem;
    letter-spacing: 0.5px;
    line-height: 1.7;
  }}
  .banner-right strong {{
    color: {RS_YELLOW};
    font-weight: 600;
    display: block;
    font-size: 0.78rem;
    letter-spacing: 1px;
    text-transform: uppercase;
  }}

  /* ── Section headers ── */
  .section-label {{
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: {RS_BLUE};
    margin-bottom: 0.4rem;
  }}
  .section-title {{
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: {RS_NAVY};
    margin-bottom: 0.2rem;
  }}
  .section-divider {{
    width: 48px;
    height: 3px;
    background: {RS_YELLOW};
    border-radius: 2px;
    margin-bottom: 1.5rem;
  }}

  /* ── Stat cards ── */
  .stat-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
  }}
  .stat-card {{
    background: {RS_WHITE};
    border: 1px solid {RS_LGRAY};
    border-top: 3px solid {RS_YELLOW};
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 8px rgba(0,48,135,0.06);
  }}
  .stat-value {{
    font-size: 1.8rem;
    font-weight: 700;
    color: {RS_NAVY};
    line-height: 1;
    margin-bottom: 0.3rem;
  }}
  .stat-label {{
    font-size: 0.72rem;
    color: {RS_GRAY};
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }}
  .stat-delta {{
    font-size: 0.75rem;
    color: {FASTTRACK};
    font-weight: 600;
    margin-top: 0.4rem;
  }}

  /* ── Score card ── */
  .score-panel {{
    background: linear-gradient(135deg, {RS_NAVY} 0%, {RS_BLUE} 100%);
    border-radius: 14px;
    padding: 2rem;
    color: white;
    box-shadow: 0 8px 32px rgba(0,48,135,0.25);
    border: 1px solid rgba(245,194,0,0.3);
  }}
  .score-label {{
    font-size: 0.65rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: {RS_YELLOW};
    font-weight: 700;
    margin-bottom: 0.5rem;
  }}
  .score-number {{
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    font-weight: 700;
    line-height: 1;
    color: {RS_WHITE};
  }}
  .score-denom {{
    font-size: 1.2rem;
    color: rgba(255,255,255,0.5);
    font-weight: 400;
  }}
  .score-bar-bg {{
    background: rgba(255,255,255,0.15);
    border-radius: 4px;
    height: 8px;
    margin: 1rem 0;
    overflow: hidden;
  }}
  .score-bar-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
  }}

  /* ── Route badge ── */
  .route-badge {{
    display: inline-block;
    padding: 0.4rem 1.1rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-top: 0.8rem;
  }}
  .route-critical  {{ background: #FEE2E2; color: {CRITICAL}; border: 1.5px solid {CRITICAL}; }}
  .route-high      {{ background: #FFEDD5; color: {HIGH};     border: 1.5px solid {HIGH}; }}
  .route-medium    {{ background: #FEF3C7; color: {MEDIUM};   border: 1.5px solid {MEDIUM}; }}
  .route-fast      {{ background: #DCFCE7; color: {FASTTRACK};border: 1.5px solid {FASTTRACK}; }}

  /* ── Prob meters ── */
  .meter-row {{
    display: flex;
    gap: 1rem;
    margin: 1.2rem 0;
  }}
  .meter-box {{
    flex: 1;
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 1rem;
    border: 1px solid rgba(255,255,255,0.12);
  }}
  .meter-title {{
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.6);
    margin-bottom: 0.4rem;
  }}
  .meter-val {{
    font-size: 1.6rem;
    font-weight: 700;
    color: white;
  }}

  /* ── SHAP reasons ── */
  .reason-card {{
    background: {RS_WHITE};
    border: 1px solid {RS_LGRAY};
    border-left: 4px solid {RS_YELLOW};
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.82rem;
    color: #1F2937;
    font-weight: 500;
  }}
  .reason-up   {{ border-left-color: {CRITICAL}; }}
  .reason-down {{ border-left-color: {FASTTRACK}; }}

  /* ── Form styling ── */
  .form-section-header {{
    background: {RS_BLUE};
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 1.2rem 0 0.8rem 0;
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
    background: {RS_NAVY} !important;
    border-right: 3px solid {RS_YELLOW};
  }}
  [data-testid="stSidebar"] * {{
    color: {RS_WHITE} !important;
  }}
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stNumberInput label,
  [data-testid="stSidebar"] .stSlider label {{
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
  }}
  [data-testid="stSidebar"] .sidebar-logo {{
    background: {RS_YELLOW};
    color: {RS_NAVY};
    text-align: center;
    padding: 1rem;
    border-radius: 8px;
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
  }}

  /* ── Info pill ── */
  .info-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    color: {RS_BLUE};
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
  }}

  /* ── Footer ── */
  .footer-bar {{
    background: {RS_NAVY};
    color: rgba(255,255,255,0.5);
    text-align: center;
    padding: 1rem 2rem;
    margin: 3rem -2rem -2rem -2rem;
    font-size: 0.72rem;
    letter-spacing: 0.5px;
    border-top: 2px solid {RS_YELLOW};
  }}
  .footer-bar span {{ color: {RS_YELLOW}; font-weight: 600; }}

  /* ── Stacked summary table ── */
  .summary-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    margin-top: 0.5rem;
  }}
  .summary-table th {{
    background: {RS_BLUE};
    color: white;
    padding: 0.6rem 0.8rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }}
  .summary-table td {{
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid {RS_LGRAY};
    color: #374151;
  }}
  .summary-table tr:nth-child(even) td {{ background: #F9FAFB; }}

  /* ── About tab ── */
  .about-card {{
    background: {RS_WHITE};
    border: 1px solid {RS_LGRAY};
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }}
  .about-card h4 {{
    color: {RS_NAVY};
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}
  .about-card p {{
    color: {RS_GRAY};
    font-size: 0.82rem;
    line-height: 1.6;
    margin: 0;
  }}

  /* Streamlit widget tweaks */
  div[data-baseweb="select"] > div {{
    border-radius: 6px !important;
    border-color: {RS_LGRAY} !important;
    font-size: 0.82rem !important;
  }}
  .stButton > button {{
    background: linear-gradient(135deg, {RS_BLUE}, {RS_NAVY}) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 2rem !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 12px rgba(0,48,135,0.3) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
  }}
  .stButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(0,48,135,0.4) !important;
  }}
</style>
""", unsafe_allow_html=True)

# ── Top Banner ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-banner">
  <div class="banner-left">
    <div class="banner-logo">RS</div>
    <div>
      <div class="banner-title">Claim Decisioning Engine</div>
      <div class="banner-sub">Motor Third-Party · AI-Driven Fraud &amp; Litigation Triage</div>
    </div>
  </div>
  <div class="banner-right">
    <strong>Sundaram Pitch Fest 2026</strong>
    Team Apex Counsel · IIT Kharagpur<br>
    Arunadithyan S · Azhagappan G · G Abiimukeshwar<br>
    Operations · Risk · Process Excellence
  </div>
</div>
""", unsafe_allow_html=True)

# ── Industry Stats Row ──────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stat-row">
  <div class="stat-card">
    <div class="stat-value">₹96,257 Cr</div>
    <div class="stat-label">Outstanding TP Liability · FY25</div>
    <div class="stat-delta">↑ 19.6% from FY23</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">10.7 L</div>
    <div class="stat-label">Open TP Cases · Industry</div>
    <div class="stat-delta">4.07L pending 3+ years</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">~6%</div>
    <div class="stat-label">Estimated Fraud Rate · Motor TP</div>
    <div class="stat-delta">↑ Organised ring exposure</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">~2×</div>
    <div class="stat-label">Fraud Catch Lift · Top-10% Review</div>
    <div class="stat-delta">Prototype on public benchmark</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load Models ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading decisioning models…")
def load_models():
    try:
        mf  = joblib.load('model_fraud_india.pkl')
        ml  = joblib.load('model_litigation_india.pkl')
        enc = joblib.load('encoder_india.pkl')
        ff  = joblib.load('features_fraud.pkl')
        lf  = joblib.load('features_litigation.pkl')
        return mf, ml, enc, ff, lf, True
    except Exception as e:
        return None, None, None, None, None, False

model_fraud, model_lit, enc, feat_fraud, feat_lit, models_ok = load_models()

if not models_ok:
    st.error("⚠️  Model files not found. Please run the training notebook first and place the `.pkl` files in the same directory as `app.py`.")
    st.info("Required files: `model_fraud_india.pkl`, `model_litigation_india.pkl`, `encoder_india.pkl`, `features_fraud.pkl`, `features_litigation.pkl`")
    st.stop()

# ── Constants ──────────────────────────────────────────────────────────────────
SEVERITY_MAP = {
    'Below ₹5 Lakh':   0.5,
    '₹5–8 Lakh':       0.7,
    '₹8–12 Lakh':      0.85,
    '₹12–20 Lakh':     1.0,
    '₹20–30 Lakh':     1.2,
    'Above ₹30 Lakh':  1.5,
}

LIT_DROP = ['FNOL_Delay_Days','FIR_Filed','Fault','Prior_Claims_Count','Claim_Filing_Delay']

def route_claim(fp, lp, rs):
    if fp >= 0.5:   return 'SIU Investigation',    'critical'
    if lp >= 0.5:   return 'ADR / Legal Prep',     'high'
    if rs < 25:     return 'Fast Track Settlement', 'fast'
    return             'Standard Processing',       'medium'

def priority_tier(score):
    if score >= 70: return 'P1 — Critical',   CRITICAL
    if score >= 45: return 'P2 — High',        HIGH
    if score >= 25: return 'P3 — Medium',      MEDIUM
    return               'P4 — Fast Track',    FASTTRACK

def engineer_features(df):
    df['High_Value_Vehicle']  = df['VehiclePrice'].isin(['₹12–20 Lakh','₹20–30 Lakh','Above ₹30 Lakh']).astype(int)
    df['Young_Driver_Flag']   = df['PolicyHolder_Age_Band'].isin(['16 to 17','18 to 20','21 to 25']).astype(int)
    df['Weak_Documentation']  = ((df['FIR_Filed']=='No') & (df['Witness_Available']=='No')).astype(int)
    df['Commercial_TP']       = ((df['VehicleCategory']=='Commercial Vehicle') & (df['BasePolicy']=='Third Party')).astype(int)
    df['Broker_Address_Risk'] = ((df['Intermediary_Type']=='Broker / POSP Agent') & (df['Address_Change_Before_Claim'].isin(['under 6 months','1 year']))).astype(int)
    return df

def score_claim(claim_dict):
    df = pd.DataFrame([claim_dict])
    df = engineer_features(df)
    # Fix pandas 3.x string dtype
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
            df[c] = df[c].astype(object)
    # Fill missing cat cols
    cat_cols = [c for c in feat_fraud if df[c].dtype == object] if feat_fraud else []
    for c in enc.feature_names_in_ if hasattr(enc,'feature_names_in_') else []:
        if c not in df.columns:
            df[c] = 'unknown'
    enc_cols = [c for c in enc.feature_names_in_] if hasattr(enc,'feature_names_in_') else []
    if enc_cols:
        present = [c for c in enc_cols if c in df.columns]
        df[present] = enc.transform(df[present]) if len(present)==len(enc_cols) else df[present]

    df_fraud = df.reindex(columns=feat_fraud, fill_value=0)
    df_lit   = df.reindex(columns=feat_lit,   fill_value=0)

    fp = float(model_fraud.predict_proba(df_fraud)[:,1][0])
    lp = float(model_lit.predict_proba(df_lit)[:,1][0])
    sev   = SEVERITY_MAP.get(claim_dict.get('VehiclePrice','₹8–12 Lakh'), 1.0)
    score = round(min((0.6*fp + 0.4*lp)*sev*100, 100), 1)
    route, tier_key = route_claim(fp, lp, score)
    tier_label, tier_color = priority_tier(score)

    # SHAP
    explainer = shap.TreeExplainer(model_fraud)
    shap_vals = explainer.shap_values(df_fraud)
    shap_ser  = pd.Series(shap_vals[0], index=feat_fraud)
    top3      = shap_ser.abs().nlargest(3).index
    reasons   = [(f, shap_ser[f]) for f in top3]

    return fp, lp, score, route, tier_key, tier_label, tier_color, reasons

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍  Score a Claim", "📊  Model Insights", "ℹ️  About the Engine"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLAIM SCORER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Live Claim Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Enter Claim Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    with st.form("claim_form"):
        # ── Policy & Vehicle ──
        st.markdown('<div class="form-section-header">01 · Policy & Vehicle Information</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        vehicle_category = c1.selectbox("Vehicle Category", ['Private Car','Two Wheeler','Commercial Vehicle'])
        vehicle_make     = c2.selectbox("Vehicle Make", ['Maruti Suzuki','Hyundai','Tata Motors','Mahindra','Honda City','Toyota Innova','BMW India','Audi India','Mercedes India','Renault','Nissan India','Skoda','Volkswagen India','Force Motors'])
        vehicle_price    = c3.selectbox("Vehicle Value", list(SEVERITY_MAP.keys()))
        vehicle_age      = c4.selectbox("Vehicle Age", ['new','2 years','3 years','4 years','5 years','6 years','7 years','more than 7'])

        c1, c2, c3, c4 = st.columns(4)
        base_policy      = c1.selectbox("Cover Type", ['Third Party','Own Damage','Comprehensive'])
        policy_type_map  = {
            ('Private Car','Third Party'):       'Private Car - Third Party',
            ('Private Car','Own Damage'):        'Private Car - Own Damage',
            ('Private Car','Comprehensive'):     'Private Car - Comprehensive',
            ('Two Wheeler','Third Party'):       'Two Wheeler - Third Party',
            ('Two Wheeler','Own Damage'):        'Two Wheeler - Own Damage',
            ('Two Wheeler','Comprehensive'):     'Two Wheeler - Comprehensive',
            ('Commercial Vehicle','Third Party'):'Commercial Vehicle - Third Party',
            ('Commercial Vehicle','Own Damage'): 'Commercial Vehicle - Own Damage',
            ('Commercial Vehicle','Comprehensive'):'Commercial Vehicle - Comprehensive',
        }
        policy_type      = policy_type_map.get((vehicle_category, base_policy), 'Private Car - Third Party')
        deductible_inr   = c2.number_input("Deductible (₹)", min_value=5000, max_value=100000, value=33000, step=1000)
        driver_rating    = c3.slider("Driver Risk Rating", 1, 4, 2, help="1=Low risk, 4=High risk")
        year             = c4.number_input("Policy Year", min_value=2018, max_value=2025, value=2023)

        # ── Claimant ──
        st.markdown('<div class="form-section-header">02 · Claimant Information</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        sex              = c1.selectbox("Sex", ['Male','Female'])
        marital_status   = c2.selectbox("Marital Status", ['Single','Married','Divorced','Widow'])
        age              = c3.number_input("Age", 18, 80, 35)
        age_band_map = lambda a: ('16 to 17' if a<=17 else '18 to 20' if a<=20 else '21 to 25' if a<=25
                                  else '26 to 30' if a<=30 else '31 to 35' if a<=35 else '36 to 40' if a<=40
                                  else '41 to 50' if a<=50 else '51 to 65' if a<=65 else 'over 65')
        age_band         = age_band_map(age)
        prior_claims     = c4.selectbox("Prior Claims Count", ['0','1','2–4','5+'])

        # ── Accident & FNOL ──
        st.markdown('<div class="form-section-header">03 · Accident & FNOL Details</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        accident_area    = c1.selectbox("Accident Area", ['Urban','Rural'])
        fault            = c2.selectbox("Fault Attribution", ['Insured Driver','Third Party'])
        fnol_delay       = c3.selectbox("FNOL Delay (Days)", ['none','1 to 7','8 to 15','15 to 30','more than 30'])
        claim_delay      = c4.selectbox("Claim Filing Delay (Days)", ['none','8 to 15','15 to 30','more than 30'])

        c1, c2, c3, c4 = st.columns(4)
        fir_filed        = c1.selectbox("FIR Filed", ['Yes','No'])
        witness          = c2.selectbox("Witness Available", ['Yes','No'])
        month            = c3.selectbox("Accident Month", ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        day_of_week      = c4.selectbox("Day of Week", ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

        # ── Distribution & Supplementary ──
        st.markdown('<div class="form-section-header">04 · Distribution & Documentation</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        intermediary     = c1.selectbox("Intermediary Type", ['Broker / POSP Agent','Direct / Branch'])
        address_change   = c2.selectbox("Address Change Before Claim", ['no change','under 6 months','1 year','2 to 3 years','4 to 8 years'])
        supp_reports     = c3.selectbox("Supplementary Reports", ['0','1–2','3–5','5+'])
        vehicles_in_pol  = c4.selectbox("Vehicles in Policy", ['1','2','3–4','5–8','8+'])

        c1, c2, c3, _ = st.columns(4)
        week_of_month    = c1.number_input("Week of Month", 1, 5, 3)
        month_claimed    = c2.selectbox("Month Claimed", ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
        surveyor_id      = c3.number_input("Surveyor ID", 1, 50, 12)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚖️  Run Claim Decisioning Engine")

    # ── Results ──────────────────────────────────────────────────────────────
    if submitted:
        claim = {
            'Month': month, 'WeekOfMonth': week_of_month, 'DayOfWeek': day_of_week,
            'VehicleMake': vehicle_make, 'AccidentArea': accident_area,
            'DayOfWeekClaimed': day_of_week, 'MonthClaimed': month_claimed,
            'WeekOfMonthClaimed': week_of_month, 'Sex': sex, 'MaritalStatus': marital_status,
            'Age': age, 'Fault': fault, 'PolicyType': policy_type,
            'VehicleCategory': vehicle_category, 'VehiclePrice': vehicle_price,
            'Surveyor_ID': surveyor_id, 'Deductible_INR': deductible_inr,
            'Driver_Risk_Rating': driver_rating, 'FNOL_Delay_Days': fnol_delay,
            'Claim_Filing_Delay': claim_delay, 'Prior_Claims_Count': prior_claims,
            'Vehicle_Age': vehicle_age, 'PolicyHolder_Age_Band': age_band,
            'FIR_Filed': fir_filed, 'Witness_Available': witness,
            'Intermediary_Type': intermediary, 'Supplementary_Reports': supp_reports,
            'Address_Change_Before_Claim': address_change, 'Vehicles_In_Policy': vehicles_in_pol,
            'Year': year, 'BasePolicy': base_policy,
        }

        with st.spinner("Scoring claim across fraud and litigation models…"):
            fp, lp, score, route, tier_key, tier_label, tier_color, reasons = score_claim(claim)

        st.markdown("---")
        st.markdown('<div class="section-label">Decisioning Output</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Claim Score Report</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        col_score, col_detail = st.columns([1, 1.6], gap="large")

        # Score panel
        bar_color = tier_color
        bar_pct   = score
        route_cls = f"route-{tier_key}"
        with col_score:
            st.markdown(f"""
            <div class="score-panel">
              <div class="score-label">Composite Risk Score</div>
              <div>
                <span class="score-number">{score}</span>
                <span class="score-denom"> / 100</span>
              </div>
              <div class="score-bar-bg">
                <div class="score-bar-fill" style="width:{bar_pct}%; background:{bar_color};"></div>
              </div>
              <div class="meter-row">
                <div class="meter-box">
                  <div class="meter-title">Fraud Probability</div>
                  <div class="meter-val">{fp:.1%}</div>
                </div>
                <div class="meter-box">
                  <div class="meter-title">Litigation Risk</div>
                  <div class="meter-val">{lp:.1%}</div>
                </div>
              </div>
              <div style="margin-top:0.5rem;">
                <div class="score-label">Routing Decision</div>
                <div style="color:white;font-size:1.1rem;font-weight:700;margin-top:0.3rem;">{route}</div>
              </div>
              <div style="margin-top:0.8rem;">
                <span style="background:{tier_color};color:white;padding:0.35rem 1rem;border-radius:999px;font-size:0.78rem;font-weight:700;">{tier_label}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Detail panel
        with col_detail:
            st.markdown("**Top Fraud Risk Drivers (SHAP Explanation)**")
            for feat, val in reasons:
                direction = "↑ Raises" if val > 0 else "↓ Lowers"
                cls = "reason-up" if val > 0 else "reason-down"
                icon = "🔴" if val > 0 else "🟢"
                st.markdown(f"""
                <div class="reason-card {cls}">
                  {icon} <strong>{feat}</strong> — {direction} fraud risk
                  <span style="float:right;color:#9CA3AF;font-size:0.75rem;">SHAP = {val:+.3f}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>**Claim Summary**", unsafe_allow_html=True)
            summary_data = [
                ("Vehicle", f"{vehicle_make} · {vehicle_category}"),
                ("Cover", f"{base_policy} · {vehicle_price}"),
                ("Claimant", f"{sex}, {age} yrs · {marital_status}"),
                ("FNOL Delay", fnol_delay),
                ("FIR Filed", fir_filed),
                ("Witness", witness),
                ("Prior Claims", prior_claims),
                ("Fault", fault),
                ("Intermediary", intermediary),
            ]
            rows = "".join(f"<tr><td style='color:#6B7280;font-weight:500;'>{k}</td><td style='font-weight:600;'>{v}</td></tr>" for k,v in summary_data)
            st.markdown(f"""
            <table class="summary-table">
              <thead><tr><th>Field</th><th>Value</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)

        # Action box
        action_map = {
            'critical': ('🔴', 'Refer to SIU immediately. Do not settle. Assign senior investigator and request full documentation audit.', CRITICAL),
            'high':     ('🟠', 'Flag for ADR / Legal team. Prepare fight-or-settle brief. Route to in-house counsel within 48 hours.', HIGH),
            'medium':   ('🟡', 'Route to Standard Processing queue. Surveyor review required before payment authorisation.', MEDIUM),
            'fast':     ('🟢', 'Eligible for Fast Track Settlement. Verify documents and initiate payment within 7 working days.', FASTTRACK),
        }
        icon, action_text, action_color = action_map[tier_key]
        st.markdown(f"""
        <div style="margin-top:1.5rem;background:#F0F9FF;border:1px solid #BAE6FD;border-left:5px solid {action_color};
                    border-radius:10px;padding:1rem 1.2rem;">
          <div style="font-size:0.68rem;letter-spacing:1.5px;text-transform:uppercase;color:{action_color};font-weight:700;margin-bottom:0.4rem;">
            {icon} Recommended Action
          </div>
          <div style="font-size:0.88rem;color:#1F2937;font-weight:500;">{action_text}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Model Transparency</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Engine Architecture & Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("#### 🧠 Fraud Detection Model")
        st.markdown(f"""
        <div class="about-card">
          <h4>Model A — Fraud Detection</h4>
          <p>
            <strong>Algorithm:</strong> XGBoost Classifier with cost-sensitive weighting<br>
            <strong>Label:</strong> SIU-confirmed fraud (binary)<br>
            <strong>Metric:</strong> PR-AUC (precision-recall, accounts for 6% fraud rate)<br>
            <strong>Split:</strong> Time-ordered 80/20 — no temporal leakage<br>
            <strong>Key Features:</strong> FNOL delay, FIR filing, witness presence,
            prior claims, vehicle value, driver rating, address change pattern<br>
            <strong>Explainability:</strong> SHAP TreeExplainer — top-3 reasons per claim
          </p>
        </div>
        """, unsafe_allow_html=True)

        # Feature importance chart
        if models_ok and feat_fraud:
            importances = pd.Series(
                model_fraud.feature_importances_, index=feat_fraud
            ).sort_values(ascending=True).tail(10)

            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#F8F9FC')
            ax.set_facecolor('#F8F9FC')
            bars = ax.barh(importances.index, importances.values,
                           color=RS_BLUE, alpha=0.85, height=0.65)
            bars[-1].set_color(RS_YELLOW)
            bars[-1].set_edgecolor(RS_GOLD)
            ax.set_xlabel("Feature Importance", fontsize=9, color=RS_GRAY)
            ax.set_title("Top 10 Fraud Model Drivers", fontsize=11, fontweight='bold',
                         color=RS_NAVY, pad=12)
            ax.tick_params(axis='both', labelsize=8, colors=RS_GRAY)
            ax.spines[['top','right','left']].set_visible(False)
            ax.spines['bottom'].set_color(RS_LGRAY)
            ax.xaxis.grid(True, alpha=0.3, color=RS_LGRAY)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with col_b:
        st.markdown("#### ⚖️ Litigation Risk Model")
        st.markdown(f"""
        <div class="about-card">
          <h4>Model B — Litigation Risk</h4>
          <p>
            <strong>Algorithm:</strong> XGBoost Classifier<br>
            <strong>Label:</strong> Expert-rule litigation risk index (MACT escalation proxy)<br>
            <strong>Design:</strong> Label-leakage-safe — proxy features excluded from training set<br>
            <strong>Metric:</strong> ROC-AUC &amp; PR-AUC<br>
            <strong>Key Features:</strong> Vehicle category, cover type, driver rating,
            intermediary channel, supplementary reports, vehicle age<br>
            <strong>Business Use:</strong> Early fight-or-settle signal at FNOL —
            routes high-risk cases to ADR before MACT filing
          </p>
        </div>
        """, unsafe_allow_html=True)

        # Routing pie chart
        route_labels = ['Fast Track Settlement','Standard Processing','ADR / Legal Prep','SIU Investigation']
        route_sizes  = [62, 22, 10, 6]
        route_colors = [FASTTRACK, RS_BLUE, HIGH, CRITICAL]

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor('#F8F9FC')
        wedges, texts, autotexts = ax2.pie(
            route_sizes, labels=route_labels, colors=route_colors,
            autopct='%1.0f%%', startangle=140, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2)
        )
        for t in texts:     t.set_fontsize(8);  t.set_color(RS_GRAY)
        for a in autotexts: a.set_fontsize(8);  a.set_color('white'); a.set_fontweight('bold')
        ax2.set_title("Expected Claim Routing Distribution", fontsize=11,
                       fontweight='bold', color=RS_NAVY, pad=12)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # Architecture flow
    st.markdown("---")
    st.markdown("#### 🏗️ Engine Architecture")
    st.markdown(f"""
    <div style="background:{RS_WHITE};border:1px solid {RS_LGRAY};border-radius:12px;padding:1.5rem;margin-top:0.5rem;">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:0.5rem;flex-wrap:wrap;">
        <div style="text-align:center;flex:1;min-width:100px;">
          <div style="background:{RS_BLUE};color:white;padding:0.7rem 0.5rem;border-radius:8px;font-size:0.75rem;font-weight:700;">FNOL<br><span style='font-size:0.65rem;opacity:0.8;'>Claim Received</span></div>
        </div>
        <div style="color:{RS_YELLOW};font-size:1.5rem;font-weight:700;">→</div>
        <div style="text-align:center;flex:1;min-width:100px;">
          <div style="background:{RS_NAVY};color:white;padding:0.7rem 0.5rem;border-radius:8px;font-size:0.75rem;font-weight:700;">Data Spine<br><span style='font-size:0.65rem;opacity:0.8;'>Feature Assembly</span></div>
        </div>
        <div style="color:{RS_YELLOW};font-size:1.5rem;font-weight:700;">→</div>
        <div style="text-align:center;flex:1.2;min-width:120px;">
          <div style="background:#4338CA;color:white;padding:0.7rem 0.5rem;border-radius:8px;font-size:0.75rem;font-weight:700;">Model A + B<br><span style='font-size:0.65rem;opacity:0.8;'>Fraud · Litigation</span></div>
        </div>
        <div style="color:{RS_YELLOW};font-size:1.5rem;font-weight:700;">→</div>
        <div style="text-align:center;flex:1;min-width:100px;">
          <div style="background:#0F766E;color:white;padding:0.7rem 0.5rem;border-radius:8px;font-size:0.75rem;font-weight:700;">SHAP Layer<br><span style='font-size:0.65rem;opacity:0.8;'>Explainability</span></div>
        </div>
        <div style="color:{RS_YELLOW};font-size:1.5rem;font-weight:700;">→</div>
        <div style="text-align:center;flex:1;min-width:100px;">
          <div style="background:{FASTTRACK};color:white;padding:0.7rem 0.5rem;border-radius:8px;font-size:0.75rem;font-weight:700;">Routing<br><span style='font-size:0.65rem;opacity:0.8;'>Fast·SIU·ADR·Legal</span></div>
        </div>
      </div>
      <div style="margin-top:1rem;font-size:0.75rem;color:{RS_GRAY};text-align:center;">
        Single-insurer scope · No consortium dependency · Human-in-the-loop on every high-risk flag
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Financial impact
    st.markdown("---")
    st.markdown("#### 💰 Projected Financial Impact (per ₹1,000 Cr TP Claims Paid)")
    fin_data = {
        'Scenario': ['🐻 Bear', '📊 Base', '🐂 Bull'],
        'Leakage Recovered': ['₹5 Cr', '₹15 Cr', '₹30 Cr'],
        'Litigation Saving': ['₹1 Cr', '₹3 Cr', '₹6 Cr'],
        'Total Benefit': ['₹6 Cr', '₹18 Cr', '₹36 Cr'],
        'Build Cost': ['₹3.5 Cr', '₹3.5 Cr', '₹3.5 Cr'],
        'Payback': ['~7 mo', '~5 mo', '~3 mo'],
    }
    fin_df = pd.DataFrame(fin_data)
    st.dataframe(fin_df.set_index('Scenario'), use_container_width=True)
    st.caption("Illustrative model normalised per ₹1,000 Cr to avoid over-claiming. Build cost held constant across scenarios.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">Sundaram Pitch Fest 2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">About the Claim Decisioning Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        for title, icon, body in [
            ("The Problem", "🎯", "India's Motor TP book carries ₹96,257 Cr of outstanding liability (FY25) across 10.7 lakh open cases. Fraud leakage, MACT litigation overload and manual triage drain margin — 4.07 lakh cases are 3+ years old."),
            ("Our Solution", "⚙️", "A single-insurer AI Claim Decisioning Engine: two XGBoost models scoring fraud probability and litigation risk at FNOL, combined into a cost-sensitive composite score that auto-routes every claim to the cheapest correct path."),
            ("Why Single-Insurer", "🏛️", "Deployable on Royal Sundaram's own data — no IIB consortium dependency, no competitor coordination. The graph network layer deepens with every claim processed, creating a proprietary, compounding moat."),
        ]:
            st.markdown(f"""
            <div class="about-card">
              <h4>{icon} {title}</h4>
              <p>{body}</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        for title, icon, body in [
            ("Explainability", "🔍", "Every flag carries SHAP-derived top-3 reasons in plain language — so SIU officers, claims managers, and MACT tribunals can act on the output. No black box; every decision is auditable."),
            ("Implementation", "🗓️", "Phased 24-month roadmap: P0 data foundation → P1 fraud shadow mode → P2 litigation model → P3 network layer → P4 live routing with feedback loop and drift monitoring."),
            ("Future Vision", "🚀", "Phase 3+: graph neural networks over the garage–lawyer entity graph, underwriting-stage risk pricing via Vahan/MoRTH signals, and federated cross-insurer learning without sharing raw claim data."),
        ]:
            st.markdown(f"""
            <div class="about-card">
              <h4>{icon} {title}</h4>
              <p>{body}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="background:{RS_NAVY};border-radius:12px;padding:1.5rem 2rem;color:white;">
      <div style="font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;color:{RS_YELLOW};font-weight:700;margin-bottom:0.8rem;">Team Apex Counsel · IIT Kharagpur</div>
      <div style="display:flex;gap:2rem;flex-wrap:wrap;">
        <div><div style="font-weight:700;font-size:0.9rem;">Arunadithyan S</div><div style="font-size:0.75rem;color:rgba(255,255,255,0.6);">Indian Institute of Technology, Kharagpur</div></div>
        <div><div style="font-weight:700;font-size:0.9rem;">Azhagappan G</div><div style="font-size:0.75rem;color:rgba(255,255,255,0.6);">Indian Institute of Technology, Kharagpur</div></div>
        <div><div style="font-weight:700;font-size:0.9rem;">G Abiimukeshwar</div><div style="font-size:0.75rem;color:rgba(255,255,255,0.6);">Indian Institute of Technology, Kharagpur</div></div>
      </div>
      <div style="margin-top:1rem;font-size:0.75rem;color:rgba(255,255,255,0.5);">
        Operations · Risk &amp; Process Excellence Track &nbsp;|&nbsp; Sundaram Pitch Fest 2026 · Round 2
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer-bar">
  <span>Royal Sundaram General Insurance Co. Ltd.</span> &nbsp;·&nbsp;
  Claim Decisioning Engine &nbsp;·&nbsp;
  Built by Team Apex Counsel · IIT Kharagpur &nbsp;·&nbsp;
  Sundaram Pitch Fest 2026 &nbsp;·&nbsp;
  <span>Operations · Risk · Process Excellence</span>
</div>
""", unsafe_allow_html=True)
