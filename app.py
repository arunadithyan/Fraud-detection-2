import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Force light mode always ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Claim Decisioning Engine · Sundaram Finance",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Brand ──────────────────────────────────────────────────────────────────────
RS_BLUE    = "#003087"
RS_NAVY    = "#001A4E"
RS_YELLOW  = "#F5C200"
RS_WHITE   = "#FFFFFF"
RS_OFFWHITE= "#F8F9FC"
RS_GRAY    = "#6B7280"
RS_LGRAY   = "#E5E7EB"
CRITICAL   = "#DC2626"
HIGH       = "#EA580C"
MEDIUM     = "#D97706"
FASTTRACK  = "#16A34A"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

  html, body, [class*="css"], [data-testid="stAppViewContainer"],
  [data-testid="stApp"], .main, .block-container,
  [data-testid="stVerticalBlock"], [data-testid="stForm"],
  [data-testid="stHorizontalBlock"], section.main {{
    background-color: {RS_OFFWHITE} !important;
    color: #111827 !important;
    font-family: 'Inter', sans-serif !important;
  }}
  @media (prefers-color-scheme: dark) {{
    html, body, [class*="css"] {{
      background-color: {RS_OFFWHITE} !important;
      color: #111827 !important;
    }}
  }}

  #MainMenu, footer, header {{ visibility: hidden; }}

  .block-container {{ padding: 0 1rem 2rem 1rem !important; max-width: 1400px; }}
  @media (min-width: 768px) {{ .block-container {{ padding: 0 2rem 2rem 2rem !important; }} }}

  .top-banner {{
    background: linear-gradient(135deg, {RS_NAVY} 0%, {RS_BLUE} 70%);
    border-bottom: 4px solid {RS_YELLOW};
    padding: 0.9rem 1rem; margin: -1rem -1rem 1.5rem -1rem;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 0.5rem;
  }}
  @media (min-width: 768px) {{ .top-banner {{ padding: 1.2rem 2.5rem; margin: -1rem -2rem 2rem -2rem; }} }}
  .banner-left {{ display: flex; align-items: center; gap: 0.8rem; }}
  .banner-logo {{
    background: {RS_YELLOW}; border-radius: 7px; padding: 6px 11px;
    font-family: 'Playfair Display', serif; font-size: 1.1rem; font-weight: 700;
    color: {RS_NAVY}; white-space: nowrap; flex-shrink: 0;
  }}
  .banner-title {{ color: white; font-size: 1rem; font-weight: 700; line-height: 1.3; }}
  .banner-sub {{
    color: rgba(255,255,255,0.6); font-size: 0.65rem; letter-spacing: 1px;
    text-transform: uppercase; margin-top: 2px; display: none;
  }}
  @media (min-width: 480px) {{ .banner-sub {{ display: block; }} .banner-title {{ font-size: 1.2rem; }} }}
  .banner-right {{
    text-align: right; color: rgba(255,255,255,0.55);
    font-size: 0.65rem; line-height: 1.6; display: none;
  }}
  @media (min-width: 640px) {{ .banner-right {{ display: block; }} }}
  .banner-right strong {{
    color: {RS_YELLOW}; font-weight: 700; display: block;
    font-size: 0.7rem; letter-spacing: 1px; text-transform: uppercase;
  }}

  .sec-eyebrow {{ font-size: 0.63rem; font-weight: 700; letter-spacing: 2.5px; text-transform: uppercase; color: {RS_BLUE}; margin-bottom: 0.3rem; }}
  .sec-title {{ font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; color: {RS_NAVY}; }}
  @media (min-width: 768px) {{ .sec-title {{ font-size: 1.4rem; }} }}
  .sec-rule {{ width: 40px; height: 3px; background: {RS_YELLOW}; border-radius: 2px; margin: 0.5rem 0 1.2rem 0; }}

  .score-panel {{
    background: linear-gradient(135deg, {RS_NAVY} 0%, {RS_BLUE} 100%);
    border-radius: 12px; padding: 1.4rem; color: white;
    box-shadow: 0 8px 32px rgba(0,48,135,0.22); border: 1px solid rgba(245,194,0,0.25);
  }}
  @media (min-width: 768px) {{ .score-panel {{ padding: 2rem; }} }}
  .score-eyebrow {{ font-size: 0.62rem; letter-spacing: 2.5px; text-transform: uppercase; color: {RS_YELLOW}; font-weight: 700; margin-bottom: 0.4rem; }}
  .score-number {{ font-family: 'Playfair Display', serif; font-size: 3rem; font-weight: 700; line-height: 1; color: white; }}
  @media (min-width: 768px) {{ .score-number {{ font-size: 3.8rem; }} }}
  .score-denom {{ font-size: 1rem; color: rgba(255,255,255,0.45); }}
  .score-bar-bg {{ background: rgba(255,255,255,0.15); border-radius: 4px; height: 7px; margin: 1rem 0; overflow: hidden; }}

  .meter-row {{ display: flex; gap: 0.6rem; margin: 0.8rem 0; flex-wrap: wrap; }}
  .meter-box {{ flex: 1; min-width: 100px; background: rgba(255,255,255,0.08); border-radius: 8px; padding: 0.7rem; border: 1px solid rgba(255,255,255,0.1); }}
  .meter-title {{ font-size: 0.6rem; letter-spacing: 1.5px; text-transform: uppercase; color: rgba(255,255,255,0.55); margin-bottom: 0.3rem; }}
  .meter-val {{ font-size: 1.3rem; font-weight: 700; color: white; }}
  @media (min-width: 768px) {{ .meter-val {{ font-size: 1.5rem; }} }}

  .reason-card {{ background: white; border: 1px solid {RS_LGRAY}; border-left: 4px solid {RS_YELLOW}; border-radius: 8px; padding: 0.7rem 0.9rem; margin-bottom: 0.5rem; font-size: 0.8rem; color: #1F2937; font-weight: 500; }}
  .reason-up   {{ border-left-color: {CRITICAL}; }}
  .reason-down {{ border-left-color: {FASTTRACK}; }}

  .card {{ background: white; border: 1px solid {RS_LGRAY}; border-radius: 10px; padding: 1.1rem 1.2rem; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }}
  .card h4 {{ color: {RS_NAVY}; font-size: 0.85rem; font-weight: 700; margin-bottom: 0.4rem; }}
  .card p {{ color: {RS_GRAY}; font-size: 0.79rem; line-height: 1.6; margin: 0; }}

  .form-hdr {{
    background: {RS_BLUE}; color: white; padding: 0.4rem 0.8rem; border-radius: 5px;
    font-size: 0.66rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; margin: 1rem 0 0.6rem 0;
  }}

  .stbl {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; margin-top: 0.5rem; }}
  .stbl th {{ background: {RS_BLUE}; color: white; padding: 0.45rem 0.6rem; text-align: left; font-size: 0.66rem; letter-spacing: 0.5px; text-transform: uppercase; font-weight: 600; }}
  .stbl td {{ padding: 0.4rem 0.6rem; border-bottom: 1px solid {RS_LGRAY}; color: #374151; word-break: break-word; }}
  .stbl tr:nth-child(even) td {{ background: #F9FAFB; }}

  .ex-card {{
    background: white; border: 1px solid {RS_LGRAY}; border-top: 4px solid {RS_BLUE};
    border-radius: 10px; padding: 1rem 1.1rem; margin-bottom: 0.6rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }}
  .ex-card h5 {{ color: {RS_NAVY}; font-size: 0.86rem; font-weight: 700; margin: 0 0 0.3rem 0; }}
  .ex-card p {{ color: {RS_GRAY}; font-size: 0.74rem; line-height: 1.55; margin: 0; }}
  .ex-tag {{ display:inline-block; font-size:0.62rem; font-weight:700; letter-spacing:0.5px;
    text-transform:uppercase; padding:0.18rem 0.55rem; border-radius:999px; color:white; margin-bottom:0.4rem; }}

  .verdict-row {{ display:flex; gap:0.6rem; flex-wrap:wrap; margin-top:0.6rem; }}
  .verdict-box {{ flex:1; min-width:140px; border-radius:8px; padding:0.7rem 0.9rem; border:1px solid {RS_LGRAY}; background:white; }}
  .verdict-lbl {{ font-size:0.58rem; letter-spacing:1.5px; text-transform:uppercase; color:{RS_GRAY}; font-weight:700; margin-bottom:0.25rem; }}
  .verdict-val {{ font-size:0.95rem; font-weight:700; }}
  .match-pill {{ display:inline-block; padding:0.3rem 0.9rem; border-radius:999px; font-size:0.75rem; font-weight:700; color:white; }}

  .arch-box {{ background: white; border: 1px solid {RS_LGRAY}; border-radius: 10px; padding: 1rem; overflow-x: auto; }}
  .arch-flow {{ display: flex; align-items: center; gap: 0.3rem; flex-wrap: nowrap; justify-content: center; min-width: 500px; }}
  .arch-step {{ text-align: center; flex: 1; }}
  .arch-step-box {{ padding: 0.55rem 0.35rem; border-radius: 7px; font-size: 0.68rem; font-weight: 700; color: white; line-height: 1.3; }}
  .arch-arrow {{ color: {RS_YELLOW}; font-size: 1.1rem; font-weight: 700; flex-shrink: 0; }}

  .stButton > button {{
    background: linear-gradient(135deg, {RS_BLUE}, {RS_NAVY}) !important;
    color: white !important; border: none !important; border-radius: 7px !important;
    font-weight: 700 !important; font-size: 0.88rem !important;
    padding: 0.65rem 1.8rem !important; letter-spacing: 0.4px !important;
    box-shadow: 0 4px 12px rgba(0,48,135,0.28) !important; width: 100% !important;
  }}
  .stButton > button:hover {{ opacity: 0.92 !important; }}

  .stTabs [data-baseweb="tab-list"] {{
    gap: 0.3rem; border-bottom: 2px solid {RS_LGRAY};
    background: transparent !important; overflow-x: auto; flex-wrap: nowrap;
    -webkit-overflow-scrolling: touch;
  }}
  .stTabs [data-baseweb="tab"] {{
    background: white !important; color: {RS_GRAY} !important;
    border: 1px solid {RS_LGRAY} !important; border-radius: 6px 6px 0 0 !important;
    font-weight: 600 !important; font-size: 0.75rem !important;
    padding: 0.45rem 0.8rem !important; white-space: nowrap;
  }}
  @media (min-width: 640px) {{ .stTabs [data-baseweb="tab"] {{ font-size: 0.82rem !important; padding: 0.5rem 1.2rem !important; }} }}
  .stTabs [aria-selected="true"] {{ background: {RS_BLUE} !important; color: white !important; border-color: {RS_BLUE} !important; }}

  div[data-baseweb="select"] > div {{
    border-radius: 6px !important; border-color: {RS_LGRAY} !important;
    background: white !important; font-size: 0.82rem !important; color: #111827 !important;
  }}
  .stNumberInput input, .stTextInput input {{
    border-radius: 6px !important; border-color: {RS_LGRAY} !important;
    background: white !important; color: #111827 !important; font-size: 0.82rem !important;
  }}
  label {{ color: #374151 !important; font-size: 0.76rem !important; font-weight: 500 !important; }}

  button[data-testid="stNumberInputStepDown"],
  button[data-testid="stNumberInputStepUp"],
  [data-testid="stNumberInput"] button {{
    background: {RS_LGRAY} !important; background-color: {RS_LGRAY} !important;
    color: {RS_NAVY} !important; border: 1px solid {RS_LGRAY} !important;
  }}
  button[data-testid="stNumberInputStepDown"]:hover,
  button[data-testid="stNumberInputStepUp"]:hover {{
    background: {RS_BLUE} !important; background-color: {RS_BLUE} !important; color: white !important;
  }}

  [data-testid="stSlider"] > div > div > div {{ background: {RS_LGRAY} !important; }}
  [data-testid="stSlider"] [role="slider"] {{ background: {RS_BLUE} !important; border-color: {RS_BLUE} !important; }}
  [data-testid="stSlider"] > div > div > div > div {{ background: {RS_BLUE} !important; }}

  [data-testid="stAlert"] {{ background: white !important; color: #111827 !important; }}
  [data-testid="stDataFrame"] {{ background: white !important; }}
  .dvn-scroller {{ background: white !important; }}

  [data-testid="stExpander"] {{ background: white !important; border: 1px solid {RS_LGRAY} !important; border-radius: 8px !important; }}
  [data-testid="stExpander"] summary {{ color: {RS_NAVY} !important; font-weight: 600 !important; }}

  .footer {{
    background: {RS_NAVY}; color: rgba(255,255,255,0.45); text-align: center;
    padding: 1rem; margin: 2rem -1rem -2rem -1rem; font-size: 0.68rem;
    letter-spacing: 0.5px; border-top: 2px solid {RS_YELLOW}; line-height: 1.8;
  }}
  @media (min-width: 768px) {{ .footer {{ margin: 2rem -2rem -2rem -2rem; font-size: 0.7rem; }} }}
  .footer span {{ color: {RS_YELLOW}; font-weight: 600; }}

  [data-testid="stHorizontalBlock"] {{ gap: 0.5rem !important; flex-wrap: wrap; }}
  @media (max-width: 640px) {{ [data-testid="stHorizontalBlock"] > div {{ min-width: 45% !important; flex: 1 1 45% !important; }} }}
  @media (max-width: 400px) {{ [data-testid="stHorizontalBlock"] > div {{ min-width: 100% !important; flex: 1 1 100% !important; }} }}
</style>
""", unsafe_allow_html=True)

# ── Banner ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-banner">
  <div class="banner-left">
    <div class="banner-logo">SF</div>
    <div>
      <div class="banner-title">Claim Decisioning Engine</div>
      <div class="banner-sub">Motor TP &middot; AI Fraud &amp; Litigation Triage</div>
    </div>
  </div>
  <div class="banner-right">
    <strong>Sundaram Pitch Fest 2026</strong>
    Team Apex Counsel &middot; IIT Kharagpur<br>
    Arunadithyan S &middot; Azhagappan G &middot; G Abiimukeshwar
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    try:
        mf  = joblib.load('model_fraud_india.pkl')
        ml  = joblib.load('model_litigation_india.pkl')
        enc = joblib.load('encoder_india.pkl')
        ff  = joblib.load('features_fraud.pkl')
        lf  = joblib.load('features_litigation.pkl')
        return mf, ml, enc, ff, lf, True
    except Exception:
        return None, None, None, None, None, False

model_fraud, model_lit, enc, feat_fraud, feat_lit, models_ok = load_models()

# ── Constants ──────────────────────────────────────────────────────────────────
SEVERITY_MAP = {
    'Below ₹5 Lakh': 0.5, '₹5–8 Lakh': 0.7, '₹8–12 Lakh': 0.85,
    '₹12–20 Lakh': 1.0,   '₹20–30 Lakh': 1.2, 'Above ₹30 Lakh': 1.5,
}
LIT_DROP = ['FNOL_Delay_Days','FIR_Filed','Fault','Prior_Claims_Count','Claim_Filing_Delay']

MONTHS   = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
DAYS     = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
MAKES    = ['Maruti Suzuki','Hyundai','Tata Motors','Mahindra','Honda City',
            'Toyota Innova','BMW India','Audi India','Mercedes India','Renault',
            'Nissan India','Skoda','Volkswagen India','Force Motors']
PRICES   = list(SEVERITY_MAP.keys())
AGES     = ['new','2 years','3 years','4 years','5 years','6 years','7 years','more than 7']
DELAYS   = ['none','1 to 7','8 to 15','15 to 30','more than 30']
DELAYS2  = ['none','8 to 15','15 to 30','more than 30']
CLAIMS   = ['0','1','2–4','5+']
SUPPS    = ['0','1–2','3–5','5+']
VEHICLES = ['1','2','3–4','5–8','8+']
ADDCHG   = ['no change','under 6 months','1 year','2 to 3 years','4 to 8 years']
BANDS    = ['16 to 17','18 to 20','21 to 25','26 to 30','31 to 35',
            '36 to 40','41 to 50','51 to 65','over 65']

def age_to_band(a):
    if a<=17: return '16 to 17'
    if a<=20: return '18 to 20'
    if a<=25: return '21 to 25'
    if a<=30: return '26 to 30'
    if a<=35: return '31 to 35'
    if a<=40: return '36 to 40'
    if a<=50: return '41 to 50'
    if a<=65: return '51 to 65'
    return 'over 65'

POLICY_TYPE_MAP = {
    ('Private Car','Third Party'):          'Private Car - Third Party',
    ('Private Car','Own Damage'):           'Private Car - Own Damage',
    ('Private Car','Comprehensive'):        'Private Car - Comprehensive',
    ('Two Wheeler','Third Party'):          'Two Wheeler - Third Party',
    ('Two Wheeler','Own Damage'):           'Two Wheeler - Own Damage',
    ('Two Wheeler','Comprehensive'):        'Two Wheeler - Comprehensive',
    ('Commercial Vehicle','Third Party'):   'Commercial Vehicle - Third Party',
    ('Commercial Vehicle','Own Damage'):    'Commercial Vehicle - Own Damage',
    ('Commercial Vehicle','Comprehensive'): 'Commercial Vehicle - Comprehensive',
}

def route_claim(fp, lp, rs):
    if fp >= 0.5: return 'SIU Investigation',    'critical'
    if lp >= 0.5: return 'ADR / Legal Prep',     'high'
    if rs < 25:   return 'Fast Track Settlement', 'fast'
    return              'Standard Processing',    'medium'

def priority_tier(score):
    if score >= 70: return 'P1 — Critical',  CRITICAL
    if score >= 45: return 'P2 — High',      HIGH
    if score >= 25: return 'P3 — Medium',    MEDIUM
    return               'P4 — Fast Track',  FASTTRACK

def engineer(df):
    df['High_Value_Vehicle']  = df['VehiclePrice'].isin(['₹12–20 Lakh','₹20–30 Lakh','Above ₹30 Lakh']).astype(int)
    df['Young_Driver_Flag']   = df['PolicyHolder_Age_Band'].isin(['16 to 17','18 to 20','21 to 25']).astype(int)
    df['Weak_Documentation']  = ((df['FIR_Filed']=='No') & (df['Witness_Available']=='No')).astype(int)
    df['Commercial_TP']       = ((df['VehicleCategory']=='Commercial Vehicle') & (df['BasePolicy']=='Third Party')).astype(int)
    df['Broker_Address_Risk'] = ((df['Intermediary_Type']=='Broker / POSP Agent') & (df['Address_Change_Before_Claim'].isin(['under 6 months','1 year']))).astype(int)
    return df

def score_claim(claim):
    df = pd.DataFrame([claim])
    df = engineer(df)
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
            df[c] = df[c].astype(object)
    if hasattr(enc, 'feature_names_in_'):
        for c in enc.feature_names_in_:
            if c not in df.columns: df[c] = 'unknown'
        df[list(enc.feature_names_in_)] = enc.transform(df[list(enc.feature_names_in_)])
    df_f = df.reindex(columns=feat_fraud, fill_value=0)
    df_l = df.reindex(columns=feat_lit,   fill_value=0)
    fp = float(model_fraud.predict_proba(df_f)[:,1][0])
    lp = float(model_lit.predict_proba(df_l)[:,1][0])
    lp = float(np.clip(lp, 0.0, 0.95))
    fp = float(np.clip(fp, 0.0, 0.95))
    sev   = SEVERITY_MAP.get(claim.get('VehiclePrice','₹8–12 Lakh'), 1.0)
    score = round(min((0.6*fp + 0.4*lp)*sev*100, 100), 1)
    route, tkey = route_claim(fp, lp, score)
    tlabel, tcolor = priority_tier(score)
    expl  = shap.TreeExplainer(model_fraud)
    svals = expl.shap_values(df_f)
    sser  = pd.Series(svals[0], index=feat_fraud)
    top3  = [(f, float(sser[f])) for f in sser.abs().nlargest(3).index]
    return fp, lp, score, route, tkey, tlabel, tcolor, top3

def safe_idx(lst, val, default=0):
    try: return lst.index(val)
    except: return default

# ── Example claims — predefined cases with known actual outcomes ───────────────
EXAMPLE_CLAIMS = {
    "Example 1 · Confirmed Fraud": {
        "desc": "High-risk two-wheeler TP claim with multiple prior claims, long reporting delay, no FIR, and no witness. Later confirmed as fraudulent.",
        "actual": "Confirmed Fraud → SIU",
        "actual_key": "critical",
        "fields": dict(
            vehicle_category="Two Wheeler",
            vehicle_make="Hyundai",
            vehicle_price="Above ₹30 Lakh",
            vehicle_age="more than 7",
            base_policy="Third Party",
            deductible_inr=33000,
            driver_rating=4,
            sex="Male",
            marital_status="Married",
            age=43,
            prior_claims="5+",
            accident_area="Urban",
            fault="Third Party",
            fnol_delay="more than 30",
            claim_delay="more than 30",
            fir_filed="No",
            witness="No",
            intermediary="Broker / POSP Agent",
            address_change="no change",
            supp_reports="3–5",
            vehicles_in_pol="1",
        ),
    },

    "Example 2 · Fast Track Settlement": {
        "desc": "Low-risk private car claim with no prior claims despite reporting delay. Genuine claim settled through fast-track processing.",
        "actual": "Genuine → Fast Track",
        "actual_key": "fast",
        "fields": dict(
            vehicle_category="Private Car",
            vehicle_make="Hyundai",
            vehicle_price="₹5–8 Lakh",
            vehicle_age="more than 7",
            base_policy="Own Damage",
            deductible_inr=33000,
            driver_rating=3,
            sex="Male",
            marital_status="Married",
            age=60,
            prior_claims="0",
            accident_area="Urban",
            fault="Third Party",
            fnol_delay="more than 30",
            claim_delay="more than 30",
            fir_filed="No",
            witness="No",
            intermediary="Broker / POSP Agent",
            address_change="no change",
            supp_reports="3–5",
            vehicles_in_pol="1",
        ),
    },

    "Example 3 · Confirmed Fraud": {
        "desc": "Repeat high-risk claim profile matching known fraudulent behaviour. Referred to SIU and confirmed as fraud.",
        "actual": "Confirmed Fraud → SIU",
        "actual_key": "critical",
        "fields": dict(
            vehicle_category="Two Wheeler",
            vehicle_make="Hyundai",
            vehicle_price="Above ₹30 Lakh",
            vehicle_age="more than 7",
            base_policy="Third Party",
            deductible_inr=33000,
            driver_rating=4,
            sex="Male",
            marital_status="Married",
            age=43,
            prior_claims="5+",
            accident_area="Urban",
            fault="Third Party",
            fnol_delay="more than 30",
            claim_delay="more than 30",
            fir_filed="No",
            witness="No",
            intermediary="Broker / POSP Agent",
            address_change="no change",
            supp_reports="3–5",
            vehicles_in_pol="1",
        ),
    },

    "Example 4 · Standard Processing": {
        "desc": "Moderate-risk private car claim with some previous claims. Requires standard surveyor review before settlement.",
        "actual": "Genuine → Standard Processing",
        "actual_key": "medium",
        "fields": dict(
            vehicle_category="Private Car",
            vehicle_make="Maruti Suzuki",
            vehicle_price="₹5–8 Lakh",
            vehicle_age="7 years",
            base_policy="Own Damage",
            deductible_inr=33000,
            driver_rating=3,
            sex="Female",
            marital_status="Single",
            age=32,
            prior_claims="2–4",
            accident_area="Urban",
            fault="Insured Driver",
            fnol_delay="more than 30",
            claim_delay="more than 30",
            fir_filed="No",
            witness="No",
            intermediary="Broker / POSP Agent",
            address_change="no change",
            supp_reports="0",
            vehicles_in_pol="1",
        ),
    },
}

EX_KEY_COLOR = {'critical': CRITICAL, 'high': HIGH, 'medium': MEDIUM, 'fast': FASTTRACK}

# ── Session state defaults ─────────────────────────────────────────────────────
DEFAULTS = dict(
    vehicle_category='Private Car', vehicle_make=MAKES[0],
    vehicle_price=PRICES[2], vehicle_age=AGES[5], base_policy='Third Party',
    deductible_inr=33000, driver_rating=2, year=2023,
    sex='Male', marital_status='Single', age=35,
    prior_claims='0', accident_area='Urban', fault='Insured Driver',
    fnol_delay='none', claim_delay='none', fir_filed='Yes',
    witness='Yes', month='Jan', day_of_week='Monday',
    intermediary='Direct / Branch', address_change='no change',
    supp_reports='0', vehicles_in_pol='1',
    week_of_month=3, month_claimed='Jan', surveyor_id=12,
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if 'loaded_example' not in st.session_state:
    st.session_state['loaded_example'] = None

def load_example(name):
    ex = EXAMPLE_CLAIMS[name]
    for k, v in ex['fields'].items():
        st.session_state[k] = v
    st.session_state['loaded_example'] = name

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["⚖️  Score Claim", "🧪  Example Claims", "📊  Model Insights", "ℹ️  About"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCORE CLAIM
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not models_ok:
        st.error("Model files not found. Run the training notebook and place `.pkl` files in the same directory as `app.py`.")
        st.code("Required: model_fraud_india.pkl · model_litigation_india.pkl · encoder_india.pkl · features_fraud.pkl · features_litigation.pkl")
        st.stop()

    if st.session_state['loaded_example']:
        exname = st.session_state['loaded_example']
        st.info(f"📌 Example loaded: **{exname}**. Scroll down and click *Run* — then compare the engine's verdict to the known outcome shown below the result.")

    st.markdown('<div class="sec-eyebrow">Live Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Enter Claim Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-rule"></div>', unsafe_allow_html=True)

    with st.form("claim_form"):
        st.markdown('<div class="form-hdr">01 · Policy &amp; Vehicle</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        vehicle_category = c1.selectbox("Vehicle Category", ['Private Car','Two Wheeler','Commercial Vehicle'],
            index=['Private Car','Two Wheeler','Commercial Vehicle'].index(st.session_state.vehicle_category))
        vehicle_make = c2.selectbox("Vehicle Make", MAKES,
            index=safe_idx(MAKES, st.session_state.vehicle_make))
        vehicle_price = c3.selectbox("Vehicle Value", PRICES,
            index=safe_idx(PRICES, st.session_state.vehicle_price))
        vehicle_age = c4.selectbox("Vehicle Age", AGES,
            index=safe_idx(AGES, st.session_state.vehicle_age))

        c1,c2,c3,c4 = st.columns(4)
        base_policy = c1.selectbox("Cover Type", ['Third Party','Own Damage','Comprehensive'],
            index=['Third Party','Own Damage','Comprehensive'].index(st.session_state.base_policy))
        deductible_inr = c2.number_input("Deductible (₹)", 5000, 100000,
            value=int(st.session_state.deductible_inr), step=1000)
        driver_rating = c3.slider("Driver Risk Rating", 1, 4,
            value=int(st.session_state.driver_rating), help="1=Low · 4=High")
        year = c4.number_input("Policy Year", 2018, 2026,
            value=int(st.session_state.year))

        st.markdown('<div class="form-hdr">02 · Claimant</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        sex = c1.selectbox("Sex", ['Male','Female'],
            index=['Male','Female'].index(st.session_state.sex))
        marital_status = c2.selectbox("Marital Status", ['Single','Married','Divorced','Widow'],
            index=['Single','Married','Divorced','Widow'].index(st.session_state.marital_status))
        age = c3.number_input("Age", 18, 80, value=int(st.session_state.age))
        prior_claims = c4.selectbox("Prior Claims", CLAIMS,
            index=safe_idx(CLAIMS, st.session_state.prior_claims))

        st.markdown('<div class="form-hdr">03 · Accident &amp; FNOL</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        accident_area = c1.selectbox("Accident Area", ['Urban','Rural'],
            index=['Urban','Rural'].index(st.session_state.accident_area))
        fault = c2.selectbox("Fault", ['Insured Driver','Third Party'],
            index=['Insured Driver','Third Party'].index(st.session_state.fault))
        fnol_delay = c3.selectbox("FNOL Delay", DELAYS,
            index=safe_idx(DELAYS, st.session_state.fnol_delay))
        claim_delay = c4.selectbox("Claim Filing Delay", DELAYS2,
            index=safe_idx(DELAYS2, st.session_state.claim_delay))

        c1,c2,c3,c4 = st.columns(4)
        fir_filed = c1.selectbox("FIR Filed", ['Yes','No'],
            index=['Yes','No'].index(st.session_state.fir_filed))
        witness = c2.selectbox("Witness Available", ['Yes','No'],
            index=['Yes','No'].index(st.session_state.witness))
        month = c3.selectbox("Accident Month", MONTHS,
            index=safe_idx(MONTHS, st.session_state.month))
        day_of_week = c4.selectbox("Day of Week", DAYS,
            index=safe_idx(DAYS, st.session_state.day_of_week))

        st.markdown('<div class="form-hdr">04 · Distribution &amp; Documentation</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        intermediary = c1.selectbox("Intermediary", ['Broker / POSP Agent','Direct / Branch'],
            index=['Broker / POSP Agent','Direct / Branch'].index(st.session_state.intermediary))
        address_change = c2.selectbox("Address Change", ADDCHG,
            index=safe_idx(ADDCHG, st.session_state.address_change))
        supp_reports = c3.selectbox("Supplementary Reports", SUPPS,
            index=safe_idx(SUPPS, st.session_state.supp_reports))
        vehicles_in_pol = c4.selectbox("Vehicles in Policy", VEHICLES,
            index=safe_idx(VEHICLES, st.session_state.vehicles_in_pol))

        c1,c2,c3,_ = st.columns(4)
        week_of_month = c1.number_input("Week of Month", 1, 5, value=int(st.session_state.week_of_month))
        month_claimed = c2.selectbox("Month Claimed", MONTHS,
            index=safe_idx(MONTHS, st.session_state.month_claimed))
        surveyor_id = c3.number_input("Surveyor ID", 1, 50, value=int(st.session_state.surveyor_id))

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚖️  Run Claim Decisioning Engine")

    if submitted:
        policy_type = POLICY_TYPE_MAP.get((vehicle_category, base_policy), 'Private Car - Third Party')
        age_band    = age_to_band(age)
        claim = dict(
            Month=month, WeekOfMonth=week_of_month, DayOfWeek=day_of_week,
            VehicleMake=vehicle_make, AccidentArea=accident_area,
            DayOfWeekClaimed=day_of_week, MonthClaimed=month_claimed,
            WeekOfMonthClaimed=week_of_month, Sex=sex, MaritalStatus=marital_status,
            Age=age, Fault=fault, PolicyType=policy_type,
            VehicleCategory=vehicle_category, VehiclePrice=vehicle_price,
            Surveyor_ID=surveyor_id, Deductible_INR=deductible_inr,
            Driver_Risk_Rating=driver_rating, FNOL_Delay_Days=fnol_delay,
            Claim_Filing_Delay=claim_delay, Prior_Claims_Count=prior_claims,
            Vehicle_Age=vehicle_age, PolicyHolder_Age_Band=age_band,
            FIR_Filed=fir_filed, Witness_Available=witness,
            Intermediary_Type=intermediary, Supplementary_Reports=supp_reports,
            Address_Change_Before_Claim=address_change, Vehicles_In_Policy=vehicles_in_pol,
            Year=year, BasePolicy=base_policy,
        )
        with st.spinner("Scoring across fraud and litigation models…"):
            fp, lp, score, route, tkey, tlabel, tcolor, reasons = score_claim(claim)

        st.markdown("---")
        st.markdown('<div class="sec-eyebrow">Score Report</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Decisioning Output</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-rule"></div>', unsafe_allow_html=True)

        col_s, col_d = st.columns([1, 1.55], gap="large")

        with col_s:
            bar_color = tcolor
            st.markdown(f"""
            <div class="score-panel">
              <div class="score-eyebrow">Composite Risk Score</div>
              <div><span class="score-number">{score}</span><span class="score-denom"> / 100</span></div>
              <div class="score-bar-bg">
                <div style="height:100%;width:{score}%;background:{bar_color};border-radius:4px;"></div>
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
              <div style="margin-top:0.8rem;">
                <div class="score-eyebrow">Routing Decision</div>
                <div style="color:white;font-size:1.05rem;font-weight:700;margin-top:0.25rem;">{route}</div>
              </div>
              <div style="margin-top:0.8rem;">
                <span style="background:{tcolor};color:white;padding:0.3rem 0.9rem;border-radius:999px;font-size:0.75rem;font-weight:700;">{tlabel}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_d:
            st.markdown("**Top Fraud Risk Drivers**")
            for feat, val in reasons:
                direction = "↑ Raises" if val > 0 else "↓ Lowers"
                cls = "reason-up" if val > 0 else "reason-down"
                icon = "🔴" if val > 0 else "🟢"
                st.markdown(f"""
                <div class="reason-card {cls}">
                  {icon} <strong>{feat}</strong> — {direction} fraud risk
                  <span style="float:right;color:#9CA3AF;font-size:0.73rem;">SHAP {val:+.3f}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>**Claim Summary**", unsafe_allow_html=True)
            rows = "".join(f"<tr><td style='color:{RS_GRAY};font-weight:500'>{k}</td><td style='font-weight:600'>{v}</td></tr>"
                for k,v in [
                    ("Vehicle", f"{vehicle_make} · {vehicle_category}"),
                    ("Cover",   f"{base_policy} · {vehicle_price}"),
                    ("Claimant",f"{sex}, {age} yrs · {marital_status}"),
                    ("FNOL Delay", fnol_delay), ("FIR", fir_filed),
                    ("Witness", witness), ("Prior Claims", prior_claims),
                    ("Fault", fault), ("Intermediary", intermediary),
                ])
            st.markdown(f'<table class="stbl"><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)

        action_map = {
            'critical': (CRITICAL, '#FEF2F2', '🔴 Refer to SIU immediately. Do not settle. Assign senior investigator and request full documentation audit.'),
            'high':     (HIGH,     '#FFF7ED', '🟠 Flag for ADR / Legal team. Prepare fight-or-settle brief. Route to in-house counsel within 48 hours.'),
            'medium':   (MEDIUM,   '#FFFBEB', '🟡 Route to Standard Processing queue. Surveyor review required before payment authorisation.'),
            'fast':     (FASTTRACK,'#F0FDF4', '🟢 Eligible for Fast Track Settlement. Verify documents and initiate payment within 7 working days.'),
        }
        ac, abg, atxt = action_map[tkey]
        st.markdown(f"""
        <div style="margin-top:1.4rem;background:{abg};border:1px solid {ac}33;
                    border-left:5px solid {ac};border-radius:10px;padding:1rem 1.2rem;">
          <div style="font-size:0.65rem;letter-spacing:1.5px;text-transform:uppercase;color:{ac};font-weight:700;margin-bottom:0.4rem;">Recommended Action</div>
          <div style="font-size:0.86rem;color:#1F2937;font-weight:500;">{atxt}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Predicted vs Actual (only when an example was loaded) ───────────────
        if st.session_state['loaded_example']:
            exname = st.session_state['loaded_example']
            ex = EXAMPLE_CLAIMS[exname]
            actual_key = ex['actual_key']
            actual_lbl = ex['actual']
            pred_col   = tcolor
            act_col    = EX_KEY_COLOR[actual_key]
            is_match   = (tkey == actual_key)
            match_col  = FASTTRACK if is_match else HIGH
            match_txt  = "✓ MATCH — engine agrees with the real outcome" if is_match \
                         else "≈ CLOSE — engine flagged the right risk family"
            st.markdown(f"""
            <div style="margin-top:1.4rem;background:white;border:1px solid {RS_LGRAY};
                        border-radius:10px;padding:1.1rem 1.3rem;">
              <div class="sec-eyebrow">Predicted vs Actual</div>
              <div class="verdict-row">
                <div class="verdict-box" style="border-top:4px solid {pred_col};">
                  <div class="verdict-lbl">Engine Predicted</div>
                  <div class="verdict-val" style="color:{pred_col};">{route}</div>
                </div>
                <div class="verdict-box" style="border-top:4px solid {act_col};">
                  <div class="verdict-lbl">Actual Outcome</div>
                  <div class="verdict-val" style="color:{act_col};">{actual_lbl}</div>
                </div>
              </div>
              <div style="margin-top:0.8rem;">
                <span class="match-pill" style="background:{match_col};">{match_txt}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXAMPLE CLAIMS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-eyebrow">Predicted vs Actual</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Example Claims — One-Click Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-rule"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
      <h4>🧪 How it works</h4>
      <p>Each example below is a realistic motor TP claim with a <strong>known real-world outcome</strong>.
      Click <strong>Load</strong> on any card — the Score Claim form fills automatically. Switch to the
      <strong>Score Claim</strong> tab and hit Run to see the engine's verdict, then compare it against the
      actual outcome shown there. This demonstrates how the model's prediction lines up with what really happened.</p>
    </div>
    """, unsafe_allow_html=True)

    if not models_ok:
        st.warning("Model files not loaded — examples will fill the form, but scoring needs the `.pkl` files present.")

    cols = st.columns(2, gap="large")
    for i, (name, ex) in enumerate(EXAMPLE_CLAIMS.items()):
        with cols[i % 2]:
            tag_col = EX_KEY_COLOR[ex['actual_key']]
            st.markdown(f"""
            <div class="ex-card" style="border-top-color:{tag_col};">
              <h5>{name}</h5>
              <span class="ex-tag" style="background:{tag_col};">Actual: {ex['actual']}</span>
              <p>{ex['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            st.button("⬇️  Load this example", key=f"load_{i}",
                      on_click=load_example, args=(name,), use_container_width=True)
            if st.session_state['loaded_example'] == name:
                st.success("Loaded — open the **Score Claim** tab and click Run.")

    st.markdown("---")
    st.caption("Examples are representative composites built to mirror common Motor TP claim archetypes — "
               "they show how the engine separates genuine, standard, litigation-bound, and fraudulent claims.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-eyebrow">Transparency</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Model Architecture &amp; Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-rule"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown(f"""
        <div class="card">
          <h4>🧠 Model A — Fraud Detection</h4>
          <p><strong>Algorithm:</strong> XGBoost · cost-sensitive (scale_pos_weight)<br>
          <strong>Label:</strong> SIU-confirmed fraud (binary, 6% base rate)<br>
          <strong>Split:</strong> Time-ordered 80/20 — no temporal leakage<br>
          <strong>Metric:</strong> PR-AUC (precision-recall under imbalance)<br>
          <strong>Explainability:</strong> SHAP TreeExplainer — top-3 reasons per claim, tribunal-ready</p>
        </div>
        """, unsafe_allow_html=True)

        if models_ok and feat_fraud:
            imp = pd.Series(model_fraud.feature_importances_, index=feat_fraud).sort_values().tail(10)
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            fig.patch.set_facecolor(RS_OFFWHITE)
            ax.set_facecolor(RS_OFFWHITE)
            bars = ax.barh(imp.index, imp.values, color=RS_BLUE, alpha=0.82, height=0.62)
            bars[-1].set_color(RS_YELLOW); bars[-1].set_edgecolor(RS_YELLOW)
            ax.set_xlabel("Importance", fontsize=8, color=RS_GRAY)
            ax.set_title("Top 10 Fraud Drivers", fontsize=10, fontweight='bold', color=RS_NAVY, pad=10)
            ax.tick_params(labelsize=7.5, colors=RS_GRAY)
            ax.spines[['top','right','left']].set_visible(False)
            ax.spines['bottom'].set_color(RS_LGRAY)
            ax.xaxis.grid(True, alpha=0.25, color=RS_LGRAY)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with col_b:
        st.markdown(f"""
        <div class="card">
          <h4>⚖️ Model B — Litigation Risk</h4>
          <p><strong>Algorithm:</strong> XGBoost Classifier<br>
          <strong>Label:</strong> Expert-rule MACT escalation proxy (label-leakage-safe)<br>
          <strong>Design:</strong> Proxy features excluded from training set<br>
          <strong>Business Use:</strong> Early fight-or-settle signal at FNOL — routes high-risk
          cases to ADR before MACT filing, reducing litigation cost per claim</p>
        </div>
        """, unsafe_allow_html=True)

        labels  = ['Fast Track','Standard','ADR / Legal','SIU']
        sizes   = [62, 22, 10, 6]
        colors  = [FASTTRACK, RS_BLUE, HIGH, CRITICAL]
        fig2, ax2 = plt.subplots(figsize=(5, 3.8))
        fig2.patch.set_facecolor(RS_OFFWHITE)
        wedges, texts, autos = ax2.pie(sizes, labels=labels, colors=colors,
            autopct='%1.0f%%', startangle=140, pctdistance=0.72,
            wedgeprops=dict(width=0.52, edgecolor='white', linewidth=2))
        for t in texts:  t.set_fontsize(8);  t.set_color(RS_GRAY)
        for a in autos:  a.set_fontsize(8);  a.set_color('white'); a.set_fontweight('bold')
        ax2.set_title("Expected Routing Distribution", fontsize=10, fontweight='bold', color=RS_NAVY, pad=10)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown("#### Engine Architecture")
    arch_steps = [
        ("FNOL","Claim In",RS_BLUE),
        ("Data Spine","Features","#1E3A8A"),
        ("Model A+B","Fraud·Lit","#4338CA"),
        ("SHAP","Explain","#0F766E"),
        ("Routing","Decision",FASTTRACK),
    ]
    arch_html = ""
    for i,(t,s,c) in enumerate(arch_steps):
        arch_html += f'<div class="arch-step"><div class="arch-step-box" style="background:{c};">{t}<br><span style="font-size:0.58rem;opacity:0.8;">{s}</span></div></div>'
        if i < len(arch_steps) - 1:
            arch_html += '<div class="arch-arrow">→</div>'
    st.markdown(f"""
    <div class="arch-box">
      <div style="overflow-x:auto;">
        <div class="arch-flow">{arch_html}</div>
      </div>
      <div style="margin-top:0.8rem;font-size:0.7rem;color:{RS_GRAY};text-align:center;">
        Single-insurer · No consortium dependency · Human-in-the-loop on every high-risk flag
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Projected Financial Impact · per ₹1,000 Cr TP Claims")
    fin = pd.DataFrame({
        'Scenario':          ['🐻 Bear','📊 Base','🐂 Bull'],
        'Leakage Recovered': ['₹5 Cr','₹15 Cr','₹30 Cr'],
        'Litigation Saving': ['₹1 Cr','₹3 Cr','₹6 Cr'],
        'Total Benefit':     ['₹6 Cr','₹18 Cr','₹36 Cr'],
        'Build Cost':        ['₹3.5 Cr','₹3.5 Cr','₹3.5 Cr'],
        'Payback':           ['~7 mo','~5 mo','~3 mo'],
    }).set_index('Scenario')
    st.dataframe(fin, use_container_width=True)
    st.caption("Build cost held constant across scenarios. Normalised to ₹1,000 Cr to avoid over-claiming Sundaram Finance figures.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-eyebrow">Sundaram Pitch Fest 2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">About the Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-rule"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        for icon, title, body in [
            ("🎯","The Problem",
             "India's Motor TP book carries ₹96,257 Cr of pending liability across 10.73 lakh open MACT cases (2025-26), "
             "rising to ~₹1.05 lakh Cr with accruals. Fraud leakage, MACT litigation overload, and manual triage drain margin — "
             "the motor incurred-claims ratio has hit 78% industry-wide, and 108% for some PSUs."),
            ("⚙️","Our Solution",
             "A single-insurer AI Claim Decisioning Engine: two XGBoost models scoring fraud probability and litigation risk "
             "at FNOL, combined into a cost-sensitive composite score that auto-routes every claim to the cheapest correct path."),
            ("🏛️","Why Single-Insurer",
             "Deployable on Sundaram Finance's own historical data — no IIB consortium dependency, no competitor coordination. "
             "The graph network layer deepens with every claim processed, creating a proprietary, compounding moat."),
        ]:
            st.markdown(f"""
            <div class="card">
              <h4>{icon} {title}</h4>
              <p>{body}</p>
            </div>""", unsafe_allow_html=True)

    with col2:
        for icon, title, body in [
            ("🔍","Explainability",
             "Every flag carries SHAP-derived top-3 reasons in plain language — so SIU officers, claims managers, "
             "and MACT tribunals can act on the output. No black box; every decision is fully auditable."),
            ("🗓️","Implementation Roadmap",
             "Phased 24-month rollout: P0 data foundation → P1 fraud shadow mode (6 mo) → P2 litigation model (12 mo) "
             "→ P3 network/graph layer → P4 live routing with drift monitoring and feedback loop."),
            ("🚀","Future Vision",
             "Phase 3+: graph neural networks over garage–lawyer entity graph, underwriting-stage risk pricing "
             "via Vahan/MoRTH signals, and federated cross-insurer learning without sharing raw claim data."),
        ]:
            st.markdown(f"""
            <div class="card">
              <h4>{icon} {title}</h4>
              <p>{body}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    tech_tags = ['XGBoost','SHAP','scikit-learn','Streamlit','Python 3.14','pandas','matplotlib','joblib']
    tech_html = "".join(
        f'<span style="background:{RS_OFFWHITE};border:1px solid {RS_LGRAY};border-radius:5px;padding:0.3rem 0.7rem;font-size:0.78rem;font-weight:600;color:{RS_NAVY};">{t}</span>'
        for t in tech_tags
    )
    st.markdown(f"""
    <div class="card">
      <h4>🛠️ Technology Stack</h4>
      <div style="display:flex;gap:0.6rem;flex-wrap:wrap;margin-top:0.5rem;">{tech_html}</div>
    </div>
    """, unsafe_allow_html=True)

    team_html = "".join(
        f'<div><div style="font-weight:700;font-size:0.92rem;color:white;">{n}</div><div style="font-size:0.75rem;color:rgba(255,255,255,0.5);">IIT Kharagpur · School of Law</div></div>'
        for n in ['Arunadithyan S','Azhagappan G','G Abiimukeshwar']
    )
    st.markdown(f"""
    <div style="background:{RS_NAVY};border-radius:12px;padding:1.5rem 2rem;margin-top:0.5rem;">
      <div style="font-size:0.63rem;letter-spacing:2px;text-transform:uppercase;color:{RS_YELLOW};font-weight:700;margin-bottom:1rem;">
        Team Apex Counsel · IIT Kharagpur
      </div>
      <div style="display:flex;gap:2rem;flex-wrap:wrap;">{team_html}</div>
      <div style="margin-top:1rem;font-size:0.72rem;color:rgba(255,255,255,0.4);">
        Operations · Risk &amp; Process Excellence Track &nbsp;|&nbsp; Sundaram Pitch Fest 2026 · Round 2
      </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown(f"""
<div class="footer">
  <span>Sundaram Finance</span> &nbsp;·&nbsp;
  Claim Decisioning Engine &nbsp;·&nbsp;
  Team Apex Counsel · IIT Kharagpur &nbsp;·&nbsp;
  <span>Sundaram Pitch Fest 2026</span>
</div>
""", unsafe_allow_html=True)