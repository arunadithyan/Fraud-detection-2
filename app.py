import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, io, re
warnings.filterwarnings('ignore')

# ── Force light mode always ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Claim Decisioning Engine · Royal Sundaram",
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

  /* ── Force light mode everywhere ── */
  html, body, [class*="css"], [data-testid="stAppViewContainer"],
  [data-testid="stApp"], .main, .block-container,
  [data-testid="stVerticalBlock"] {{
    background-color: {RS_OFFWHITE} !important;
    color: #111827 !important;
    font-family: 'Inter', sans-serif !important;
  }}

  /* Kill dark mode media query overrides */
  @media (prefers-color-scheme: dark) {{
    html, body, [class*="css"] {{
      background-color: {RS_OFFWHITE} !important;
      color: #111827 !important;
    }}
  }}

  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding: 0 2rem 2rem 2rem !important; max-width: 1400px; }}

  /* ── Banner ── */
  .top-banner {{
    background: linear-gradient(135deg, {RS_NAVY} 0%, {RS_BLUE} 70%);
    border-bottom: 4px solid {RS_YELLOW};
    padding: 1.2rem 2.5rem;
    margin: -1rem -2rem 2rem -2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .banner-logo {{
    background: {RS_YELLOW};
    border-radius: 8px;
    padding: 7px 14px;
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: {RS_NAVY};
    margin-right: 1.2rem;
    white-space: nowrap;
  }}
  .banner-title {{ color: white; font-size: 1.25rem; font-weight: 700; }}
  .banner-sub {{ color: rgba(255,255,255,0.6); font-size: 0.72rem; letter-spacing: 1.5px; text-transform: uppercase; margin-top: 2px; }}
  .banner-right {{ text-align: right; color: rgba(255,255,255,0.55); font-size: 0.72rem; line-height: 1.7; }}
  .banner-right strong {{ color: {RS_YELLOW}; font-weight: 700; display: block; font-size: 0.75rem; letter-spacing: 1px; text-transform: uppercase; }}

  /* ── Section ── */
  .sec-eyebrow {{ font-size: 0.63rem; font-weight: 700; letter-spacing: 2.5px; text-transform: uppercase; color: {RS_BLUE}; margin-bottom: 0.3rem; }}
  .sec-title {{ font-family: 'Playfair Display', serif; font-size: 1.4rem; font-weight: 700; color: {RS_NAVY}; }}
  .sec-rule {{ width: 40px; height: 3px; background: {RS_YELLOW}; border-radius: 2px; margin: 0.5rem 0 1.5rem 0; }}

  /* ── Score panel ── */
  .score-panel {{
    background: linear-gradient(135deg, {RS_NAVY} 0%, {RS_BLUE} 100%);
    border-radius: 14px;
    padding: 2rem;
    color: white;
    box-shadow: 0 8px 32px rgba(0,48,135,0.22);
    border: 1px solid rgba(245,194,0,0.25);
  }}
  .score-eyebrow {{ font-size: 0.62rem; letter-spacing: 2.5px; text-transform: uppercase; color: {RS_YELLOW}; font-weight: 700; margin-bottom: 0.4rem; }}
  .score-number {{ font-family: 'Playfair Display', serif; font-size: 3.8rem; font-weight: 700; line-height: 1; color: white; }}
  .score-denom {{ font-size: 1.1rem; color: rgba(255,255,255,0.45); }}
  .score-bar-bg {{ background: rgba(255,255,255,0.15); border-radius: 4px; height: 7px; margin: 1rem 0; overflow: hidden; }}

  /* ── Meter boxes ── */
  .meter-row {{ display: flex; gap: 0.8rem; margin: 1rem 0; }}
  .meter-box {{ flex: 1; background: rgba(255,255,255,0.08); border-radius: 8px; padding: 0.8rem; border: 1px solid rgba(255,255,255,0.1); }}
  .meter-title {{ font-size: 0.62rem; letter-spacing: 1.5px; text-transform: uppercase; color: rgba(255,255,255,0.55); margin-bottom: 0.3rem; }}
  .meter-val {{ font-size: 1.5rem; font-weight: 700; color: white; }}

  /* ── SHAP reasons ── */
  .reason-card {{ background: white; border: 1px solid {RS_LGRAY}; border-left: 4px solid {RS_YELLOW}; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.5rem; font-size: 0.82rem; color: #1F2937; font-weight: 500; }}
  .reason-up   {{ border-left-color: {CRITICAL}; }}
  .reason-down {{ border-left-color: {FASTTRACK}; }}

  /* ── Cards ── */
  .card {{ background: white; border: 1px solid {RS_LGRAY}; border-radius: 10px; padding: 1.3rem 1.5rem; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }}
  .card h4 {{ color: {RS_NAVY}; font-size: 0.88rem; font-weight: 700; margin-bottom: 0.5rem; }}
  .card p {{ color: {RS_GRAY}; font-size: 0.8rem; line-height: 1.6; margin: 0; }}

  /* ── Form section header ── */
  .form-hdr {{
    background: {RS_BLUE};
    color: white;
    padding: 0.45rem 0.9rem;
    border-radius: 5px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 1.2rem 0 0.7rem 0;
  }}

  /* ── Action box ── */
  .action-box {{ border-radius: 10px; padding: 1rem 1.2rem; margin-top: 1.2rem; border: 1px solid; }}

  /* ── Summary table ── */
  .stbl {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 0.5rem; }}
  .stbl th {{ background: {RS_BLUE}; color: white; padding: 0.5rem 0.7rem; text-align: left; font-size: 0.68rem; letter-spacing: 0.5px; text-transform: uppercase; font-weight: 600; }}
  .stbl td {{ padding: 0.45rem 0.7rem; border-bottom: 1px solid {RS_LGRAY}; color: #374151; }}
  .stbl tr:nth-child(even) td {{ background: #F9FAFB; }}

  /* ── PDF drop zone ── */
  .pdf-zone {{ background: white; border: 2px dashed {RS_LGRAY}; border-radius: 12px; padding: 2rem; text-align: center; color: {RS_GRAY}; }}

  /* ── Arch flow ── */
  .arch-box {{ background: white; border: 1px solid {RS_LGRAY}; border-radius: 10px; padding: 1.3rem 1.8rem; }}

  /* ── Buttons ── */
  .stButton > button {{
    background: linear-gradient(135deg, {RS_BLUE}, {RS_NAVY}) !important;
    color: white !important;
    border: none !important;
    border-radius: 7px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 1.8rem !important;
    letter-spacing: 0.4px !important;
    box-shadow: 0 4px 12px rgba(0,48,135,0.28) !important;
    width: 100% !important;
  }}
  .stButton > button:hover {{ opacity: 0.92 !important; }}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {{
    gap: 0.5rem;
    border-bottom: 2px solid {RS_LGRAY};
    background: transparent !important;
  }}
  .stTabs [data-baseweb="tab"] {{
    background: white !important;
    color: {RS_GRAY} !important;
    border: 1px solid {RS_LGRAY} !important;
    border-radius: 6px 6px 0 0 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 1.2rem !important;
  }}
  .stTabs [aria-selected="true"] {{
    background: {RS_BLUE} !important;
    color: white !important;
    border-color: {RS_BLUE} !important;
  }}

  /* ── Selectbox, number input ── */
  div[data-baseweb="select"] > div {{
    border-radius: 6px !important;
    border-color: {RS_LGRAY} !important;
    background: white !important;
    font-size: 0.82rem !important;
    color: #111827 !important;
  }}
  .stNumberInput input, .stTextInput input {{
    border-radius: 6px !important;
    border-color: {RS_LGRAY} !important;
    background: white !important;
    color: #111827 !important;
    font-size: 0.82rem !important;
  }}
  label {{ color: #374151 !important; font-size: 0.78rem !important; font-weight: 500 !important; }}

  /* ── Footer ── */
  .footer {{ background: {RS_NAVY}; color: rgba(255,255,255,0.45); text-align: center; padding: 1rem; margin: 2rem -2rem -2rem -2rem; font-size: 0.7rem; letter-spacing: 0.5px; border-top: 2px solid {RS_YELLOW}; }}
  .footer span {{ color: {RS_YELLOW}; font-weight: 600; }}

  /* Uploaded file widget */
  [data-testid="stFileUploadDropzone"] {{ background: white !important; border-color: {RS_LGRAY} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Banner ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-banner">
  <div style="display:flex;align-items:center;">
    <div class="banner-logo">RS</div>
    <div>
      <div class="banner-title">Claim Decisioning Engine</div>
      <div class="banner-sub">Motor Third-Party · AI Fraud &amp; Litigation Triage</div>
    </div>
  </div>
  <div class="banner-right">
    <strong>Sundaram Pitch Fest 2026</strong>
    Team Apex Counsel · IIT Kharagpur<br>
    Arunadithyan S · Azhagappan G · G Abiimukeshwar
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
    except Exception as e:
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
    sev   = SEVERITY_MAP.get(claim.get('VehiclePrice','₹8–12 Lakh'), 1.0)
    score = round(min((0.6*fp + 0.4*lp)*sev*100, 100), 1)
    route, tkey = route_claim(fp, lp, score)
    tlabel, tcolor = priority_tier(score)
    expl  = shap.TreeExplainer(model_fraud)
    svals = expl.shap_values(df_f)
    sser  = pd.Series(svals[0], index=feat_fraud)
    top3  = [(f, float(sser[f])) for f in sser.abs().nlargest(3).index]
    return fp, lp, score, route, tkey, tlabel, tcolor, top3

# ── PDF extraction — pdfplumber + regex, no API needed ────────────────────────
def extract_text_from_pdf(pdf_bytes):
    """Extract all text from PDF using pdfplumber."""
    try:
        import pdfplumber
        text_pages = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: text_pages.append(t)
        return "\n".join(text_pages).lower(), None
    except Exception as e:
        return "", str(e)

def find_field(text, patterns, default=None):
    """Try each regex pattern in order, return first match group 1."""
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try: return m.group(1).strip()
            except: return m.group(0).strip()
    return default

def extract_claim_from_pdf(pdf_bytes):
    """
    Extract structured claim fields from a motor TP claim form PDF.
    Uses pdfplumber text extraction + keyword/regex matching.
    Works best on structured FNOL forms, survey reports, claim intimation letters.
    """
    text, err = extract_text_from_pdf(pdf_bytes)
    if err or not text:
        return {}, err or "No text found in PDF"

    extracted = {}

    # ── Vehicle Make ──────────────────────────────────────────────────────────
    make_keywords = {
        'Maruti Suzuki': ['maruti', 'suzuki', 'swift', 'baleno', 'alto', 'dzire', 'ertiga', 'brezza', 'celerio'],
        'Hyundai':       ['hyundai', 'i20', 'creta', 'verna', 'xcent', 'santro', 'tucson', 'venue', 'i10'],
        'Tata Motors':   ['tata', 'nexon', 'safari', 'harrier', 'tigor', 'tiago', 'altroz', 'punch'],
        'Mahindra':      ['mahindra', 'scorpio', 'bolero', 'xuv', 'thar', 'kuv', 'marazzo'],
        'Honda City':    ['honda', 'city', 'amaze', 'jazz', 'wr-v', 'cr-v'],
        'Toyota Innova': ['toyota', 'innova', 'fortuner', 'corolla', 'glanza', 'camry'],
        'Renault':       ['renault', 'kwid', 'duster', 'triber', 'kiger'],
        'Nissan India':  ['nissan', 'magnite', 'kicks', 'terrano'],
        'Skoda':         ['skoda', 'rapid', 'octavia', 'superb', 'kushaq', 'slavia'],
        'Volkswagen India': ['volkswagen', 'vw', 'polo', 'vento', 'taigun', 'virtus'],
        'BMW India':     ['bmw'],
        'Audi India':    ['audi'],
        'Mercedes India':['mercedes', 'benz'],
        'Force Motors':  ['force', 'tempo'],
    }
    for make, kws in make_keywords.items():
        if any(kw in text for kw in kws):
            extracted['VehicleMake'] = make
            break

    # ── Vehicle Category ──────────────────────────────────────────────────────
    if any(w in text for w in ['two wheeler','two-wheeler','motorcycle','bike','scooter','moped','tvs','bajaj','hero','royal enfield','activa']):
        extracted['VehicleCategory'] = 'Two Wheeler'
    elif any(w in text for w in ['commercial vehicle','goods vehicle','truck','lorry','bus','tempo traveller','auto rickshaw','cab','taxi','cvp','lmv-commercial']):
        extracted['VehicleCategory'] = 'Commercial Vehicle'
    elif any(w in text for w in ['private car','four wheeler','car','sedan','suv','hatchback','mpv','lmv']):
        extracted['VehicleCategory'] = 'Private Car'

    # ── Cover / Base Policy ───────────────────────────────────────────────────
    if any(w in text for w in ['comprehensive','package policy','od + tp','own damage and third party']):
        extracted['BasePolicy'] = 'Comprehensive'
    elif any(w in text for w in ['own damage','od policy','own damage only']):
        extracted['BasePolicy'] = 'Own Damage'
    elif any(w in text for w in ['third party','tp only','third-party','tp policy','act only']):
        extracted['BasePolicy'] = 'Third Party'

    # ── Vehicle Value / IDV ───────────────────────────────────────────────────
    # Look for IDV or vehicle value in lakhs
    idv_match = find_field(text, [
        r'idv[:\s₹rs.]*([0-9,]+)',
        r'insured declared value[:\s₹rs.]*([0-9,]+)',
        r'vehicle value[:\s₹rs.]*([0-9,]+)',
        r'sum insured[:\s₹rs.]*([0-9,]+)',
    ])
    if idv_match:
        try:
            idv_num = int(re.sub(r'[,\s]', '', idv_match))
            idv_lakh = idv_num / 100000
            if idv_lakh < 5:    extracted['VehiclePrice'] = 'Below ₹5 Lakh'
            elif idv_lakh < 8:  extracted['VehiclePrice'] = '₹5–8 Lakh'
            elif idv_lakh < 12: extracted['VehiclePrice'] = '₹8–12 Lakh'
            elif idv_lakh < 20: extracted['VehiclePrice'] = '₹12–20 Lakh'
            elif idv_lakh < 30: extracted['VehiclePrice'] = '₹20–30 Lakh'
            else:               extracted['VehiclePrice'] = 'Above ₹30 Lakh'
        except: pass

    # ── Vehicle Age ───────────────────────────────────────────────────────────
    yr_match = find_field(text, [
        r'year of manufacture[:\s]*(\d{4})',
        r'manufacturing year[:\s]*(\d{4})',
        r'model year[:\s]*(\d{4})',
        r'vehicle year[:\s]*(\d{4})',
        r'mfg[.:\s]*year[:\s]*(\d{4})',
    ])
    if yr_match:
        try:
            age_yrs = 2025 - int(yr_match)
            if age_yrs <= 0:    extracted['Vehicle_Age'] = 'new'
            elif age_yrs == 2:  extracted['Vehicle_Age'] = '2 years'
            elif age_yrs == 3:  extracted['Vehicle_Age'] = '3 years'
            elif age_yrs == 4:  extracted['Vehicle_Age'] = '4 years'
            elif age_yrs == 5:  extracted['Vehicle_Age'] = '5 years'
            elif age_yrs == 6:  extracted['Vehicle_Age'] = '6 years'
            elif age_yrs == 7:  extracted['Vehicle_Age'] = '7 years'
            else:               extracted['Vehicle_Age'] = 'more than 7'
        except: pass

    # ── Sex ───────────────────────────────────────────────────────────────────
    sex_match = find_field(text, [
        r'gender[:\s]*(male|female)',
        r'sex[:\s]*(male|female)',
        r'\b(mr|shri|sri)\b',   # male titles
        r'\b(mrs|ms|smt|kumari)\b',  # female titles
    ])
    if sex_match:
        s = sex_match.lower()
        if s in ['male','mr','shri','sri']:    extracted['Sex'] = 'Male'
        elif s in ['female','mrs','ms','smt','kumari']: extracted['Sex'] = 'Female'

    # ── Marital Status ────────────────────────────────────────────────────────
    marital_match = find_field(text, [r'marital status[:\s]*(single|married|divorced|widow)'])
    if marital_match:
        m = marital_match.lower()
        if 'single' in m:   extracted['MaritalStatus'] = 'Single'
        elif 'married' in m: extracted['MaritalStatus'] = 'Married'
        elif 'divorced' in m: extracted['MaritalStatus'] = 'Divorced'
        elif 'widow' in m:   extracted['MaritalStatus'] = 'Widow'

    # ── Age ───────────────────────────────────────────────────────────────────
    age_match = find_field(text, [
        r'age[:\s]*(\d{2})\s*(?:years|yrs)',
        r'age of (?:insured|policyholder|claimant)[:\s]*(\d{2})',
        r'date of birth.*?(\d{2})[\/\-](\d{2})[\/\-](\d{4})',
    ])
    if age_match:
        try:
            a = int(age_match)
            if 18 <= a <= 80:
                extracted['Age'] = a
        except: pass

    # ── Accident Area ─────────────────────────────────────────────────────────
    if any(w in text for w in ['rural area','village','gram','taluk','district highway','nh','sh','state highway','national highway']):
        extracted['AccidentArea'] = 'Rural'
    elif any(w in text for w in ['urban','city','municipal','town','metro','ward']):
        extracted['AccidentArea'] = 'Urban'

    # ── Fault ─────────────────────────────────────────────────────────────────
    if any(w in text for w in ['insured at fault','fault of insured','insured driver at fault','our insured caused','policyholder caused']):
        extracted['Fault'] = 'Insured Driver'
    elif any(w in text for w in ['third party at fault','fault of third party','tp at fault','third party caused','other vehicle']):
        extracted['Fault'] = 'Third Party'

    # ── FIR Filed ─────────────────────────────────────────────────────────────
    fir_match = find_field(text, [
        r'fir[:\s]*(yes|no|filed|not filed)',
        r'police report[:\s]*(yes|no|filed|not filed)',
        r'police complaint[:\s]*(yes|no)',
    ])
    if fir_match:
        f = fir_match.lower()
        extracted['FIR_Filed'] = 'Yes' if any(w in f for w in ['yes','filed']) and 'not' not in f else 'No'
    elif 'fir no' in text or 'fir number' in text or 'police station' in text:
        extracted['FIR_Filed'] = 'Yes'

    # ── Witness ───────────────────────────────────────────────────────────────
    wit_match = find_field(text, [
        r'witness[:\s]*(yes|no|available|not available|present|nil)',
        r'eyewitness[:\s]*(yes|no)',
    ])
    if wit_match:
        w = wit_match.lower()
        extracted['Witness_Available'] = 'Yes' if any(x in w for x in ['yes','available','present']) and 'not' not in w else 'No'

    # ── Prior Claims ──────────────────────────────────────────────────────────
    prior_match = find_field(text, [
        r'(?:previous|prior|past|number of)\s*claims?[:\s]*(\d+|nil|none|zero)',
        r'no[.\s]*of\s*claims?[:\s]*(\d+|nil|none|zero)',
    ])
    if prior_match:
        p = prior_match.lower().strip()
        if p in ['nil','none','zero','0']:     extracted['Prior_Claims_Count'] = '0'
        elif p == '1':                          extracted['Prior_Claims_Count'] = '1'
        elif p in ['2','3','4']:               extracted['Prior_Claims_Count'] = '2–4'
        else:
            try:
                n = int(p)
                extracted['Prior_Claims_Count'] = '2–4' if n <= 4 else '5+'
            except: pass

    # ── FNOL Delay ────────────────────────────────────────────────────────────
    # Look for accident date and intimation date, compute gap
    acc_date  = find_field(text, [r'(?:date of )?accident[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})'])
    intm_date = find_field(text, [r'(?:date of )?intimation[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
                                   r'(?:date of )?fnol[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})'])
    if acc_date and intm_date:
        try:
            from datetime import datetime
            fmts = ['%d/%m/%Y','%d-%m-%Y','%d/%m/%y','%d-%m-%y']
            d1 = d2 = None
            for fmt in fmts:
                try: d1 = datetime.strptime(acc_date, fmt); break
                except: pass
            for fmt in fmts:
                try: d2 = datetime.strptime(intm_date, fmt); break
                except: pass
            if d1 and d2:
                gap = (d2 - d1).days
                if gap <= 0:    extracted['FNOL_Delay_Days'] = 'none'
                elif gap <= 7:  extracted['FNOL_Delay_Days'] = '1 to 7'
                elif gap <= 15: extracted['FNOL_Delay_Days'] = '8 to 15'
                elif gap <= 30: extracted['FNOL_Delay_Days'] = '15 to 30'
                else:           extracted['FNOL_Delay_Days'] = 'more than 30'
        except: pass

    # ── Claim Filing Delay ────────────────────────────────────────────────────
    claim_date = find_field(text, [r'(?:date of )?claim (?:filed|submission|lodged)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})'])
    if acc_date and claim_date:
        try:
            from datetime import datetime
            fmts = ['%d/%m/%Y','%d-%m-%Y','%d/%m/%y','%d-%m-%y']
            d1 = d3 = None
            for fmt in fmts:
                try: d1 = datetime.strptime(acc_date, fmt); break
                except: pass
            for fmt in fmts:
                try: d3 = datetime.strptime(claim_date, fmt); break
                except: pass
            if d1 and d3:
                gap2 = (d3 - d1).days
                if gap2 <= 0:    extracted['Claim_Filing_Delay'] = 'none'
                elif gap2 <= 15: extracted['Claim_Filing_Delay'] = '8 to 15'
                elif gap2 <= 30: extracted['Claim_Filing_Delay'] = '15 to 30'
                else:            extracted['Claim_Filing_Delay'] = 'more than 30'
        except: pass

    # ── Intermediary Type ─────────────────────────────────────────────────────
    if any(w in text for w in ['broker','posp','pos agent','point of sale','insurance agent','dsp']):
        extracted['Intermediary_Type'] = 'Broker / POSP Agent'
    elif any(w in text for w in ['direct','branch','online','website','app','self']):
        extracted['Intermediary_Type'] = 'Direct / Branch'

    # ── Supplementary Reports ─────────────────────────────────────────────────
    supp_match = find_field(text, [
        r'(?:supplementary|addl|additional)\s*(?:survey|report)s?[:\s]*(\d+|nil|none)',
        r'no[.\s]*of\s*(?:supplementary|addl)\s*reports?[:\s]*(\d+)',
    ])
    if supp_match:
        try:
            s = supp_match.lower().strip()
            if s in ['nil','none','0']: extracted['Supplementary_Reports'] = '0'
            else:
                n = int(s)
                if n <= 2:  extracted['Supplementary_Reports'] = '1–2'
                elif n <= 5: extracted['Supplementary_Reports'] = '3–5'
                else:        extracted['Supplementary_Reports'] = '5+'
        except: pass

    # ── Deductible ────────────────────────────────────────────────────────────
    ded_match = find_field(text, [
        r'(?:compulsory\s+)?excess[:\s₹rs.]*([0-9,]+)',
        r'deductible[:\s₹rs.]*([0-9,]+)',
        r'policy\s+excess[:\s₹rs.]*([0-9,]+)',
    ])
    if ded_match:
        try:
            extracted['Deductible_INR'] = int(re.sub(r'[,\s]', '', ded_match))
        except: pass

    # ── Accident Month ────────────────────────────────────────────────────────
    month_match = find_field(text, [
        r'(?:date of )?accident[:\s]*\d{1,2}[\/\-](\d{1,2})[\/\-]\d{2,4}',
        r'(?:date of )?accident[:\s]*\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
    ])
    month_map = {'1':'Jan','2':'Feb','3':'Mar','4':'Apr','5':'May','6':'Jun',
                 '7':'Jul','8':'Aug','9':'Sep','10':'Oct','11':'Nov','12':'Dec',
                 'jan':'Jan','feb':'Feb','mar':'Mar','apr':'Apr','may':'May','jun':'Jun',
                 'jul':'Jul','aug':'Aug','sep':'Sep','oct':'Oct','nov':'Nov','dec':'Dec'}
    if month_match:
        extracted['Month'] = month_map.get(month_match.lower().strip(), 'Jan')

    # ── Address Change ────────────────────────────────────────────────────────
    if 'address change' in text or 'change of address' in text:
        addr_match = find_field(text, [r'address\s+change[:\s]*(\d+)\s*(month|year)'])
        if addr_match:
            try:
                n = int(addr_match)
                if n <= 6:  extracted['Address_Change_Before_Claim'] = 'under 6 months'
                elif n <= 12: extracted['Address_Change_Before_Claim'] = '1 year'
                elif n <= 36: extracted['Address_Change_Before_Claim'] = '2 to 3 years'
                else:         extracted['Address_Change_Before_Claim'] = '4 to 8 years'
            except: pass
    else:
        extracted['Address_Change_Before_Claim'] = 'no change'

    return extracted, None

def safe_idx(lst, val, default=0):
    try: return lst.index(val)
    except: return default

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

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["⚖️  Score Claim", "📊  Model Insights", "📄  PDF Auto-Fill"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCORE CLAIM
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not models_ok:
        st.error("Model files not found. Run the training notebook and place `.pkl` files in the same directory as `app.py`.")
        st.code("Required: model_fraud_india.pkl · model_litigation_india.pkl · encoder_india.pkl · features_fraud.pkl · features_litigation.pkl")
        st.stop()

    st.markdown('<div class="sec-eyebrow">Live Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Enter Claim Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-rule"></div>', unsafe_allow_html=True)

    with st.form("claim_form"):
        st.markdown('<div class="form-hdr">01 · Policy & Vehicle</div>', unsafe_allow_html=True)
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
        year = c4.number_input("Policy Year", 2018, 2025,
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

        st.markdown('<div class="form-hdr">03 · Accident & FNOL</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="form-hdr">04 · Distribution & Documentation</div>', unsafe_allow_html=True)
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-eyebrow">Transparency</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Model Architecture & Performance</div>', unsafe_allow_html=True)
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
    st.markdown(f"""
    <div class="arch-box">
      <div style="display:flex;align-items:center;gap:0.4rem;flex-wrap:wrap;justify-content:center;">
        {''.join(f'<div style="text-align:center;flex:1;min-width:90px;"><div style="background:{c};color:white;padding:0.65rem 0.4rem;border-radius:7px;font-size:0.72rem;font-weight:700;">{t}<br><span style="font-size:0.6rem;opacity:0.75;">{s}</span></div></div><div style="color:{RS_YELLOW};font-size:1.3rem;font-weight:700;">{"" if i==4 else "→"}</div>'
        for i,(t,s,c) in enumerate([
            ("FNOL","Claim Received",RS_BLUE),
            ("Data Spine","Feature Assembly",RS_NAVY),
            ("Model A + B","Fraud · Litigation","#4338CA"),
            ("SHAP Layer","Explainability","#0F766E"),
            ("Routing","Fast·SIU·ADR·Legal",FASTTRACK),
        ]))}
      </div>
      <div style="margin-top:1rem;font-size:0.72rem;color:{RS_GRAY};text-align:center;">
        Single-insurer scope · No consortium dependency · Human-in-the-loop on every high-risk flag
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
    st.caption("Build cost held constant across scenarios. Normalised to ₹1,000 Cr to avoid over-claiming Royal Sundaram figures.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PDF AUTO-FILL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-eyebrow">Smart Document Reader</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">PDF Auto-Fill</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-rule"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
      <h4>📄 How it works</h4>
      <p>Upload any motor TP claim document — survey report, FNOL form, claim intimation letter, or police report.
      The engine reads the PDF using Claude's vision API, extracts all available claim fields,
      and pre-fills the Score Claim form automatically. Switch to the <strong>Score Claim</strong> tab and click Run.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload claim PDF", type=['pdf'],
        help="FNOL form, survey report, claim intimation letter, FIR — any motor TP document"
    )

    if uploaded:
        pdf_bytes = uploaded.read()
        st.success(f"✅ Uploaded: **{uploaded.name}** ({len(pdf_bytes)//1024} KB)")

        with st.spinner("Reading document and extracting claim fields…"):
            extracted, err = extract_claim_from_pdf(pdf_bytes)

        if err:
            st.error(f"Extraction error: {err}")
        elif not extracted:
            st.warning("No fields could be extracted from this document.")
        else:
            # Map extracted fields into session state
            field_map = {
                'VehicleMake':          ('vehicle_make',    MAKES,                              None),
                'VehicleCategory':      ('vehicle_category',['Private Car','Two Wheeler','Commercial Vehicle'], None),
                'VehiclePrice':         ('vehicle_price',   PRICES,                             None),
                'Vehicle_Age':          ('vehicle_age',     AGES,                               None),
                'BasePolicy':           ('base_policy',     ['Third Party','Own Damage','Comprehensive'], None),
                'Sex':                  ('sex',             ['Male','Female'],                  None),
                'MaritalStatus':        ('marital_status',  ['Single','Married','Divorced','Widow'], None),
                'Prior_Claims_Count':   ('prior_claims',    CLAIMS,                             None),
                'AccidentArea':         ('accident_area',   ['Urban','Rural'],                  None),
                'Fault':                ('fault',           ['Insured Driver','Third Party'],   None),
                'FNOL_Delay_Days':      ('fnol_delay',      DELAYS,                             None),
                'Claim_Filing_Delay':   ('claim_delay',     DELAYS2,                            None),
                'FIR_Filed':            ('fir_filed',       ['Yes','No'],                       None),
                'Witness_Available':    ('witness',         ['Yes','No'],                       None),
                'Intermediary_Type':    ('intermediary',    ['Broker / POSP Agent','Direct / Branch'], None),
                'Address_Change_Before_Claim': ('address_change', ADDCHG,                      None),
                'Supplementary_Reports':('supp_reports',   SUPPS,                              None),
                'Month':                ('month',           MONTHS,                             None),
                'MonthClaimed':         ('month_claimed',   MONTHS,                             None),
            }
            filled, skipped = [], []
            for pdf_key, (ss_key, valid_list, _) in field_map.items():
                val = extracted.get(pdf_key)
                if val and val in valid_list:
                    st.session_state[ss_key] = val
                    filled.append(pdf_key)
                elif val:
                    skipped.append(f"{pdf_key}: '{val}'")

            # Numeric fields
            if extracted.get('Age') and 18 <= int(extracted['Age']) <= 80:
                st.session_state['age'] = int(extracted['Age'])
                filled.append('Age')
            if extracted.get('Deductible_INR') and extracted['Deductible_INR']:
                st.session_state['deductible_inr'] = int(extracted['Deductible_INR'])
                filled.append('Deductible_INR')
            if extracted.get('Driver_Risk_Rating') and 1 <= int(extracted['Driver_Risk_Rating']) <= 4:
                st.session_state['driver_rating'] = int(extracted['Driver_Risk_Rating'])
                filled.append('Driver_Risk_Rating')

            st.markdown("---")
            col_ok, col_sk = st.columns(2)
            with col_ok:
                st.markdown(f"**✅ {len(filled)} fields extracted**")
                for f in filled:
                    st.markdown(f"&nbsp;&nbsp;• {f}", unsafe_allow_html=True)
            with col_sk:
                if skipped:
                    st.markdown(f"**⚠️ {len(skipped)} fields could not be mapped**")
                    for s in skipped:
                        st.markdown(f"&nbsp;&nbsp;• {s}", unsafe_allow_html=True)
                else:
                    st.markdown("**✅ All detected fields mapped successfully**")

            st.markdown(f"""
            <div style="margin-top:1rem;background:#EFF6FF;border:1px solid #BFDBFE;
                        border-left:4px solid {RS_BLUE};border-radius:8px;padding:0.9rem 1.1rem;">
              <div style="font-size:0.82rem;color:{RS_NAVY};font-weight:600;">
                📋 Form pre-filled — switch to the <strong>Score Claim</strong> tab to review and run the engine.
              </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Raw JSON from document"):
                st.json(extracted)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  <span>Royal Sundaram General Insurance Co. Ltd.</span> &nbsp;·&nbsp;
  Claim Decisioning Engine &nbsp;·&nbsp;
  Team Apex Counsel · IIT Kharagpur &nbsp;·&nbsp;
  <span>Sundaram Pitch Fest 2026</span>
</div>
""", unsafe_allow_html=True)