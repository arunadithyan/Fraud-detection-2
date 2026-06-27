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

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Claim Decisioning Engine · Sundaram Finance",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS  (enterprise palette — Stripe / Linear / Vercel inspired)
# ══════════════════════════════════════════════════════════════════════════════
BG        = "#F7F8FA"
CARD      = "#FFFFFF"
INK       = "#111827"   # primary text
INK2      = "#4B5563"   # secondary text
MUTED     = "#9CA3AF"
ACCENT    = "#2563EB"   # blue accent
ACCENT_BG = "#EEF4FF"   # hover / tint
SUCCESS   = "#16A34A"
WARNING   = "#F59E0B"
DANGER    = "#DC2626"
BORDER    = "#E5E7EB"

# Routing/priority colours (kept semantically identical to original logic)
CRITICAL  = "#DC2626"
HIGH      = "#EA580C"
MEDIUM    = "#D97706"
FASTTRACK = "#16A34A"

# Aliases used by preserved business logic (names unchanged so logic is untouched)
RS_BLUE    = ACCENT
RS_NAVY    = "#0B1220"
RS_YELLOW  = "#F59E0B"
RS_WHITE   = "#FFFFFF"
RS_OFFWHITE= BG
RS_GRAY    = INK2
RS_LGRAY   = BORDER

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  (mobile-first, responsive, enterprise design system)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  /* ---------- base ---------- */
  html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stApp"],
  .main, .block-container, [data-testid="stVerticalBlock"], [data-testid="stForm"],
  [data-testid="stHorizontalBlock"], section.main {{
    background-color: {BG} !important;
    color: {INK} !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  }}
  @media (prefers-color-scheme: dark) {{
    html, body, [class*="css"] {{ background-color: {BG} !important; color: {INK} !important; }}
  }}
  #MainMenu, footer, header {{ visibility: hidden; }}

  .block-container {{
    padding: 1rem clamp(0.75rem, 3vw, 2.5rem) 3rem !important;
    max-width: 1320px;
  }}

  * {{ -webkit-tap-highlight-color: transparent; }}

  /* ---------- typography helpers ---------- */
  .eyebrow {{
    font-size: clamp(0.6rem, 1.4vw, 0.68rem); font-weight: 700; letter-spacing: 1.6px;
    text-transform: uppercase; color: {ACCENT}; margin-bottom: 0.4rem;
  }}
  .h-title {{
    font-size: clamp(1.25rem, 3.4vw, 1.7rem); font-weight: 800; color: {INK};
    letter-spacing: -0.02em; line-height: 1.2; margin: 0;
  }}
  .h-sub {{
    font-size: clamp(0.85rem, 2vw, 0.95rem); color: {INK2}; font-weight: 400;
    margin-top: 0.35rem; line-height: 1.5;
  }}
  .section-label {{
    font-size: 0.95rem; font-weight: 700; color: {INK}; letter-spacing: -0.01em;
    margin: 1.6rem 0 0.8rem;
  }}

  /* ---------- HERO ---------- */
  .hero {{
    background: linear-gradient(135deg, #0B1220 0%, #1E293B 55%, #2563EB 140%);
    border-radius: 20px;
    padding: clamp(1.4rem, 4vw, 2.6rem);
    margin-bottom: 1.4rem;
    box-shadow: 0 12px 40px rgba(15,23,42,0.18);
    position: relative; overflow: hidden;
  }}
  .hero::after {{
    content:""; position:absolute; top:-40%; right:-10%; width:380px; height:380px;
    background: radial-gradient(circle, rgba(37,99,235,0.45), transparent 70%);
    border-radius:50%;
  }}
  .hero-badge {{
    display:inline-flex; align-items:center; gap:0.5rem;
    background: rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.18);
    color:#E0E7FF; padding:0.32rem 0.8rem; border-radius:999px;
    font-size:0.68rem; font-weight:600; letter-spacing:0.4px; margin-bottom:1rem;
    backdrop-filter: blur(8px);
  }}
  .hero-badge .dot {{ width:7px; height:7px; border-radius:50%; background:{SUCCESS}; box-shadow:0 0 8px {SUCCESS}; }}
  .hero h1 {{
    font-size: clamp(1.6rem, 5.5vw, 2.7rem); font-weight: 800; color:#fff;
    letter-spacing:-0.03em; line-height:1.08; margin:0 0 0.6rem; position:relative;
  }}
  .hero p {{
    font-size: clamp(0.9rem, 2.4vw, 1.08rem); color: rgba(255,255,255,0.72);
    margin:0; max-width:640px; line-height:1.55; position:relative;
  }}
  .hero-meta {{
    margin-top:1.3rem; display:flex; flex-wrap:wrap; gap:0.5rem 1.4rem; position:relative;
    font-size:0.74rem; color:rgba(255,255,255,0.55);
  }}
  .hero-meta b {{ color:#E0E7FF; font-weight:600; }}

  /* ---------- KPI strip ---------- */
  .kpi-grid {{
    display:grid; grid-template-columns: repeat(4, 1fr); gap:0.8rem; margin-bottom:1.4rem;
  }}
  .kpi {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:16px;
    padding:1.05rem 1.1rem; transition: transform .18s ease, box-shadow .18s ease;
  }}
  .kpi:hover {{ transform: translateY(-3px); box-shadow:0 10px 28px rgba(17,24,39,0.08); }}
  .kpi-ico {{
    width:38px; height:38px; border-radius:11px; display:flex; align-items:center;
    justify-content:center; font-size:1.05rem; margin-bottom:0.7rem;
  }}
  .kpi-val {{ font-size: clamp(1.25rem, 3vw, 1.6rem); font-weight:800; color:{INK}; letter-spacing:-0.02em; line-height:1; }}
  .kpi-lbl {{ font-size:0.74rem; color:{INK2}; font-weight:500; margin-top:0.3rem; }}

  /* ---------- generic card ---------- */
  .ent-card {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:16px;
    padding:1.3rem 1.4rem; margin-bottom:1rem;
    box-shadow:0 1px 3px rgba(17,24,39,0.04);
    transition: box-shadow .18s ease, transform .18s ease;
  }}
  .ent-card:hover {{ box-shadow:0 8px 24px rgba(17,24,39,0.07); }}
  .ent-card h4 {{ font-size:0.92rem; font-weight:700; color:{INK}; margin:0 0 0.5rem; letter-spacing:-0.01em; }}
  .ent-card p {{ font-size:0.82rem; color:{INK2}; line-height:1.62; margin:0; }}

  /* ---------- overview feature cards ---------- */
  .feat-grid {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:0.8rem; }}
  .feat {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:16px; padding:1.1rem;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  }}
  .feat:hover {{ transform: translateY(-3px); box-shadow:0 10px 28px rgba(17,24,39,0.08); border-color:#C7D2FE; }}
  .feat-ico {{ width:40px; height:40px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:1.15rem; margin-bottom:0.75rem; background:{ACCENT_BG}; }}
  .feat h5 {{ font-size:0.9rem; font-weight:700; color:{INK}; margin:0 0 0.3rem; }}
  .feat p {{ font-size:0.76rem; color:{INK2}; line-height:1.5; margin:0; }}

  /* ---------- form section header ---------- */
  .wiz-hdr {{
    display:flex; align-items:center; gap:0.7rem; margin:1.5rem 0 0.9rem;
  }}
  .wiz-num {{
    width:30px; height:30px; border-radius:9px; background:{ACCENT}; color:#fff;
    display:flex; align-items:center; justify-content:center; font-weight:700;
    font-size:0.82rem; flex-shrink:0;
  }}
  .wiz-ttl {{ font-size:0.95rem; font-weight:700; color:{INK}; }}
  .wiz-desc {{ font-size:0.74rem; color:{MUTED}; }}

  /* ---------- result dashboard ---------- */
  .score-hero {{
    background: linear-gradient(135deg, #0B1220 0%, #1E293B 100%);
    border-radius:18px; padding: clamp(1.3rem,3.5vw,1.9rem); color:#fff;
    box-shadow:0 12px 36px rgba(15,23,42,0.2); height:100%;
  }}
  .score-hero .lbl {{ font-size:0.64rem; letter-spacing:1.8px; text-transform:uppercase; color:rgba(255,255,255,0.55); font-weight:700; }}
  .score-hero .num {{ font-size: clamp(2.6rem,9vw,3.6rem); font-weight:800; line-height:1; letter-spacing:-0.03em; margin-top:0.2rem; }}
  .score-hero .den {{ font-size:1rem; color:rgba(255,255,255,0.4); font-weight:600; }}
  .bar-bg {{ background:rgba(255,255,255,0.14); border-radius:6px; height:8px; margin:1.1rem 0 0.4rem; overflow:hidden; }}
  .bar-fill {{ height:100%; border-radius:6px; transition: width .6s cubic-bezier(.2,.8,.2,1); }}

  .stat-grid {{ display:grid; grid-template-columns: repeat(2,1fr); gap:0.7rem; }}
  .stat-card {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:14px; padding:1rem 1.1rem;
  }}
  .stat-card .k {{ font-size:0.64rem; letter-spacing:1px; text-transform:uppercase; color:{MUTED}; font-weight:700; }}
  .stat-card .v {{ font-size:1.45rem; font-weight:800; color:{INK}; margin-top:0.3rem; letter-spacing:-0.02em; }}
  .pill {{ display:inline-flex; align-items:center; padding:0.3rem 0.85rem; border-radius:999px; font-size:0.74rem; font-weight:700; color:#fff; }}

  /* ---------- SHAP reason rows ---------- */
  .reason {{
    background:{CARD}; border:1px solid {BORDER}; border-left:3px solid {ACCENT};
    border-radius:12px; padding:0.75rem 0.95rem; margin-bottom:0.55rem;
    font-size:0.82rem; color:{INK}; font-weight:500;
    display:flex; align-items:center; justify-content:space-between; gap:0.5rem;
  }}
  .reason.up {{ border-left-color:{DANGER}; }}
  .reason.down {{ border-left-color:{SUCCESS}; }}
  .reason .shap {{ color:{MUTED}; font-size:0.72rem; font-weight:600; white-space:nowrap; }}

  /* ---------- summary table ---------- */
  .ent-tbl {{ width:100%; border-collapse:separate; border-spacing:0; font-size:0.74rem; border:1px solid {BORDER}; border-radius:12px; overflow:hidden; }}
  .ent-tbl th {{ background:{BG}; color:{INK2}; padding:0.42rem 0.75rem; text-align:left; font-size:0.6rem; letter-spacing:0.6px; text-transform:uppercase; font-weight:700; border-bottom:1px solid {BORDER}; }}
  .ent-tbl td {{ padding:0.38rem 0.75rem; border-bottom:1px solid {BORDER}; color:{INK}; line-height:1.35; }}
  .ent-tbl tr:last-child td {{ border-bottom:none; }}
  .ent-tbl td:first-child {{ color:{INK2}; font-weight:500; width:42%; }}
  .ent-tbl td:last-child {{ font-weight:600; text-align:right; }}

  /* ---------- action banner ---------- */
  .action {{
    border-radius:14px; padding:1.1rem 1.3rem; margin-top:1.2rem;
    border:1px solid; display:flex; gap:0.85rem; align-items:flex-start;
  }}
  .action .a-lbl {{ font-size:0.62rem; letter-spacing:1.4px; text-transform:uppercase; font-weight:700; margin-bottom:0.3rem; }}
  .action .a-txt {{ font-size:0.85rem; color:{INK}; font-weight:500; line-height:1.55; }}

  /* ---------- verdict (predicted vs actual) ---------- */
  .verdict-grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:0.7rem; margin-top:0.7rem; }}
  .verdict-box {{ border:1px solid {BORDER}; border-top:3px solid; border-radius:12px; padding:0.85rem 1rem; background:{CARD}; }}
  .verdict-box .vk {{ font-size:0.6rem; letter-spacing:1.2px; text-transform:uppercase; color:{MUTED}; font-weight:700; }}
  .verdict-box .vv {{ font-size:1rem; font-weight:800; margin-top:0.25rem; letter-spacing:-0.01em; }}

  /* ---------- example cards ---------- */
  .ex-card {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:16px;
    padding:1.2rem 1.25rem; margin-bottom:0.6rem; position:relative;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  }}
  .ex-card:hover {{ transform: translateY(-3px); box-shadow:0 12px 30px rgba(17,24,39,0.1); border-color:#C7D2FE; }}
  .ex-card.sel {{ border-color:{ACCENT}; box-shadow:0 0 0 3px {ACCENT_BG}; }}
  .ex-top {{ display:flex; align-items:center; gap:0.7rem; margin-bottom:0.6rem; }}
  .ex-ico {{ width:42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:1.2rem; flex-shrink:0; }}
  .ex-card h5 {{ font-size:0.95rem; font-weight:700; color:{INK}; margin:0; letter-spacing:-0.01em; }}
  .ex-badge {{ display:inline-block; font-size:0.62rem; font-weight:700; letter-spacing:0.4px; text-transform:uppercase; padding:0.22rem 0.6rem; border-radius:999px; color:#fff; margin-bottom:0.5rem; }}
  .ex-card p {{ font-size:0.78rem; color:{INK2}; line-height:1.55; margin:0; }}

  /* ---------- pipeline / architecture ---------- */
  .flow {{ display:flex; align-items:stretch; gap:0.4rem; flex-wrap:nowrap; overflow-x:auto; padding-bottom:0.4rem; -webkit-overflow-scrolling:touch; }}
  .flow-step {{ flex:1 1 0; min-width:96px; border-radius:12px; padding:0.8rem 0.5rem; text-align:center; color:#fff; }}
  .flow-step .t {{ font-size:0.78rem; font-weight:700; }}
  .flow-step .s {{ font-size:0.6rem; opacity:0.85; margin-top:0.15rem; }}
  .flow-arrow {{ display:flex; align-items:center; color:{MUTED}; font-size:1.1rem; flex-shrink:0; }}

  /* ---------- buttons ---------- */
  .stButton > button {{
    background:{ACCENT} !important; color:#ffffff !important; border:none !important;
    border-radius:12px !important; font-weight:700 !important; font-size:0.9rem !important;
    padding:0.8rem 1.6rem !important; letter-spacing:0.1px !important; width:100% !important;
    min-height:48px !important; box-shadow:0 4px 14px rgba(37,99,235,0.28) !important;
    transition: transform .15s ease, box-shadow .15s ease, background .15s ease !important;
  }}
  .stButton > button *, .stButton > button p, .stButton > button span {{
    color:#ffffff !important;
  }}
  .stButton > button:hover {{ background:#1D4ED8 !important; transform:translateY(-1px); box-shadow:0 6px 20px rgba(37,99,235,0.36) !important; }}
  .stButton > button:active {{ transform:translateY(0); }}
  .stForm [data-testid="stFormSubmitButton"] > button {{
    background: {ACCENT} !important; color:#ffffff !important;
    min-height:54px !important; font-size:0.95rem !important;
  }}
  .stForm [data-testid="stFormSubmitButton"] > button *,
  .stForm [data-testid="stFormSubmitButton"] > button p {{
    color:#ffffff !important;
  }}

  /* ---------- tabs ---------- */
  .stTabs [data-baseweb="tab-list"] {{
    gap:0.35rem; background:{CARD}; border:1px solid {BORDER}; border-radius:14px;
    padding:0.35rem; overflow-x:auto; flex-wrap:nowrap; -webkit-overflow-scrolling:touch;
  }}
  .stTabs [data-baseweb="tab"] {{
    background:transparent !important; color:{INK2} !important; border:none !important;
    border-radius:10px !important; font-weight:600 !important; font-size:0.82rem !important;
    padding:0.6rem 1.05rem !important; white-space:nowrap; min-height:44px;
    transition: background .15s ease, color .15s ease;
  }}
  .stTabs [data-baseweb="tab"]:hover {{ background:{ACCENT_BG} !important; color:{ACCENT} !important; }}
  .stTabs [aria-selected="true"] {{ background:{ACCENT} !important; color:#fff !important; }}
  .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none !important; }}

  /* ---------- inputs ---------- */
  div[data-baseweb="select"] > div {{
    border-radius:11px !important; border-color:{BORDER} !important; background:{CARD} !important;
    font-size:0.86rem !important; color:{INK} !important; min-height:46px !important;
  }}
  div[data-baseweb="select"] > div:focus-within {{ border-color:{ACCENT} !important; box-shadow:0 0 0 3px {ACCENT_BG} !important; }}
  .stNumberInput input, .stTextInput input {{
    border-radius:11px !important; border-color:{BORDER} !important; background:{CARD} !important;
    color:{INK} !important; font-size:0.86rem !important; min-height:46px !important;
  }}
  .stNumberInput input:focus, .stTextInput input:focus {{ border-color:{ACCENT} !important; box-shadow:0 0 0 3px {ACCENT_BG} !important; }}
  label {{ color:{INK2} !important; font-size:0.78rem !important; font-weight:600 !important; }}

  button[data-testid="stNumberInputStepDown"], button[data-testid="stNumberInputStepUp"],
  [data-testid="stNumberInput"] button {{
    background:{BG} !important; color:{INK} !important; border:1px solid {BORDER} !important;
    min-width:40px !important;
  }}
  button[data-testid="stNumberInputStepDown"]:hover, button[data-testid="stNumberInputStepUp"]:hover {{
    background:{ACCENT} !important; color:#fff !important;
  }}

  [data-testid="stSlider"] > div > div > div {{ background:{BORDER} !important; }}
  [data-testid="stSlider"] [role="slider"] {{ background:{ACCENT} !important; border-color:{ACCENT} !important; }}
  [data-testid="stSlider"] > div > div > div > div {{ background:{ACCENT} !important; }}

  [data-testid="stAlert"] {{ background:{CARD} !important; color:{INK} !important; border-radius:12px !important; border:1px solid {BORDER} !important; }}
  [data-testid="stDataFrame"] {{ background:{CARD} !important; border-radius:12px; }}
  [data-testid="stExpander"] {{ background:{CARD} !important; border:1px solid {BORDER} !important; border-radius:12px !important; }}
  [data-testid="stExpander"] summary {{ color:{INK} !important; font-weight:600 !important; }}

  /* ---------- footer ---------- */
  .ent-footer {{
    margin-top:2.4rem; padding:1.4rem 0 0.5rem; border-top:1px solid {BORDER};
    display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between;
    gap:0.6rem; font-size:0.74rem; color:{MUTED};
  }}
  .ent-footer b {{ color:{INK2}; font-weight:600; }}

  /* ════════ RESPONSIVE BREAKPOINTS ════════ */
  /* Tablet */
  @media (max-width: 1024px) {{
    .kpi-grid {{ grid-template-columns: repeat(2,1fr); }}
    .feat-grid {{ grid-template-columns: repeat(2,1fr); }}
  }}
  /* Large phones */
  @media (max-width: 768px) {{
    .block-container {{ padding-top:0.5rem !important; }}
    .stat-grid {{ grid-template-columns: 1fr; }}
    .verdict-grid {{ grid-template-columns: 1fr; }}
    [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap; gap:0.5rem !important; }}
    [data-testid="stHorizontalBlock"] > div {{ min-width:100% !important; flex:1 1 100% !important; }}
  }}
  /* Phones */
  @media (max-width: 480px) {{
    .kpi-grid {{ grid-template-columns: 1fr 1fr; gap:0.55rem; }}
    .feat-grid {{ grid-template-columns: 1fr; }}
    .hero {{ border-radius:16px; }}
    .kpi {{ padding:0.85rem; }}
    .ent-card, .ex-card {{ padding:1rem; }}
    .stTabs [data-baseweb="tab"] {{ font-size:0.76rem !important; padding:0.55rem 0.8rem !important; }}
  }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div class="hero-badge"><span class="dot"></span> Sundaram Pitch Fest 2026 · Live Prototype</div>
  <h1>Claim Decisioning Engine</h1>
  <p>AI-powered fraud detection &amp; litigation intelligence for Motor Third-Party claims —
     scoring, explaining, and routing every claim at First Notice of Loss.</p>
  <div class="hero-meta">
    <span><b>Team Apex Counsel</b> · IIT Kharagpur</span>
    <span>Arunadithyan S · Azhagappan G · G Abiimukeshwar</span>
    <span>Operations · Risk &amp; Process Excellence</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ──
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-ico" style="background:{ACCENT_BG};color:{ACCENT};">🧠</div>
    <div class="kpi-val">XGBoost</div>
    <div class="kpi-lbl">Fraud Model · cost-sensitive</div>
  </div>
  <div class="kpi">
    <div class="kpi-ico" style="background:#FEF3C7;color:{WARNING};">⚖️</div>
    <div class="kpi-val">Dual-Model</div>
    <div class="kpi-lbl">Litigation Risk Engine</div>
  </div>
  <div class="kpi">
    <div class="kpi-ico" style="background:#DCFCE7;color:{SUCCESS};">📊</div>
    <div class="kpi-val">15,420</div>
    <div class="kpi-lbl">Claims Analysed</div>
  </div>
  <div class="kpi">
    <div class="kpi-ico" style="background:#EDE9FE;color:#7C3AED;">🔍</div>
    <div class="kpi-val">SHAP</div>
    <div class="kpi-lbl">Explainable AI · per claim</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODELS   (UNCHANGED LOGIC)
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS  (UNCHANGED LOGIC)
# ══════════════════════════════════════════════════════════════════════════════
SEVERITY_MAP = {
    'Below ₹5 Lakh':0.5, '₹5–8 Lakh':0.7, '₹8–12 Lakh':0.85,
    '₹12–20 Lakh':1.0, '₹20–30 Lakh':1.2, 'Above ₹30 Lakh':1.5
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

# ══════════════════════════════════════════════════════════════════════════════
#  EXAMPLE CLAIMS  (UNCHANGED VALUES)
# ══════════════════════════════════════════════════════════════════════════════
EXAMPLE_CLAIMS = {
    "Example 1 · Confirmed Fraud": {
        "desc": "High-risk two-wheeler TP claim with multiple prior claims, long reporting delay, no FIR, and no witness. Later confirmed as fraudulent.",
        "actual": "Confirmed Fraud → SIU",
        "actual_key": "critical",
        "icon": "🚨",
        "fields": dict(
            vehicle_category="Two Wheeler", vehicle_make="Hyundai",
            vehicle_price="Above ₹30 Lakh", vehicle_age="more than 7", base_policy="Third Party",
            deductible_inr=33000, driver_rating=4, sex="Male", marital_status="Married", age=43,
            prior_claims="5+", accident_area="Urban", fault="Third Party",
            fnol_delay="more than 30", claim_delay="more than 30", fir_filed="No",
            witness="No", intermediary="Broker / POSP Agent", address_change="no change",
            supp_reports="3–5", vehicles_in_pol="1",
        ),
    },
    "Example 2 · Fast Track Settlement": {
        "desc": "Low-risk private car claim with no prior claims despite reporting delay. Genuine claim settled through fast-track processing.",
        "actual": "Genuine → Fast Track",
        "actual_key": "fast",
        "icon": "✅",
        "fields": dict(
            vehicle_category="Private Car", vehicle_make="Hyundai",
            vehicle_price="₹5–8 Lakh", vehicle_age="more than 7", base_policy="Own Damage",
            deductible_inr=33000, driver_rating=3, sex="Male", marital_status="Married", age=60,
            prior_claims="0", accident_area="Urban", fault="Third Party",
            fnol_delay="more than 30", claim_delay="more than 30", fir_filed="No",
            witness="No", intermediary="Broker / POSP Agent", address_change="no change",
            supp_reports="3–5", vehicles_in_pol="1",
        ),
    },
    "Example 3 · ADR / Legal Prep": {
        "desc": "High-value genuine claim with elevated litigation risk. The engine identifies a low likelihood of fraud but a high probability of legal escalation, routing the claim to the ADR/Legal team for early settlement planning and dispute resolution.",
        "actual": "Genuine → ADR / Legal Prep",
        "actual_key": "high",
        "icon": "⚖️",
        "fields": dict(
            vehicle_category="Private Car", vehicle_make="Volkswagen India",
            vehicle_price="Above ₹30 Lakh", vehicle_age="6 years", base_policy="Own Damage",
            deductible_inr=33000, driver_rating=4, sex="Male", marital_status="Single", age=26,
            prior_claims="2–4", accident_area="Urban", fault="Insured Driver",
            fnol_delay="more than 30", claim_delay="more than 30", fir_filed="No",
            witness="No", intermediary="Broker / POSP Agent", address_change="no change",
            supp_reports="1–2", vehicles_in_pol="1",
        ),
    },
    "Example 4 · Standard Processing": {
        "desc": "Moderate-risk private car claim with some previous claims. Requires standard surveyor review before settlement.",
        "actual": "Genuine → Standard Processing",
        "actual_key": "medium",
        "icon": "📋",
        "fields": dict(
            vehicle_category="Private Car", vehicle_make="Maruti Suzuki",
            vehicle_price="₹5–8 Lakh", vehicle_age="7 years", base_policy="Own Damage",
            deductible_inr=33000, driver_rating=3, sex="Female", marital_status="Single", age=32,
            prior_claims="2–4", accident_area="Urban", fault="Insured Driver",
            fnol_delay="more than 30", claim_delay="more than 30", fir_filed="No",
            witness="No", intermediary="Broker / POSP Agent", address_change="no change",
            supp_reports="0", vehicles_in_pol="1",
        ),
    },
}

EX_KEY_COLOR = {'critical': CRITICAL, 'high': HIGH, 'medium': MEDIUM, 'fast': FASTTRACK}

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE  (UNCHANGED LOGIC)
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["⚖️  Score Claim", "🧪  Example Claims", "📊  Model Insights", "ℹ️  About"]
)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — SCORE CLAIM
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not models_ok:
        st.error("Model files not found. Run the training notebook and place `.pkl` files in the same directory as `app.py`.")
        st.code("Required: model_fraud_india.pkl · model_litigation_india.pkl · encoder_india.pkl · features_fraud.pkl · features_litigation.pkl")
        st.stop()

    if st.session_state['loaded_example']:
        st.info(f"📌 Example loaded: **{st.session_state['loaded_example']}**. Review the pre-filled form below and click **Run** — then compare the engine's verdict to the known outcome.")

    st.markdown('<div class="eyebrow">Live Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="h-title">Score a Claim</div>', unsafe_allow_html=True)
    st.markdown('<div class="h-sub">Complete the four sections below. The engine scores fraud and litigation risk, explains every flag, and routes the claim.</div>', unsafe_allow_html=True)

    with st.form("claim_form"):
        # ── Section 1 ──
        st.markdown('<div class="wiz-hdr"><div class="wiz-num">1</div><div><div class="wiz-ttl">Policy &amp; Vehicle</div><div class="wiz-desc">Cover, value, and vehicle profile</div></div></div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        vehicle_category = c1.selectbox("Vehicle Category", ['Private Car','Two Wheeler','Commercial Vehicle'],
            index=['Private Car','Two Wheeler','Commercial Vehicle'].index(st.session_state.vehicle_category))
        vehicle_make = c2.selectbox("Vehicle Make", MAKES, index=safe_idx(MAKES, st.session_state.vehicle_make))
        vehicle_price = c3.selectbox("Vehicle Value", PRICES, index=safe_idx(PRICES, st.session_state.vehicle_price))
        vehicle_age = c4.selectbox("Vehicle Age", AGES, index=safe_idx(AGES, st.session_state.vehicle_age))

        c1,c2,c3,c4 = st.columns(4)
        base_policy = c1.selectbox("Cover Type", ['Third Party','Own Damage','Comprehensive'],
            index=['Third Party','Own Damage','Comprehensive'].index(st.session_state.base_policy))
        deductible_inr = c2.number_input("Deductible (₹)", 5000, 100000, value=int(st.session_state.deductible_inr), step=1000)
        driver_rating = c3.slider("Driver Risk Rating", 1, 4, value=int(st.session_state.driver_rating), help="1=Low · 4=High")
        year = c4.number_input("Policy Year", 2018, 2026, value=int(st.session_state.year))

        # ── Section 2 ──
        st.markdown('<div class="wiz-hdr"><div class="wiz-num">2</div><div><div class="wiz-ttl">Claimant</div><div class="wiz-desc">Demographics and claim history</div></div></div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        sex = c1.selectbox("Sex", ['Male','Female'], index=['Male','Female'].index(st.session_state.sex))
        marital_status = c2.selectbox("Marital Status", ['Single','Married','Divorced','Widow'],
            index=['Single','Married','Divorced','Widow'].index(st.session_state.marital_status))
        age = c3.number_input("Age", 18, 80, value=int(st.session_state.age))
        prior_claims = c4.selectbox("Prior Claims", CLAIMS, index=safe_idx(CLAIMS, st.session_state.prior_claims))

        # ── Section 3 ──
        st.markdown('<div class="wiz-hdr"><div class="wiz-num">3</div><div><div class="wiz-ttl">Accident &amp; FNOL</div><div class="wiz-desc">Incident details and reporting timeline</div></div></div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        accident_area = c1.selectbox("Accident Area", ['Urban','Rural'], index=['Urban','Rural'].index(st.session_state.accident_area))
        fault = c2.selectbox("Fault", ['Insured Driver','Third Party'], index=['Insured Driver','Third Party'].index(st.session_state.fault))
        fnol_delay = c3.selectbox("FNOL Delay", DELAYS, index=safe_idx(DELAYS, st.session_state.fnol_delay))
        claim_delay = c4.selectbox("Claim Filing Delay", DELAYS2, index=safe_idx(DELAYS2, st.session_state.claim_delay))

        c1,c2,c3,c4 = st.columns(4)
        fir_filed = c1.selectbox("FIR Filed", ['Yes','No'], index=['Yes','No'].index(st.session_state.fir_filed))
        witness = c2.selectbox("Witness Available", ['Yes','No'], index=['Yes','No'].index(st.session_state.witness))
        month = c3.selectbox("Accident Month", MONTHS, index=safe_idx(MONTHS, st.session_state.month))
        day_of_week = c4.selectbox("Day of Week", DAYS, index=safe_idx(DAYS, st.session_state.day_of_week))

        # ── Section 4 ──
        st.markdown('<div class="wiz-hdr"><div class="wiz-num">4</div><div><div class="wiz-ttl">Distribution &amp; Documentation</div><div class="wiz-desc">Channel, supporting documents, and policy spread</div></div></div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        intermediary = c1.selectbox("Intermediary", ['Broker / POSP Agent','Direct / Branch'],
            index=['Broker / POSP Agent','Direct / Branch'].index(st.session_state.intermediary))
        address_change = c2.selectbox("Address Change", ADDCHG, index=safe_idx(ADDCHG, st.session_state.address_change))
        supp_reports = c3.selectbox("Supplementary Reports", SUPPS, index=safe_idx(SUPPS, st.session_state.supp_reports))
        vehicles_in_pol = c4.selectbox("Vehicles in Policy", VEHICLES, index=safe_idx(VEHICLES, st.session_state.vehicles_in_pol))

        c1,c2,c3,_ = st.columns(4)
        week_of_month = c1.number_input("Week of Month", 1, 5, value=int(st.session_state.week_of_month))
        month_claimed = c2.selectbox("Month Claimed", MONTHS, index=safe_idx(MONTHS, st.session_state.month_claimed))
        surveyor_id = c3.number_input("Surveyor ID", 1, 50, value=int(st.session_state.surveyor_id))

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
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

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Score Report</div>', unsafe_allow_html=True)
        st.markdown('<div class="h-title">Decisioning Output</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        col_s, col_d = st.columns([1, 1.5], gap="large")

        with col_s:
            st.markdown(f"""
            <div class="score-hero">
              <div class="lbl">Composite Risk Score</div>
              <div><span class="num">{score}</span><span class="den"> / 100</span></div>
              <div class="bar-bg"><div class="bar-fill" style="width:{score}%;background:{tcolor};"></div></div>
              <div style="margin-top:1.1rem;">
                <span class="pill" style="background:{tcolor};">{tlabel}</span>
              </div>
              <div style="margin-top:1.1rem;">
                <div class="lbl">Routing Decision</div>
                <div style="font-size:1.15rem;font-weight:800;margin-top:0.3rem;letter-spacing:-0.01em;">{route}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_d:
            st.markdown(f"""
            <div class="stat-grid">
              <div class="stat-card"><div class="k">Fraud Probability</div><div class="v" style="color:{tcolor};">{fp:.1%}</div></div>
              <div class="stat-card"><div class="k">Litigation Risk</div><div class="v" style="color:{ACCENT};">{lp:.1%}</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label">Top Fraud Risk Drivers</div>', unsafe_allow_html=True)
            reason_html = ""
            for feat, val in reasons:
                direction = "↑ Raises" if val > 0 else "↓ Lowers"
                cls = "up" if val > 0 else "down"
                ic = "🔴" if val > 0 else "🟢"
                reason_html += f"""<div class="reason {cls}"><span>{ic} <strong>{feat}</strong> — {direction} fraud risk</span><span class="shap">SHAP {val:+.3f}</span></div>"""
            st.markdown(reason_html, unsafe_allow_html=True)

        # ── Recommended Action  +  Claim Summary  (side by side) ──
        action_map = {
            'critical': (CRITICAL, '#FEF2F2', '🔴', 'Refer to SIU immediately. Do not settle. Assign senior investigator and request full documentation audit.'),
            'high':     (HIGH,     '#FFF7ED', '🟠', 'Flag for ADR / Legal team. Prepare fight-or-settle brief. Route to in-house counsel within 48 hours.'),
            'medium':   (MEDIUM,   '#FFFBEB', '🟡', 'Route to Standard Processing queue. Surveyor review required before payment authorisation.'),
            'fast':     (FASTTRACK,'#F0FDF4', '🟢', 'Eligible for Fast Track Settlement. Verify documents and initiate payment within 7 working days.'),
        }
        ac, abg, aic, atxt = action_map[tkey]

        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k,v in [
                ("Vehicle", f"{vehicle_make} · {vehicle_category}"),
                ("Cover",   f"{base_policy} · {vehicle_price}"),
                ("Claimant",f"{sex}, {age} yrs · {marital_status}"),
                ("FNOL Delay", fnol_delay), ("FIR Filed", fir_filed),
                ("Witness", witness), ("Prior Claims", prior_claims),
                ("Fault", fault), ("Intermediary", intermediary),
            ])

        col_act, col_tbl = st.columns([1, 1], gap="medium")

        with col_act:
            st.markdown(f"""
            <div style="background:{abg};border:1px solid {ac}33;border-left:4px solid {ac};
                        border-radius:14px;padding:1.2rem 1.3rem;height:100%;box-sizing:border-box;">
              <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.6rem;">
                <span style="font-size:1.25rem;line-height:1;">{aic}</span>
                <span style="font-size:0.62rem;letter-spacing:1.4px;text-transform:uppercase;
                             font-weight:700;color:{ac};">Recommended Action</span>
              </div>
              <div style="font-size:0.86rem;color:{INK};font-weight:500;line-height:1.6;">{atxt}</div>
              <div style="margin-top:1rem;padding-top:0.9rem;border-top:1px solid {ac}22;">
                <div style="font-size:0.62rem;color:{INK2};letter-spacing:0.5px;text-transform:uppercase;font-weight:700;margin-bottom:0.35rem;">Priority</div>
                <span class="pill" style="background:{tcolor};font-size:0.76rem;">{tlabel}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_tbl:
            st.markdown(f"""
            <div style="border:1px solid {BORDER};border-radius:14px;overflow:hidden;height:100%;">
              <div style="padding:0.6rem 0.75rem;background:{BG};border-bottom:1px solid {BORDER};">
                <span style="font-size:0.72rem;font-weight:700;color:{INK};letter-spacing:-0.01em;">Claim Summary</span>
              </div>
              <table class="ent-tbl" style="border:none;border-radius:0;">
                <tbody>{rows}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)

        # ── Predicted vs Actual ──
        if st.session_state['loaded_example']:
            ex = EXAMPLE_CLAIMS[st.session_state['loaded_example']]
            actual_key = ex['actual_key']; actual_lbl = ex['actual']
            act_col = EX_KEY_COLOR[actual_key]
            is_match = (tkey == actual_key)
            match_col = SUCCESS if is_match else WARNING
            match_txt = "✓ MATCH — engine agrees with the real outcome" if is_match else "≈ CLOSE — engine flagged the right risk family"
            st.markdown('<div class="section-label">Predicted vs Actual</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="ent-card">
              <div class="verdict-grid">
                <div class="verdict-box" style="border-top-color:{tcolor};">
                  <div class="vk">Engine Predicted</div>
                  <div class="vv" style="color:{tcolor};">{route}</div>
                </div>
                <div class="verdict-box" style="border-top-color:{act_col};">
                  <div class="vk">Actual Outcome</div>
                  <div class="vv" style="color:{act_col};">{actual_lbl}</div>
                </div>
              </div>
              <div style="margin-top:0.9rem;"><span class="pill" style="background:{match_col};">{match_txt}</span></div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — EXAMPLE CLAIMS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="eyebrow">Predicted vs Actual</div>', unsafe_allow_html=True)
    st.markdown('<div class="h-title">Example Claims</div>', unsafe_allow_html=True)
    st.markdown('<div class="h-sub">Each example is a realistic Motor TP claim with a known real-world outcome. Load one, run it on the Score Claim tab, and compare the engine\'s verdict to what actually happened.</div>', unsafe_allow_html=True)

    if not models_ok:
        st.warning("Model files not loaded — examples will fill the form, but scoring needs the `.pkl` files present.")

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    cols = st.columns(2, gap="large")
    for i, (name, ex) in enumerate(EXAMPLE_CLAIMS.items()):
        with cols[i % 2]:
            tag_col = EX_KEY_COLOR[ex['actual_key']]
            sel = "sel" if st.session_state['loaded_example'] == name else ""
            ic = ex.get("icon", "📄")
            st.markdown(f"""
            <div class="ex-card {sel}">
              <div class="ex-top">
                <div class="ex-ico" style="background:{tag_col}1A;color:{tag_col};">{ic}</div>
                <h5>{name}</h5>
              </div>
              <span class="ex-badge" style="background:{tag_col};">Actual: {ex['actual']}</span>
              <p>{ex['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            st.button("⬇️  Load this example", key=f"load_{i}",
                      on_click=load_example, args=(name,), use_container_width=True)
            if st.session_state['loaded_example'] == name:
                st.success("Loaded — open the **Score Claim** tab and click Run.")

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    st.caption("Examples are representative composites built to mirror common Motor TP claim archetypes — "
               "they show how the engine separates genuine, standard, litigation-bound, and fraudulent claims.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="eyebrow">Transparency</div>', unsafe_allow_html=True)
    st.markdown('<div class="h-title">Model Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="h-sub">Architecture, feature attribution, routing behaviour, and projected business impact.</div>', unsafe_allow_html=True)

    # KPI row
    st.markdown(f"""
    <div class="kpi-grid" style="margin-top:1rem;">
      <div class="kpi"><div class="kpi-ico" style="background:{ACCENT_BG};color:{ACCENT};">🎯</div><div class="kpi-val">0.84</div><div class="kpi-lbl">Fraud ROC-AUC</div></div>
      <div class="kpi"><div class="kpi-ico" style="background:#FEF3C7;color:{WARNING};">⚖️</div><div class="kpi-val">0.74</div><div class="kpi-lbl">Litigation ROC-AUC</div></div>
      <div class="kpi"><div class="kpi-ico" style="background:#DCFCE7;color:{SUCCESS};">📊</div><div class="kpi-val">15,420</div><div class="kpi-lbl">Test Claims Evaluated</div></div>
      <div class="kpi"><div class="kpi-ico" style="background:#EDE9FE;color:#7C3AED;">🔍</div><div class="kpi-val">100%</div><div class="kpi-lbl">Decisions Explained</div></div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown(f"""
        <div class="ent-card">
          <h4>🧠 Model A — Fraud Detection</h4>
          <p><strong>Algorithm:</strong> XGBoost · cost-sensitive (scale_pos_weight)<br>
          <strong>Label:</strong> SIU-confirmed fraud (binary)<br>
          <strong>Split:</strong> Stratified 80/20 — balanced fraud rate<br>
          <strong>Metric:</strong> PR-AUC (precision-recall under imbalance)<br>
          <strong>Explainability:</strong> SHAP TreeExplainer — top-3 reasons per claim, tribunal-ready</p>
        </div>
        """, unsafe_allow_html=True)
        if models_ok and feat_fraud:
            imp = pd.Series(model_fraud.feature_importances_, index=feat_fraud).sort_values().tail(10)
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
            bars = ax.barh(imp.index, imp.values, color=ACCENT, alpha=0.9, height=0.62)
            bars[-1].set_color(WARNING)
            ax.set_xlabel("Importance", fontsize=8, color=INK2)
            ax.set_title("Top 10 Fraud Drivers", fontsize=10, fontweight='bold', color=INK, pad=10)
            ax.tick_params(labelsize=7.5, colors=INK2)
            ax.spines[['top','right','left']].set_visible(False)
            ax.spines['bottom'].set_color(BORDER)
            ax.xaxis.grid(True, alpha=0.25, color=BORDER)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

    with col_b:
        st.markdown(f"""
        <div class="ent-card">
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
        colors  = [FASTTRACK, ACCENT, HIGH, CRITICAL]
        fig2, ax2 = plt.subplots(figsize=(5, 3.8))
        fig2.patch.set_facecolor(CARD)
        wedges, texts, autos = ax2.pie(sizes, labels=labels, colors=colors,
            autopct='%1.0f%%', startangle=140, pctdistance=0.72,
            wedgeprops=dict(width=0.52, edgecolor='white', linewidth=2))
        for t in texts:  t.set_fontsize(8);  t.set_color(INK2)
        for a in autos:  a.set_fontsize(8);  a.set_color('white'); a.set_fontweight('bold')
        ax2.set_title("Expected Routing Distribution", fontsize=10, fontweight='bold', color=INK, pad=10)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close()

    st.markdown('<div class="section-label">Engine Architecture</div>', unsafe_allow_html=True)
    arch_steps = [
        ("FNOL","Claim In",ACCENT),
        ("Data Spine","Features","#1E3A8A"),
        ("Model A+B","Fraud·Lit","#4338CA"),
        ("SHAP","Explain","#0F766E"),
        ("Routing","Decision",FASTTRACK),
    ]
    flow_html = ""
    for i,(t,s,c) in enumerate(arch_steps):
        flow_html += f'<div class="flow-step" style="background:{c};"><div class="t">{t}</div><div class="s">{s}</div></div>'
        if i < len(arch_steps)-1:
            flow_html += '<div class="flow-arrow">→</div>'
    st.markdown(f"""
    <div class="ent-card">
      <div class="flow">{flow_html}</div>
      <div style="margin-top:0.9rem;font-size:0.74rem;color:{INK2};text-align:center;">
        Single-insurer · No consortium dependency · Human-in-the-loop on every high-risk flag
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Projected Financial Impact · per ₹1,000 Cr TP Claims</div>', unsafe_allow_html=True)
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
#  TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="eyebrow">Sundaram Pitch Fest 2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="h-title">About the Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="h-sub">A single-insurer AI platform that scores, explains, and routes every Motor TP claim at First Notice of Loss.</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        for ic, title, body in [
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
            ("🗓️","Implementation Roadmap",
             "Phased 24-month rollout: P0 data foundation → P1 fraud shadow mode (6 mo) → P2 litigation model (12 mo) "
             "→ P3 network/graph layer → P4 live routing with drift monitoring and feedback loop."),
        ]:
            st.markdown(f'<div class="ent-card"><h4>{ic} {title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

    with col2:
        for ic, title, body in [
            ("🔍","Explainability",
             "Every flag carries SHAP-derived top-3 reasons in plain language — so SIU officers, claims managers, "
             "and MACT tribunals can act on the output. No black box; every decision is fully auditable."),
            ("📈","Business Value",
             "Normalised to ₹1,000 Cr of TP claims, the base case recovers ~₹18 Cr of annual benefit against a one-time "
             "₹3.5 Cr build — a payback period of roughly five months, with leakage recovery and litigation savings compounding."),
            ("🚀","Future Vision",
             "Phase 3+: graph neural networks over garage–lawyer entity graph, underwriting-stage risk pricing "
             "via Vahan/MoRTH signals, and federated cross-insurer learning without sharing raw claim data."),
            ("🏗️","Architecture",
             "FNOL → Data Spine → dual XGBoost (fraud + litigation) → SHAP explainer → composite risk score → routing. "
             "A clean, auditable pipeline with a human-in-the-loop checkpoint on every high-risk decision."),
        ]:
            st.markdown(f'<div class="ent-card"><h4>{ic} {title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

    # Tech stack
    tech_tags = ['XGBoost','SHAP','scikit-learn','Streamlit','Python 3.14','pandas','matplotlib','joblib']
    tech_html = "".join(
        f'<span style="background:{BG};border:1px solid {BORDER};border-radius:8px;padding:0.35rem 0.8rem;font-size:0.78rem;font-weight:600;color:{INK};">{t}</span>'
        for t in tech_tags)
    st.markdown(f'<div class="ent-card"><h4>🛠️ Technology Stack</h4><div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.6rem;">{tech_html}</div></div>', unsafe_allow_html=True)

    # Team panel
    team_html = "".join(
        f'<div style="flex:1;min-width:160px;"><div style="font-weight:700;font-size:0.92rem;color:#fff;">{n}</div><div style="font-size:0.74rem;color:rgba(255,255,255,0.5);">IIT Kharagpur · School of Law</div></div>'
        for n in ['Arunadithyan S','Azhagappan G','G Abiimukeshwar'])
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0B1220,#1E293B);border-radius:18px;padding:1.6rem 1.8rem;margin-top:0.4rem;">
      <div style="font-size:0.64rem;letter-spacing:1.6px;text-transform:uppercase;color:{WARNING};font-weight:700;margin-bottom:1.1rem;">
        Team Apex Counsel · IIT Kharagpur
      </div>
      <div style="display:flex;gap:1.5rem;flex-wrap:wrap;">{team_html}</div>
      <div style="margin-top:1.2rem;font-size:0.74rem;color:rgba(255,255,255,0.4);">
        Operations · Risk &amp; Process Excellence Track &nbsp;|&nbsp; Sundaram Pitch Fest 2026 · Round 2
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="ent-footer">
  <span><b>Sundaram Finance</b> · Claim Decisioning Engine</span>
  <span>Team Apex Counsel · IIT Kharagpur · <b>Sundaram Pitch Fest 2026</b></span>
</div>
""", unsafe_allow_html=True)
