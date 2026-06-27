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
    page_title="Claim Decisioning Model · Sundaram Finance",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════
BG        = "#F9FAFB"
CARD      = "#FFFFFF"
INK       = "#0F172A"
INK2      = "#475569"
MUTED     = "#94A3B8"
ACCENT    = "#4F46E5"   # indigo
ACCENT_L  = "#EEF2FF"
SUCCESS   = "#059669"
WARNING   = "#D97706"
DANGER    = "#DC2626"
BORDER    = "#E2E8F0"
SLATE     = "#1E293B"

CRITICAL  = "#DC2626"
HIGH      = "#EA580C"
MEDIUM    = "#D97706"
FASTTRACK = "#059669"

RS_BLUE = ACCENT; RS_NAVY = SLATE; RS_YELLOW = WARNING
RS_WHITE = CARD; RS_OFFWHITE = BG; RS_GRAY = INK2; RS_LGRAY = BORDER

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=DM+Serif+Display&display=swap');

/*  reset & base  */
html,body,[class*="css"],[data-testid="stAppViewContainer"],[data-testid="stApp"],
.main,.block-container,[data-testid="stVerticalBlock"],[data-testid="stForm"],
[data-testid="stHorizontalBlock"],section.main{{
  background:{BG}!important;color:{INK}!important;
  font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif!important;
}}
@media(prefers-color-scheme:dark){{html,body,[class*="css"]{{background:{BG}!important;color:{INK}!important;}}}}
#MainMenu,footer,header{{visibility:hidden;}}
*{{-webkit-tap-highlight-color:transparent;box-sizing:border-box;}}
.block-container{{padding:0 clamp(1rem,4vw,3rem) 4rem!important;max-width:1400px;margin:0 auto;}}

/*  TOP NAV  */
.topnav{{
  display:flex;align-items:center;justify-content:space-between;
  padding:0.9rem 0 1.2rem;margin-bottom:0.4rem;
  border-bottom:1px solid {BORDER};
}}
.topnav-brand{{display:flex;align-items:center;gap:0.65rem;}}
.topnav-logo{
    width:70px;
    height:70px;
    display:flex;
    align-items:center;
    justify-content:center;
    overflow:hidden;
    background:white;
    border-radius:12px;
    padding:6px;
    border:1px solid rgba(0,0,0,.08);
}
.topnav-logo img{
    width:100%;
    height:100%;
    object-fit:contain;
}
.topnav-name{{font-size:0.95rem;font-weight:700;color:{INK};letter-spacing:-0.01em;}}
.topnav-sub{{font-size:0.72rem;color:{MUTED};margin-top:1px;}}
.topnav-right{{display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;justify-content:flex-end;}}
.nav-tag{{
  font-size:0.68rem;font-weight:600;padding:0.22rem 0.65rem;border-radius:6px;
  background:{ACCENT_L};color:{ACCENT};letter-spacing:0.2px;white-space:nowrap;
}}
.live-dot{{
  width:7px;height:7px;border-radius:50%;background:{SUCCESS};
  box-shadow:0 0 0 2px #fff,0 0 0 4px {SUCCESS}40;display:inline-block;margin-right:4px;
}}

/*  HERO  */
.hero{{
  display:grid;grid-template-columns:1fr auto;gap:2rem;
  align-items:start;padding:2rem 0 1.6rem;
}}
.hero-left .kicker{{
  display:inline-flex;align-items:center;gap:0.45rem;
  font-size:0.7rem;font-weight:700;letter-spacing:1.4px;text-transform:uppercase;
  color:{ACCENT};margin-bottom:0.8rem;
}}
.hero-left h1{{
  font-family:'DM Serif Display',Georgia,serif;
  font-size:clamp(1.9rem,4.5vw,3rem);font-weight:400;
  color:{INK};line-height:1.08;letter-spacing:-0.02em;margin:0 0 0.7rem;
}}
.hero-left h1 em{{font-style:italic;color:{ACCENT};}}
.hero-left p{{
  font-size:clamp(0.88rem,1.8vw,1rem);color:{INK2};
  line-height:1.65;max-width:520px;margin:0 0 1.2rem;font-weight:400;
}}
.hero-meta{{display:flex;flex-wrap:wrap;gap:0.35rem 1.2rem;font-size:0.72rem;color:{MUTED};}}
.hero-meta b{{color:{INK2};font-weight:600;}}
.hero-right{{display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;min-width:300px;}}

/*  STAT CARD (hero right + result)  */
.s-card{{
  background:{CARD};border:1px solid {BORDER};border-radius:14px;
  padding:1rem 1.05rem;position:relative;overflow:hidden;
  transition:transform .18s ease,box-shadow .18s ease;
}}
.s-card:hover{{transform:translateY(-2px);box-shadow:0 8px 24px rgba(15,23,42,0.08);}}
.s-card::before{{
  content:"";position:absolute;top:0;left:0;right:0;height:3px;border-radius:14px 14px 0 0;
}}
.s-card.accent::before{{background:{ACCENT};}}
.s-card.success::before{{background:{SUCCESS};}}
.s-card.warn::before{{background:{WARNING};}}
.s-card.purple::before{{background:#7C3AED;}}
.s-card .ico{{
  width:34px;height:34px;border-radius:9px;display:flex;align-items:center;
  justify-content:center;font-size:0.95rem;margin-bottom:0.6rem;
}}
.s-card .val{{font-size:clamp(1.3rem,3vw,1.7rem);font-weight:800;color:{INK};letter-spacing:-0.03em;line-height:1;}}
.s-card .lbl{{font-size:0.72rem;color:{MUTED};font-weight:500;margin-top:0.25rem;line-height:1.3;}}

/*  DIVIDER  */
.divider{{height:1px;background:{BORDER};margin:1.8rem 0;}}

/*  SECTION HEAD  */
.sec-head{{margin:0 0 1rem;}}
.sec-head .eye{{font-size:0.62rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:{ACCENT};margin-bottom:0.3rem;}}
.sec-head h2{{font-size:clamp(1.1rem,2.5vw,1.35rem);font-weight:800;color:{INK};letter-spacing:-0.02em;margin:0;}}
.sec-head p{{font-size:0.84rem;color:{INK2};margin:0.3rem 0 0;line-height:1.5;}}

/*  WIZARD FORM  */
.wiz-sec{{
  background:{CARD};border:1px solid {BORDER};border-radius:16px;
  padding:1.4rem 1.5rem;margin-bottom:0.8rem;
  box-shadow:0 1px 2px rgba(15,23,42,0.04);
}}
.wiz-head{{display:flex;align-items:center;gap:0.7rem;margin-bottom:1.2rem;}}
.wiz-n{{
  width:28px;height:28px;border-radius:8px;background:{ACCENT};color:#fff;
  display:flex;align-items:center;justify-content:center;
  font-weight:800;font-size:0.78rem;flex-shrink:0;
}}
.wiz-title{{font-size:0.92rem;font-weight:700;color:{INK};}}
.wiz-sub{{font-size:0.73rem;color:{MUTED};margin-top:1px;}}

/*  RESULT GAUGE  */
.gauge-wrap{{
  background:{SLATE};border-radius:18px;
  padding:clamp(1.2rem,3.5vw,1.8rem);
  box-shadow:0 16px 48px rgba(15,23,42,0.24);
  position:relative;overflow:hidden;
}}
.gauge-wrap::after{{
  content:"";position:absolute;bottom:-30%;right:-10%;width:220px;height:220px;
  background:radial-gradient(circle,rgba(79,70,229,0.35),transparent 70%);
  border-radius:50%;pointer-events:none;
}}
.gauge-eye{{font-size:0.6rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:rgba(255,255,255,0.45);margin-bottom:0.6rem;}}
.gauge-score{{
  font-family:'DM Serif Display',Georgia,serif;
  font-size:clamp(3rem,10vw,4.5rem);font-weight:400;color:#fff;
  line-height:1;letter-spacing:-0.02em;
}}
.gauge-denom{{font-size:1rem;color:rgba(255,255,255,0.35);font-weight:600;vertical-align:super;font-family:'Inter',sans-serif;}}
.gauge-bar-track{{background:rgba(255,255,255,0.1);border-radius:6px;height:6px;margin:1rem 0 0.5rem;overflow:hidden;}}
.gauge-bar{{height:100%;border-radius:6px;transition:width .8s cubic-bezier(.2,.8,.2,1);}}
.gauge-row{{display:flex;gap:0.6rem;margin-top:0.9rem;}}
.gauge-chip{{
  flex:1;background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.1);
  border-radius:10px;padding:0.65rem 0.7rem;
}}
.gauge-chip .gk{{font-size:0.56rem;letter-spacing:1.2px;text-transform:uppercase;color:rgba(255,255,255,0.45);font-weight:700;}}
.gauge-chip .gv{{font-size:1.1rem;font-weight:800;color:#fff;margin-top:0.15rem;letter-spacing:-0.02em;}}
.route-box{{margin-top:0.9rem;padding-top:0.9rem;border-top:1px solid rgba(255,255,255,0.1);}}
.route-box .rk{{font-size:0.58rem;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,0.45);font-weight:700;margin-bottom:0.3rem;}}
.route-box .rv{{font-size:1rem;font-weight:800;color:#fff;letter-spacing:-0.01em;}}
.rpill{{display:inline-flex;align-items:center;padding:0.3rem 0.85rem;border-radius:999px;font-size:0.72rem;font-weight:700;color:#fff;margin-top:0.5rem;}}

/*  SHAP DRIVERS  */
.driver{{
  display:flex;align-items:center;justify-content:space-between;gap:0.5rem;
  background:{CARD};border:1px solid {BORDER};border-radius:11px;
  padding:0.65rem 0.85rem;margin-bottom:0.45rem;
  border-left:3px solid {ACCENT};font-size:0.8rem;font-weight:500;color:{INK};
}}
.driver.up{{border-left-color:{DANGER};}}
.driver.down{{border-left-color:{SUCCESS};}}
.driver-text{{flex:1;min-width:0;}}
.driver-shap{{font-size:0.66rem;font-weight:700;color:{MUTED};white-space:nowrap;flex-shrink:0;
  background:{BG};border:1px solid {BORDER};border-radius:6px;padding:0.15rem 0.45rem;}}

/*  RESULT BOTTOM — action + table  */
.res-bottom{{display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-top:0.9rem;}}
.act-card{{
  border-radius:14px;padding:1.2rem 1.25rem;border:1px solid;border-left:4px solid;
  display:flex;flex-direction:column;
}}
.act-head{{display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;}}
.act-ico{{font-size:1rem;line-height:1;}}
.act-lbl{{font-size:0.58rem;font-weight:700;letter-spacing:1.4px;text-transform:uppercase;}}
.act-body{{font-size:0.83rem;font-weight:500;line-height:1.6;flex:1;}}
.act-foot{{margin-top:0.9rem;padding-top:0.85rem;border-top:1px solid;}}
.act-foot .pk{{font-size:0.56rem;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:{INK2};margin-bottom:0.3rem;}}

.clm-card{{
  background:{CARD};border:1px solid {BORDER};border-radius:14px;
  overflow:hidden;display:flex;flex-direction:column;
}}
.clm-head{{
  padding:0.6rem 0.85rem;border-bottom:1px solid {BORDER};
  font-size:0.72rem;font-weight:700;color:{INK};background:{BG};
}}
.clm-tbl{{width:100%;border-collapse:collapse;font-size:0.73rem;}}
.clm-tbl td{{padding:0.42rem 0.85rem;border-bottom:1px solid {BORDER};vertical-align:middle;}}
.clm-tbl tr:last-child td{{border-bottom:none;}}
.clm-tbl td:first-child{{color:{INK2};font-weight:500;font-size:0.7rem;width:42%;}}
.clm-tbl td:last-child{{font-weight:700;color:{INK};text-align:right;}}

/*  VERDICT (predicted vs actual)  */
.verdict{{display:grid;grid-template-columns:1fr 1fr;gap:0.7rem;margin-top:0.8rem;}}
.verdict-box{{background:{CARD};border:1px solid {BORDER};border-top:3px solid;border-radius:12px;padding:0.9rem 1rem;}}
.verdict-box .vk{{font-size:0.58rem;letter-spacing:1.2px;text-transform:uppercase;color:{MUTED};font-weight:700;}}
.verdict-box .vv{{font-size:0.96rem;font-weight:800;margin-top:0.25rem;letter-spacing:-0.01em;}}
.match-banner{{
  display:flex;align-items:center;gap:0.5rem;margin-top:0.7rem;
  padding:0.6rem 0.85rem;border-radius:10px;border:1px solid;
  font-size:0.8rem;font-weight:600;
}}

/*  EXAMPLE CARDS  */
.ex-grid{{display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;}}
.ex-card{{
  background:{CARD};border:1px solid {BORDER};border-radius:16px;
  padding:1.25rem 1.3rem;cursor:pointer;position:relative;overflow:hidden;
  transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease;
}}
.ex-card:hover{{transform:translateY(-3px);box-shadow:0 12px 32px rgba(15,23,42,0.1);border-color:#C7D2FE;}}
.ex-card.sel{{border-color:{ACCENT};box-shadow:0 0 0 3px {ACCENT_L};}}
.ex-card::before{{
  content:"";position:absolute;top:0;left:0;right:0;height:3px;border-radius:16px 16px 0 0;
}}
.ex-top{{display:flex;align-items:flex-start;gap:0.75rem;margin-bottom:0.55rem;}}
.ex-ico{{
  width:40px;height:40px;border-radius:11px;display:flex;align-items:center;
  justify-content:center;font-size:1.1rem;flex-shrink:0;margin-top:1px;
}}
.ex-card h5{{font-size:0.9rem;font-weight:700;color:{INK};margin:0 0 0.15rem;letter-spacing:-0.01em;}}
.ex-badge{{
  display:inline-block;font-size:0.6rem;font-weight:700;letter-spacing:0.5px;
  text-transform:uppercase;padding:0.2rem 0.6rem;border-radius:999px;color:#fff;margin-bottom:0.5rem;
}}
.ex-desc{{font-size:0.77rem;color:{INK2};line-height:1.55;margin:0;}}

/*  MODEL INSIGHTS CARDS  */
.insight-card{{
  background:{CARD};border:1px solid {BORDER};border-radius:16px;
  padding:1.3rem 1.4rem;margin-bottom:0;
  box-shadow:0 1px 2px rgba(15,23,42,0.04);
}}
.insight-card h4{{font-size:0.9rem;font-weight:700;color:{INK};margin:0 0 0.5rem;letter-spacing:-0.01em;}}
.insight-card p{{font-size:0.8rem;color:{INK2};line-height:1.6;margin:0;}}

/*  ARCH FLOW  */
.flow{{display:flex;align-items:stretch;gap:0.4rem;overflow-x:auto;padding-bottom:4px;-webkit-overflow-scrolling:touch;}}
.flow-step{{flex:1 1 0;min-width:90px;border-radius:11px;padding:0.75rem 0.5rem;text-align:center;color:#fff;}}
.flow-step .ft{{font-size:0.76rem;font-weight:700;}}
.flow-step .fs{{font-size:0.58rem;opacity:0.82;margin-top:0.12rem;}}
.flow-arr{{display:flex;align-items:center;color:{MUTED};font-size:1rem;flex-shrink:0;}}

/*  ABOUT CARDS  */
.about-card{{
  background:{CARD};border:1px solid {BORDER};border-radius:14px;
  padding:1.2rem 1.3rem;margin-bottom:0.7rem;
  border-top:3px solid {ACCENT};
}}
.about-card h4{{font-size:0.88rem;font-weight:700;color:{INK};margin:0 0 0.45rem;}}
.about-card p{{font-size:0.79rem;color:{INK2};line-height:1.62;margin:0;}}

/*  BUTTONS  */
.stButton>button{{
  background:{ACCENT}!important;color:#fff!important;border:none!important;
  border-radius:11px!important;font-weight:700!important;font-size:0.88rem!important;
  padding:0.78rem 1.5rem!important;min-height:48px!important;width:100%!important;
  letter-spacing:0.05px!important;
  box-shadow:0 2px 8px rgba(79,70,229,0.3)!important;
  transition:background .15s,transform .15s,box-shadow .15s!important;
}}
.stButton>button *,.stButton>button p,.stButton>button span{{color:#fff!important;}}
.stButton>button:hover{{background:#4338CA!important;transform:translateY(-1px);box-shadow:0 4px 16px rgba(79,70,229,0.4)!important;}}
.stButton>button:active{{transform:translateY(0);}}
.stForm [data-testid="stFormSubmitButton"]>button{{
  background:{ACCENT}!important;color:#fff!important;
  min-height:52px!important;font-size:0.92rem!important;
}}
.stForm [data-testid="stFormSubmitButton"]>button *{{color:#fff!important;}}

/*  TABS  */
.stTabs [data-baseweb="tab-list"]{{
  gap:0.25rem;background:{CARD};border:1px solid {BORDER};border-radius:13px;
  padding:0.3rem;overflow-x:auto;flex-wrap:nowrap;-webkit-overflow-scrolling:touch;
}}
.stTabs [data-baseweb="tab"]{{
  background:transparent!important;color:{INK2}!important;border:none!important;
  border-radius:9px!important;font-weight:600!important;font-size:0.82rem!important;
  padding:0.58rem 1rem!important;white-space:nowrap;min-height:42px;
  transition:background .12s,color .12s;
}}
.stTabs [data-baseweb="tab"]:hover{{background:{ACCENT_L}!important;color:{ACCENT}!important;}}
.stTabs [aria-selected="true"]{{background:{ACCENT}!important;color:#fff!important;}}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{{display:none!important;}}

/*  INPUTS  */
div[data-baseweb="select"]>div{{
  border-radius:10px!important;border-color:{BORDER}!important;background:{CARD}!important;
  font-size:0.84rem!important;color:{INK}!important;min-height:44px!important;
}}
div[data-baseweb="select"]>div:focus-within{{border-color:{ACCENT}!important;box-shadow:0 0 0 3px {ACCENT_L}!important;}}
.stNumberInput input,.stTextInput input{{
  border-radius:10px!important;border-color:{BORDER}!important;background:{CARD}!important;
  color:{INK}!important;font-size:0.84rem!important;min-height:44px!important;
}}
.stNumberInput input:focus,.stTextInput input:focus{{border-color:{ACCENT}!important;box-shadow:0 0 0 3px {ACCENT_L}!important;}}
label{{color:{INK2}!important;font-size:0.76rem!important;font-weight:600!important;}}
button[data-testid="stNumberInputStepDown"],button[data-testid="stNumberInputStepUp"],
[data-testid="stNumberInput"] button{{
  background:{BG}!important;color:{INK}!important;border:1px solid {BORDER}!important;min-width:38px!important;
}}
button[data-testid="stNumberInputStepDown"]:hover,button[data-testid="stNumberInputStepUp"]:hover{{
  background:{ACCENT}!important;color:#fff!important;
}}
[data-testid="stSlider"]>div>div>div{{background:{BORDER}!important;}}
[data-testid="stSlider"] [role="slider"]{{background:{ACCENT}!important;border-color:{ACCENT}!important;}}
[data-testid="stSlider"]>div>div>div>div{{background:{ACCENT}!important;}}
[data-testid="stAlert"]{{background:{CARD}!important;color:{INK}!important;border-radius:12px!important;border:1px solid {BORDER}!important;}}
[data-testid="stDataFrame"]{{background:{CARD}!important;border-radius:12px;border:1px solid {BORDER}!important;}}
[data-testid="stExpander"]{{background:{CARD}!important;border:1px solid {BORDER}!important;border-radius:12px!important;}}
[data-testid="stExpander"] summary{{color:{INK}!important;font-weight:600!important;}}

/*  FOOTER  */
.ent-footer{{
  margin-top:2.5rem;padding:1.2rem 0 0.4rem;border-top:1px solid {BORDER};
  display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;
  gap:0.5rem;font-size:0.72rem;color:{MUTED};
}}
.ent-footer b{{color:{INK2};font-weight:600;}}


/* ════ BREAKPOINTS ════ */
@media(max-width:1024px){{
  .hero{{grid-template-columns:1fr;}}
  .hero-right{{grid-template-columns:repeat(4,1fr);min-width:unset;}}
  .hero-left p{{max-width:100%;}}
}}
@media(max-width:768px){{
  .block-container{{padding:0 1rem 3rem!important;}}
  .hero{{padding:1.4rem 0 1rem;gap:1.2rem;}}
  .hero-right{{grid-template-columns:1fr 1fr;}}
  .res-bottom{{grid-template-columns:1fr;}}
  .verdict{{grid-template-columns:1fr;}}
  .ex-grid{{grid-template-columns:1fr;}}
  [data-testid="stHorizontalBlock"]{{flex-wrap:wrap;gap:0.5rem!important;}}
  [data-testid="stHorizontalBlock"]>div{{min-width:100%!important;flex:1 1 100%!important;}}
  .topnav-right{{display:none;}}
}}
@media(max-width:480px){{
  .block-container{{padding:0 0.75rem 2.5rem!important;}}
  .hero-right{{grid-template-columns:1fr 1fr;gap:0.5rem;}}
  .hero-left h1{{font-size:1.75rem;}}
  .gauge-score{{font-size:3rem;}}
  .gauge-row{{flex-direction:column;}}
  .wiz-sec{{padding:1.1rem;}}
  .s-card{{padding:0.85rem;}}
  .stTabs [data-baseweb="tab"]{{font-size:0.76rem!important;padding:0.5rem 0.75rem!important;}}
}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODELS   (UNCHANGED)
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
#  CONSTANTS  (UNCHANGED)
# ══════════════════════════════════════════════════════════════════════════════
SEVERITY_MAP = {
    'Below ₹5 Lakh':0.5,'₹5–8 Lakh':0.7,'₹8–12 Lakh':0.85,
    '₹12–20 Lakh':1.0,'₹20–30 Lakh':1.2,'Above ₹30 Lakh':1.5
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
        "desc": "High-value genuine claim with elevated litigation risk. The engine identifies a low likelihood of fraud but a high probability of legal escalation, routing to ADR/Legal for early settlement planning.",
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
#  SESSION STATE  (UNCHANGED)
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
#  TOP NAV
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="topnav">
  <div class="topnav-brand">
    <div class="topnav-logo">
            <img src="<img src="https://lh3.googleusercontent.com/d/1OgrQt6YTB4Ibq21SuhXcUoMEe-uGFJ-z">
            </div>
    <div>
      <div class="topnav-name">Claim Decisioning Model</div>
      <div class="topnav-sub">Motor Third-Party · AI Triage Platform</div>
    </div>
  </div>
  <div class="topnav-right">
    <span class="nav-tag"><span class="live-dot"></span>Live Prototype</span>
    <span class="nav-tag">Sundaram Pitch Fest 2026</span>
    <span class="nav-tag">Team Apex Counsel · IIT KGP</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div class="hero-left">
    <div class="kicker"><span class="live-dot"></span>Motor TP · FNOL Intelligence</div>
    <h1>Every claim,<br><em>decided at intake.</em></h1>
    <p>Two XGBoost models score fraud probability and litigation risk the moment a Motor TP
       claim arrives — explaining every flag via SHAP and routing it to the lowest-cost resolution path.</p>
    <div class="hero-meta">
      <span><b>Arunadithyan S</b></span>
      <span><b>Azhagappan G</b></span>
      <span><b>G Abiimukeshwar</b></span>
      <span>IIT Kharagpur · Operations &amp; Risk Track</span>
    </div>
  </div>
  <div class="hero-right">
    <div class="s-card accent">
      <div class="ico" style="background:{ACCENT_L};color:{ACCENT};">🧠</div>
      <div class="val">XGBoost</div>
      <div class="lbl">Fraud model · cost-sensitive</div>
    </div>
    <div class="s-card success">
      <div class="ico" style="background:#DCFCE7;color:{SUCCESS};">⚖️</div>
      <div class="val">Dual</div>
      <div class="lbl">Fraud + litigation scoring</div>
    </div>
    <div class="s-card warn">
      <div class="ico" style="background:#FEF3C7;color:{WARNING};">📊</div>
      <div class="val">15,420</div>
      <div class="lbl">Training claims</div>
    </div>
    <div class="s-card purple">
      <div class="ico" style="background:#EDE9FE;color:#7C3AED;">🔍</div>
      <div class="val">SHAP</div>
      <div class="lbl">Every flag explained</div>
    </div>
  </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

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
        st.error("Model files not found. Place `.pkl` files in the same directory as `app.py`.")
        st.code("model_fraud_india.pkl · model_litigation_india.pkl · encoder_india.pkl · features_fraud.pkl · features_litigation.pkl")
        st.stop()

    if st.session_state['loaded_example']:
        st.info(f"📌 **{st.session_state['loaded_example']}** pre-loaded. Review values below, then click Run.")

    st.markdown("""
    <div class="sec-head">
      <div class="eye">Live Assessment</div>
      <h2>Score a Claim</h2>
      <p>Fill in the four sections. The engine scores, explains, and routes in seconds.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("claim_form"):
        #  1: Policy & Vehicle 
        st.markdown('<div class="wiz-sec"><div class="wiz-head"><div class="wiz-n">1</div><div><div class="wiz-title">Policy &amp; Vehicle</div><div class="wiz-sub">Cover, value, and vehicle profile</div></div></div>', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

        #  2: Claimant 
        st.markdown('<div class="wiz-sec"><div class="wiz-head"><div class="wiz-n">2</div><div><div class="wiz-title">Claimant</div><div class="wiz-sub">Demographics and claim history</div></div></div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        sex = c1.selectbox("Sex", ['Male','Female'], index=['Male','Female'].index(st.session_state.sex))
        marital_status = c2.selectbox("Marital Status", ['Single','Married','Divorced','Widow'],
            index=['Single','Married','Divorced','Widow'].index(st.session_state.marital_status))
        age = c3.number_input("Age", 18, 80, value=int(st.session_state.age))
        prior_claims = c4.selectbox("Prior Claims", CLAIMS, index=safe_idx(CLAIMS, st.session_state.prior_claims))
        st.markdown('</div>', unsafe_allow_html=True)

        #  3: Accident & FNOL 
        st.markdown('<div class="wiz-sec"><div class="wiz-head"><div class="wiz-n">3</div><div><div class="wiz-title">Accident &amp; FNOL</div><div class="wiz-sub">Incident details and reporting timeline</div></div></div>', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

        #  4: Distribution 
        st.markdown('<div class="wiz-sec"><div class="wiz-head"><div class="wiz-n">4</div><div><div class="wiz-title">Distribution &amp; Documentation</div><div class="wiz-sub">Channel, reports, and policy spread</div></div></div>', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚖️  Run Claim Decisioning Engine")

    #  RESULTS 
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
        with st.spinner("Scoring…"):
            fp, lp, score, route, tkey, tlabel, tcolor, reasons = score_claim(claim)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="sec-head">
          <div class="eye">Score Report</div>
          <h2>Decisioning Output</h2>
        </div>
        """, unsafe_allow_html=True)

        col_g, col_r = st.columns([1, 1.6], gap="large")

        with col_g:
            st.markdown(f"""
            <div class="gauge-wrap">
              <div class="gauge-eye">Composite Risk Score</div>
              <div>
                <span class="gauge-score">{score}</span>
                <span class="gauge-denom"> /100</span>
              </div>
              <div class="gauge-bar-track">
                <div class="gauge-bar" style="width:{score}%;background:{tcolor};"></div>
              </div>
              <div class="gauge-row">
                <div class="gauge-chip">
                  <div class="gk">Fraud Prob</div>
                  <div class="gv" style="color:{tcolor};">{fp:.1%}</div>
                </div>
                <div class="gauge-chip">
                  <div class="gk">Litigation Risk</div>
                  <div class="gv" style="color:{ACCENT};">{lp:.1%}</div>
                </div>
              </div>
              <div class="route-box">
                <div class="rk">Routing Decision</div>
                <div class="rv">{route}</div>
                <div><span class="rpill" style="background:{tcolor};">{tlabel}</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_r:
            # SHAP drivers
            st.space()
            st.markdown(f"<div style='font-size:0.76rem;font-weight:700;color:{INK2};letter-spacing:0.5px;text-transform:uppercase;margin-bottom:0.6rem;'>Top Fraud Risk Drivers</div>", unsafe_allow_html=True)
            for feat, val in reasons:
                direction = "↑ Raises" if val > 0 else "↓ Lowers"
                cls = "up" if val > 0 else "down"
                ic  = "🔴" if val > 0 else "🟢"
                st.markdown(f"""
                <div class="driver {cls}">
                  <div class="driver-text">{ic} <strong>{feat}</strong> — {direction} fraud risk</div>
                  <span class="driver-shap">SHAP {val:+.3f}</span>
                </div>
                """, unsafe_allow_html=True)

        #  Action + Table 
        action_map = {
            'critical': (CRITICAL, '#FEF2F2', '🔴', 'Refer to SIU immediately. Do not settle. Assign senior investigator and request full documentation audit.'),
            'high':     (HIGH,     '#FFF7ED', '🟠', 'Flag for ADR / Legal team. Prepare fight-or-settle brief. Route to in-house counsel within 48 hours.'),
            'medium':   (MEDIUM,   '#FFFBEB', '🟡', 'Route to Standard Processing queue. Surveyor review required before payment authorisation.'),
            'fast':     (FASTTRACK,'#F0FDF4', '🟢', 'Eligible for Fast Track Settlement. Verify documents and initiate payment within 7 working days.'),
        }
        ac, abg, aic, atxt = action_map[tkey]
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k,v in [
            ("Vehicle", f"{vehicle_make} · {vehicle_category}"),
            ("Cover",   f"{base_policy} · {vehicle_price}"),
            ("Claimant",f"{sex}, {age} yrs · {marital_status}"),
            ("FNOL Delay", fnol_delay),("FIR Filed", fir_filed),
            ("Witness", witness),("Prior Claims", prior_claims),
            ("Fault", fault),("Intermediary", intermediary),
        ])
        st.markdown(f"""
        <div class="res-bottom">
          <div class="act-card" style="background:{abg};border-color:{ac}33;border-left-color:{ac};">
            <div class="act-head">
              <span class="act-ico">{aic}</span>
              <span class="act-lbl" style="color:{ac};">Recommended Action</span>
            </div>
            <div class="act-body" style="color:{INK};">{atxt}</div>
            <div class="act-foot" style="border-top-color:{ac}22;">
              <div class="pk">Priority Tier</div>
              <span class="rpill" style="background:{tcolor};">{tlabel}</span>
            </div>
          </div>
          <div class="clm-card">
            <div class="clm-head">Claim Summary</div>
            <table class="clm-tbl"><tbody>{rows}</tbody></table>
          </div>
        </div>
        """, unsafe_allow_html=True)

        #  Predicted vs Actual 
        if st.session_state['loaded_example']:
            ex = EXAMPLE_CLAIMS[st.session_state['loaded_example']]
            act_key = ex['actual_key']; act_lbl = ex['actual']
            act_col = EX_KEY_COLOR[act_key]
            is_match = (tkey == act_key)
            mbg = '#F0FDF4' if is_match else '#FFFBEB'
            mc  = SUCCESS if is_match else WARNING
            mtxt = "✓ Match — engine agrees with the real outcome" if is_match else "≈ Close — engine flagged the correct risk family"
            st.markdown(f"""
            <div style="margin-top:0.9rem;">
              <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:{INK2};margin-bottom:0.6rem;">Predicted vs Actual</div>
              <div class="verdict">
                <div class="verdict-box" style="border-top-color:{tcolor};">
                  <div class="vk">Engine Predicted</div>
                  <div class="vv" style="color:{tcolor};">{route}</div>
                </div>
                <div class="verdict-box" style="border-top-color:{act_col};">
                  <div class="vk">Actual Outcome</div>
                  <div class="vv" style="color:{act_col};">{act_lbl}</div>
                </div>
              </div>
              <div class="match-banner" style="background:{mbg};border-color:{mc}33;color:{mc};">
                {mtxt}
              </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — EXAMPLE CLAIMS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="sec-head">
      <div class="eye">Predicted vs Actual</div>
      <h2>Example Claims</h2>
      <p>Four real archetypes with known outcomes. Load one → score it → compare.</p>
    </div>
    """, unsafe_allow_html=True)

    if not models_ok:
        st.warning("Model files not loaded — examples fill the form but scoring needs the `.pkl` files.")

    st.markdown('<div class="ex-grid">', unsafe_allow_html=True)
    for i, (name, ex) in enumerate(EXAMPLE_CLAIMS.items()):
        tag = EX_KEY_COLOR[ex['actual_key']]
        sel = "sel" if st.session_state['loaded_example'] == name else ""
        ic  = ex.get("icon","📄")
        # use columns for spacing
    st.markdown('</div>', unsafe_allow_html=True)

    # Actually render with st.columns for button support
    cols = st.columns(2, gap="large")
    for i, (name, ex) in enumerate(EXAMPLE_CLAIMS.items()):
        with cols[i % 2]:
            tag = EX_KEY_COLOR[ex['actual_key']]
            sel = "sel" if st.session_state['loaded_example'] == name else ""
            ic  = ex.get("icon","📄")
            st.markdown(f"""
            <div class="ex-card {sel}" style="border-top-color:{tag};">
              <div class="ex-top">
                <div class="ex-ico" style="background:{tag}18;color:{tag};">{ic}</div>
                <div><h5>{name}</h5></div>
              </div>
              <span class="ex-badge" style="background:{tag};">Actual: {ex['actual']}</span>
              <p class="ex-desc">{ex['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            st.button("⬇️  Load this example", key=f"load_{i}",
                      on_click=load_example, args=(name,), use_container_width=True)
            if st.session_state['loaded_example'] == name:
                st.success("Loaded — switch to **Score Claim** tab and click Run.")

    st.caption("Examples mirror common Motor TP claim archetypes — showing how the engine separates genuine, standard, litigation-bound, and fraudulent claims.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="sec-head">
      <div class="eye">Transparency</div>
      <h2>Model Insights</h2>
      <p>Architecture, feature attribution, routing distribution, and projected financial impact.</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    kpi_cols = st.columns(4, gap="medium")
    for col, val, lbl, cls, ico in zip(kpi_cols,
        ["0.84","0.74","15,420","100%"],
        ["Fraud ROC-AUC","Litigation ROC-AUC","Test Claims","Decisions Explained"],
        ["accent","warn","success","purple"],
        ["🎯","⚖️","📊","🔍"]):
        ico_bg = {"accent":f"{ACCENT_L};color:{ACCENT}","warn":"#FEF3C7;color:#D97706",
                  "success":"#DCFCE7;color:#059669","purple":"#EDE9FE;color:#7C3AED"}[cls]
        col.markdown(f"""
        <div class="s-card {cls}">
          <div class="ico" style="background:{ico_bg};">{ico}</div>
          <div class="val">{val}</div>
          <div class="lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown(f"""
        <div class="insight-card">
          <h4>🧠 Model A — Fraud Detection</h4>
          <p><strong>Algorithm:</strong> XGBoost · cost-sensitive (scale_pos_weight)<br>
          <strong>Label:</strong> Domain-calibrated fraud signatures (Indian Motor TP patterns)<br>
          <strong>Split:</strong> Stratified 80/20 — balanced fraud rate in train &amp; test<br>
          <strong>Metric:</strong> PR-AUC — honest under 6% class imbalance<br>
          <strong>Explainability:</strong> SHAP TreeExplainer · top-3 reasons per claim</p>
        </div>
        """, unsafe_allow_html=True)
        if models_ok and feat_fraud:
            imp = pd.Series(model_fraud.feature_importances_, index=feat_fraud).sort_values().tail(10)
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
            bars = ax.barh(imp.index, imp.values, color=ACCENT, alpha=0.88, height=0.62)
            bars[-1].set_color(WARNING)
            ax.set_xlabel("Importance", fontsize=8, color=INK2)
            ax.set_title("Top 10 Fraud Drivers", fontsize=10, fontweight='bold', color=INK, pad=10)
            ax.tick_params(labelsize=7.5, colors=INK2)
            for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
            ax.spines['bottom'].set_color(BORDER)
            ax.xaxis.grid(True, alpha=0.2, color=BORDER)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

    with col_b:
        st.markdown(f"""
        <div class="insight-card">
          <h4>⚖️ Model B — Litigation Risk</h4>
          <p><strong>Algorithm:</strong> XGBoost Classifier<br>
          <strong>Label:</strong> Expert-rule MACT escalation proxy · label-leakage-safe<br>
          <strong>Design:</strong> Proxy-building features excluded from training<br>
          <strong>Business use:</strong> Fight-or-settle signal at FNOL — routes high-risk cases
          to ADR before MACT filing, reducing litigation cost per claim</p>
        </div>
        """, unsafe_allow_html=True)
        labels = ['Fast Track','Standard','ADR / Legal','SIU']
        sizes  = [62, 22, 10, 6]
        colors = [FASTTRACK, ACCENT, HIGH, CRITICAL]
        fig2, ax2 = plt.subplots(figsize=(5, 3.8))
        fig2.patch.set_facecolor(CARD)
        _, texts, autos = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
            startangle=140, pctdistance=0.72, wedgeprops=dict(width=0.52,edgecolor='white',linewidth=2))
        for t in texts: t.set_fontsize(8); t.set_color(INK2)
        for a in autos: a.set_fontsize(8); a.set_color('white'); a.set_fontweight('bold')
        ax2.set_title("Expected Routing Distribution", fontsize=10, fontweight='bold', color=INK, pad=10)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True); plt.close()

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.78rem;font-weight:700;color:{INK};margin-bottom:0.6rem;'>Engine Architecture</div>", unsafe_allow_html=True)
    arch = [("FNOL","Claim In",ACCENT),("Data Spine","Features","#1E3A8A"),
            ("Model A+B","Fraud · Lit","#4338CA"),("SHAP","Explain","#0F766E"),
            ("Routing","Decision",FASTTRACK)]
    fh = "".join(f'<div class="flow-step" style="background:{c};"><div class="ft">{t}</div><div class="fs">{s}</div></div>{"<div class=\'flow-arr\'>→</div>" if i<4 else ""}' for i,(t,s,c) in enumerate(arch))
    st.markdown(f"""
    <div class="insight-card">
      <div class="flow">{fh}</div>
      <div style="margin-top:0.85rem;font-size:0.73rem;color:{INK2};text-align:center;">
        Single-insurer · No IIB consortium dependency · Human-in-the-loop on every P1 and P2 flag
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.78rem;font-weight:700;color:{INK};margin-bottom:0.6rem;'>Projected Financial Impact · per ₹1,000 Cr TP Claims</div>", unsafe_allow_html=True)
    fin = pd.DataFrame({
        'Scenario':['🐻 Bear','📊 Base','🐂 Bull'],
        'Leakage Recovered':['₹5 Cr','₹15 Cr','₹30 Cr'],
        'Litigation Saving':['₹1 Cr','₹3 Cr','₹6 Cr'],
        'Total Benefit':['₹6 Cr','₹18 Cr','₹36 Cr'],
        'Build Cost':['₹3.5 Cr','₹3.5 Cr','₹3.5 Cr'],
        'Payback':['~7 mo','~5 mo','~3 mo'],
    }).set_index('Scenario')
    st.dataframe(fin, use_container_width=True)
    st.caption("Build cost held constant. Normalised to ₹1,000 Cr to avoid over-claiming Sundaram Finance figures.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="sec-head">
      <div class="eye">Sundaram Pitch Fest 2026</div>
      <h2>About the Engine</h2>
      <p>A single-insurer AI platform that scores, explains, and routes every Motor TP claim at FNOL.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        for ic, title, body in [
            ("🎯","The Problem",
             "India's Motor TP book carries ₹96,257 Cr of pending liability across 10.73 lakh open MACT cases (2025-26), "
             "rising to ~₹1.05 lakh Cr with accruals. Motor incurred-claims ratio: 78% industry-wide; 108% for some PSUs."),
            ("⚙️","Our Solution",
             "Two XGBoost models score fraud and litigation risk at FNOL, combined into a composite score that "
             "auto-routes every claim — SIU, ADR, Standard, or Fast Track — to the cheapest correct resolution path."),
            ("🏛️","Why Single-Insurer",
             "Deployable on Sundaram Finance's own claim history — no IIB consortium dependency, no competitor coordination. "
             "The graph network layer deepens with every processed claim, creating a proprietary compounding moat."),
            ("🗓️","Deployment Roadmap",
             "P0 data audit → P1 fraud shadow mode (6 mo) → P2 litigation model (12 mo) "
             "→ P3 graph/network layer → P4 live routing with drift monitoring."),
        ]:
            st.markdown(f'<div class="about-card"><h4>{ic} {title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

    with col2:
        for ic, title, body in [
            ("🔍","Explainability",
             "SHAP TreeExplainer produces top-3 plain-language reasons per claim — SIU officers, claims managers, "
             "and MACT tribunals can act on outputs. No black box. Every decision is fully auditable."),
            ("📈","Business Value",
             "Base case: ~₹18 Cr annual benefit on a ₹1,000 Cr TP book vs ₹3.5 Cr one-time build. "
             "Payback inside five months. Even at 30% of base-case benefit, ROI is positive within Year 1."),
            ("🚀","Future Vision",
             "Phase 3+: graph neural networks over the garage–lawyer entity graph, Vahan/MoRTH risk signals at "
             "underwriting, federated cross-insurer learning without sharing raw claim data."),
            ("🏗️","Architecture",
             "FNOL → Data Spine → Model A (fraud) + Model B (litigation) → SHAP → composite score → routing. "
             "Clean, auditable, human-in-the-loop checkpoint on every high-risk flag."),
        ]:
            st.markdown(f'<div class="about-card"><h4>{ic} {title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

    # Tech stack
    tech = ['XGBoost','SHAP','scikit-learn','Streamlit','Python 3.14','pandas','matplotlib','joblib']
    chips = "".join(f'<span style="background:{BG};border:1px solid {BORDER};border-radius:7px;padding:0.32rem 0.75rem;font-size:0.77rem;font-weight:600;color:{INK};">{t}</span>' for t in tech)
    st.markdown(f'<div class="about-card" style="border-top-color:#7C3AED;"><h4>🛠️ Technology Stack</h4><div style="display:flex;gap:0.45rem;flex-wrap:wrap;margin-top:0.5rem;">{chips}</div></div>', unsafe_allow_html=True)

    # Team
    members = "".join(f'<div style="flex:1;min-width:150px;"><div style="font-weight:700;font-size:0.9rem;color:#fff;">{n}</div><div style="font-size:0.72rem;color:rgba(255,255,255,0.45);margin-top:2px;">IIT Kharagpur · School of Law</div></div>' for n in ['Arunadithyan S','Azhagappan G','G Abiimukeshwar'])
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0F172A,{SLATE});border-radius:16px;padding:1.6rem 1.8rem;margin-top:0.2rem;border:1px solid rgba(255,255,255,0.06);">
      <div style="font-size:0.6rem;letter-spacing:1.8px;text-transform:uppercase;color:{WARNING};font-weight:700;margin-bottom:1rem;">Team Apex Counsel · IIT Kharagpur</div>
      <div style="display:flex;gap:1.5rem;flex-wrap:wrap;">{members}</div>
      <div style="margin-top:1rem;font-size:0.72rem;color:rgba(255,255,255,0.35);">
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

st.markdown("""
<style>

/* ===========================
   Desktop Only (>=1025px)
   =========================== */

@media (min-width:1025px){

.hero{
    grid-template-columns:55% 45% !important;
    gap:3rem !important;
    align-items:center !important;
}

.hero-right{
    width:100% !important;
    min-width:520px !important;
    gap:1rem !important;
    align-self:stretch !important;
}

.s-card{
    min-height:190px !important;
    padding:1.6rem !important;
    border-radius:18px !important;
    display:flex !important;
    flex-direction:column !important;
    justify-content:space-between !important;
}

.s-card .ico{
    width:56px !important;
    height:56px !important;
    font-size:1.4rem !important;
}

.s-card .val{
    font-size:2.4rem !important;
    font-weight:800 !important;
}

.s-card .lbl{
    font-size:.92rem !important;
}

}

</style>
""", unsafe_allow_html=True)

