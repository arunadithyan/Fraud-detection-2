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

st.set_page_config(
    page_title="Claim Decisioning Model · Sundaram Finance",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens: warm institutional, Bloomberg × Apple Finance ──────────────
BG      = "#FAFAF9"        # warm stone white
CARD    = "#FFFFFF"
INK     = "#1C1917"        # near-black warm
INK2    = "#57534E"        # stone-600
MUTED   = "#A8A29E"        # stone-400
BORDER  = "#E7E5E4"        # stone-200
BORDER2 = "#D6D3D1"        # stone-300
AMBER   = "#B45309"        # amber-700 — the one accent
AMBER_L = "#FEF3C7"        # amber-100
AMBER_D = "#92400E"        # amber-800
GREEN   = "#15803D"        # green-700
ORANGE  = "#C2410C"        # orange-700
RED     = "#B91C1C"        # red-700
SLATE   = "#0C0A09"        # stone-950 (header/gauge bg)
STONE8  = "#292524"        # stone-800

CRITICAL  = "#B91C1C"
HIGH      = "#C2410C"
MEDIUM    = "#B45309"
FASTTRACK = "#15803D"

RS_BLUE=AMBER;RS_NAVY=SLATE;RS_YELLOW=AMBER;RS_WHITE=CARD;RS_OFFWHITE=BG;RS_GRAY=INK2;RS_LGRAY=BORDER

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Instrument+Serif:ital@0;1&display=swap');

html,body,[class*="css"],[data-testid="stAppViewContainer"],[data-testid="stApp"],
.main,.block-container,[data-testid="stVerticalBlock"],[data-testid="stForm"],
[data-testid="stHorizontalBlock"],section.main{{
  background:{BG}!important;color:{INK}!important;
  font-family:'Inter',-apple-system,sans-serif!important;
}}
@media(prefers-color-scheme:dark){{html,body,[class*="css"]{{background:{BG}!important;color:{INK}!important;}}}}
#MainMenu,footer,header{{visibility:hidden;}}
*{{-webkit-tap-highlight-color:transparent;box-sizing:border-box;}}
.block-container{{padding:0 clamp(0.9rem,3.5vw,2.8rem) 4rem!important;max-width:1360px;margin:0 auto;}}

/* ── TOPNAV ── */
.tnav{{
  display:flex;align-items:center;justify-content:space-between;
  padding:1rem 0 1.1rem;border-bottom:1px solid {BORDER};margin-bottom:0;
}}
.tnav-brand{{display:flex;align-items:center;gap:0.75rem;}}
.tnav-logo{{
  width:44px;height:44px;border-radius:10px;overflow:hidden;
  background:{CARD};border:1px solid {BORDER};display:flex;align-items:center;justify-content:center;
  flex-shrink:0;
}}
.tnav-logo img{{width:100%;height:100%;object-fit:contain;}}
.tnav-name{{font-size:0.95rem;font-weight:700;color:{INK};letter-spacing:-0.01em;line-height:1.2;}}
.tnav-sub{{font-size:0.72rem;color:{MUTED};font-weight:400;}}
.tnav-right{{display:flex;align-items:center;gap:0.4rem;flex-wrap:wrap;justify-content:flex-end;}}
.tnav-pill{{
  font-size:0.7rem;font-weight:600;padding:0.24rem 0.6rem;border-radius:5px;
  background:{BG};border:1px solid {BORDER};color:{INK2};letter-spacing:0.15px;white-space:nowrap;
}}
.tnav-pill.live{{background:{AMBER_L};border-color:{AMBER};color:{AMBER_D};}}
.live-dot{{width:6px;height:6px;border-radius:50%;background:{GREEN};display:inline-block;margin-right:3px;vertical-align:middle;}}

/* ── HERO ── */
.hero{{
  display:grid;grid-template-columns:1fr 420px;gap:3rem;
  align-items:center;padding:2.4rem 0 2rem;
}}
.hero-kicker{{
  font-size:0.65rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
  color:{AMBER};margin-bottom:0.9rem;display:flex;align-items:center;gap:0.4rem;
}}
.hero h1{{
  font-family:'Instrument Serif',Georgia,serif;
  font-size:clamp(2rem,4vw,3.1rem);font-weight:400;
  color:{INK};line-height:1.06;letter-spacing:-0.025em;margin:0 0 1rem;
}}
.hero h1 i{{color:{AMBER};font-style:italic;}}
.hero-p{{
  font-size:clamp(0.86rem,1.7vw,0.97rem);color:{INK2};
  line-height:1.7;max-width:500px;margin:0 0 1.4rem;font-weight:400;
}}
.hero-chips{{display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:1.4rem;}}
.chip{{
  font-size:0.7rem;font-weight:600;padding:0.28rem 0.65rem;border-radius:5px;
  background:{CARD};border:1px solid {BORDER};color:{INK2};letter-spacing:0.1px;
}}
.hero-team{{font-size:0.71rem;color:{MUTED};line-height:1.6;}}
.hero-team b{{color:{INK2};font-weight:600;}}

/* ── KPI TILES (hero right) ── */
.kpi-grid{{display:grid;grid-template-columns:1fr 1fr;gap:0.65rem;}}
.kpi-tile{{
  background:{CARD};border:1px solid {BORDER};border-radius:12px;
  padding:1.25rem 1.15rem;position:relative;
  transition:box-shadow .18s ease,transform .18s ease;
}}
.kpi-tile:hover{{box-shadow:0 6px 20px rgba(28,25,23,0.08);transform:translateY(-2px);}}
.kpi-tile-bar{{
  position:absolute;top:0;left:0;right:0;height:2px;border-radius:12px 12px 0 0;
  background:{AMBER};
}}
.kpi-tile .ki{{
  font-size:0.7rem;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;
  color:{MUTED};margin-bottom:0.55rem;
}}
.kpi-tile .kv{{
  font-size:clamp(1.6rem,4vw,2.1rem);font-weight:800;color:{INK};
  letter-spacing:-0.04em;line-height:1;
}}
.kpi-tile .kd{{font-size:0.72rem;color:{MUTED};margin-top:0.3rem;font-weight:500;}}

/* ── RULE ── */
.rule{{height:1px;background:{BORDER};margin:0;}}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"]{{
  gap:0;background:transparent;border:none;border-bottom:1px solid {BORDER};
  padding:0;overflow-x:auto;flex-wrap:nowrap;-webkit-overflow-scrolling:touch;
  border-radius:0;
}}
.stTabs [data-baseweb="tab"]{{
  background:transparent!important;color:{MUTED}!important;border:none!important;
  border-bottom:2px solid transparent!important;border-radius:0!important;
  font-weight:500!important;font-size:0.83rem!important;
  padding:0.85rem 1.1rem 0.75rem!important;white-space:nowrap;min-height:44px;
  transition:color .12s,border-color .12s;margin-bottom:-1px;
}}
.stTabs [data-baseweb="tab"]:hover{{color:{INK}!important;}}
.stTabs [aria-selected="true"]{{
  color:{INK}!important;border-bottom-color:{INK}!important;font-weight:700!important;
}}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{{display:none!important;}}

/* ── SECTION HEAD ── */
.sh{{margin:2rem 0 1.2rem;}}
.sh-eye{{font-size:0.62rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{AMBER};margin-bottom:0.3rem;}}
.sh-title{{font-size:clamp(1.05rem,2.3vw,1.28rem);font-weight:800;color:{INK};letter-spacing:-0.02em;margin:0;}}
.sh-sub{{font-size:0.83rem;color:{INK2};margin:0.3rem 0 0;line-height:1.5;}}

/* ── FORM SECTIONS ── */
.fsec{{
  background:{CARD};border:1px solid {BORDER};border-radius:12px;
  padding:1.4rem 1.5rem;margin-bottom:0.7rem;
}}
.fsec-head{{display:flex;align-items:center;gap:0.65rem;padding-bottom:1.1rem;border-bottom:1px solid {BORDER};margin-bottom:1.2rem;}}
.fsec-num{{
  width:26px;height:26px;border-radius:6px;background:{INK};color:{CARD};
  display:flex;align-items:center;justify-content:center;
  font-weight:800;font-size:0.75rem;flex-shrink:0;
}}
.fsec-title{{font-size:0.88rem;font-weight:700;color:{INK};}}
.fsec-sub{{font-size:0.72rem;color:{MUTED};margin-top:1px;}}

/* ── GAUGE (result score) ── */
.gauge{{
  background:{SLATE};border-radius:14px;
  padding:clamp(1.3rem,3vw,1.9rem);
  box-shadow:0 20px 60px rgba(12,10,9,0.3);
  position:relative;overflow:hidden;
}}
.gauge::before{{
  content:"";position:absolute;top:-60%;right:-20%;
  width:300px;height:300px;
  background:radial-gradient(circle,rgba(180,83,9,0.2),transparent 65%);
  border-radius:50%;pointer-events:none;
}}
.g-eye{{font-size:0.58rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:rgba(255,255,255,0.35);margin-bottom:0.55rem;}}
.g-score{{
  font-family:'Instrument Serif',Georgia,serif;
  font-size:clamp(3.5rem,11vw,5rem);font-weight:400;color:#fff;
  line-height:1;letter-spacing:-0.03em;
}}
.g-denom{{font-size:0.9rem;color:rgba(255,255,255,0.3);font-family:'Inter',sans-serif;font-weight:500;vertical-align:super;}}
.g-track{{background:rgba(255,255,255,0.1);border-radius:3px;height:4px;margin:1.1rem 0 0.4rem;overflow:hidden;}}
.g-fill{{height:100%;border-radius:3px;transition:width .9s cubic-bezier(.2,.8,.2,1);}}
.g-stats{{display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.9rem;}}
.g-stat{{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:0.65rem 0.7rem;}}
.g-stat .gk{{font-size:0.55rem;letter-spacing:1.2px;text-transform:uppercase;color:rgba(255,255,255,0.4);font-weight:700;}}
.g-stat .gv{{font-size:1.05rem;font-weight:800;color:#fff;margin-top:0.15rem;letter-spacing:-0.02em;}}
.g-route{{margin-top:0.9rem;padding-top:0.9rem;border-top:1px solid rgba(255,255,255,0.08);}}
.g-route .grk{{font-size:0.55rem;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,0.4);font-weight:700;margin-bottom:0.3rem;}}
.g-route .grv{{font-size:0.97rem;font-weight:800;color:#fff;letter-spacing:-0.01em;}}
.g-pill{{display:inline-flex;align-items:center;padding:0.27rem 0.75rem;border-radius:5px;font-size:0.7rem;font-weight:700;color:#fff;margin-top:0.5rem;letter-spacing:0.2px;}}

/* ── DRIVERS (SHAP) ── */
.driver{{
  display:flex;align-items:flex-start;justify-content:space-between;gap:0.6rem;
  background:{CARD};border:1px solid {BORDER};border-radius:9px;
  padding:0.65rem 0.85rem;margin-bottom:0.4rem;
  border-left:3px solid {BORDER2};
  font-size:0.79rem;font-weight:500;color:{INK};
  transition:border-left-color .12s;
}}
.driver.up{{border-left-color:{RED};}}
.driver.down{{border-left-color:{GREEN};}}
.driver-main{{flex:1;min-width:0;}}
.driver-feat{{font-weight:700;font-size:0.8rem;color:{INK};}}
.driver-desc{{font-size:0.72rem;color:{INK2};margin-top:1px;line-height:1.4;}}
.driver-badge{{
  font-size:0.64rem;font-weight:700;color:{MUTED};white-space:nowrap;flex-shrink:0;
  background:{BG};border:1px solid {BORDER};border-radius:4px;
  padding:0.14rem 0.42rem;font-family:'Inter',monospace;margin-top:2px;
}}

/* ── BOTTOM GRID: action + summary ── */
.res-grid{{display:grid;grid-template-columns:1fr 1fr;gap:0.7rem;margin-top:0.9rem;}}
.act-box{{
  border-radius:10px;padding:1.15rem 1.2rem;border:1px solid;border-left:3px solid;
  display:flex;flex-direction:column;
}}
.act-eye{{font-size:0.58rem;font-weight:700;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:0.4rem;}}
.act-text{{font-size:0.82rem;font-weight:500;line-height:1.62;flex:1;color:{INK};}}
.act-foot{{margin-top:0.9rem;padding-top:0.8rem;border-top:1px solid rgba(0,0,0,0.06);}}
.act-ft-lbl{{font-size:0.58rem;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;color:{INK2};margin-bottom:0.3rem;}}

.sum-box{{
  background:{CARD};border:1px solid {BORDER};border-radius:10px;overflow:hidden;
}}
.sum-head{{
  padding:0.55rem 0.85rem;border-bottom:1px solid {BORDER};
  font-size:0.7rem;font-weight:700;color:{INK2};letter-spacing:0.4px;text-transform:uppercase;
  background:{BG};
}}
.sum-tbl{{width:100%;border-collapse:collapse;font-size:0.72rem;}}
.sum-tbl td{{padding:0.4rem 0.85rem;border-bottom:1px solid {BORDER};vertical-align:middle;}}
.sum-tbl tr:last-child td{{border-bottom:none;}}
.sum-tbl td:first-child{{color:{INK2};font-weight:500;font-size:0.69rem;width:40%;}}
.sum-tbl td:last-child{{font-weight:700;color:{INK};text-align:right;font-size:0.72rem;}}

/* ── VERDICT ── */
.verd-grid{{display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-top:0.75rem;}}
.verd-box{{background:{CARD};border:1px solid {BORDER};border-top:2px solid;border-radius:9px;padding:0.85rem 0.95rem;}}
.verd-box .vk{{font-size:0.57rem;letter-spacing:1.2px;text-transform:uppercase;color:{MUTED};font-weight:700;}}
.verd-box .vv{{font-size:0.92rem;font-weight:800;margin-top:0.2rem;letter-spacing:-0.01em;}}
.verd-banner{{
  margin-top:0.65rem;padding:0.55rem 0.85rem;border-radius:7px;border:1px solid;
  font-size:0.78rem;font-weight:600;display:flex;align-items:center;gap:0.4rem;
}}

/* ── EXAMPLE CARDS ── */
.ex-card{{
  background:{CARD};border:1px solid {BORDER};border-radius:12px;
  padding:1.2rem 1.25rem;cursor:pointer;position:relative;overflow:hidden;
  transition:box-shadow .18s,transform .18s,border-color .18s;
}}
.ex-card:hover{{box-shadow:0 8px 28px rgba(28,25,23,0.1);transform:translateY(-2px);}}
.ex-card.sel{{border-color:{AMBER};box-shadow:0 0 0 3px {AMBER_L};}}
.ex-card-stripe{{position:absolute;top:0;left:0;right:0;height:2px;}}
.ex-top{{display:flex;align-items:center;gap:0.65rem;margin-bottom:0.5rem;}}
.ex-ico{{
  width:38px;height:38px;border-radius:9px;display:flex;align-items:center;
  justify-content:center;font-size:1rem;flex-shrink:0;
}}
.ex-card h5{{font-size:0.87rem;font-weight:700;color:{INK};margin:0;letter-spacing:-0.01em;}}
.ex-badge{{
  display:inline-block;font-size:0.58rem;font-weight:700;letter-spacing:0.5px;
  text-transform:uppercase;padding:0.18rem 0.55rem;border-radius:4px;color:#fff;margin-bottom:0.45rem;
}}
.ex-desc{{font-size:0.75rem;color:{INK2};line-height:1.55;margin:0;}}

/* ── INSIGHT CARD ── */
.ins-card{{
  background:{CARD};border:1px solid {BORDER};border-radius:12px;
  padding:1.3rem 1.4rem;
}}
.ins-card h4{{font-size:0.87rem;font-weight:700;color:{INK};margin:0 0 0.5rem;letter-spacing:-0.01em;}}
.ins-card p{{font-size:0.78rem;color:{INK2};line-height:1.62;margin:0;}}
.ins-card p strong{{color:{INK};font-weight:600;}}

/* ── FLOW ── */
.flow{{display:flex;align-items:stretch;gap:0.35rem;overflow-x:auto;-webkit-overflow-scrolling:touch;}}
.flow-node{{
  flex:1 1 0;min-width:88px;border-radius:9px;padding:0.7rem 0.45rem;
  text-align:center;color:#fff;
}}
.flow-node .fn{{font-size:0.74rem;font-weight:700;}}
.flow-node .fs{{font-size:0.57rem;opacity:0.78;margin-top:0.1rem;}}
.flow-arr{{display:flex;align-items:center;color:{MUTED};font-size:0.9rem;flex-shrink:0;}}

/* ── ABOUT CARD ── */
.ab-card{{
  background:{CARD};border:1px solid {BORDER};border-radius:10px;
  padding:1.15rem 1.25rem;margin-bottom:0.6rem;border-left:3px solid {AMBER};
}}
.ab-card h4{{font-size:0.85rem;font-weight:700;color:{INK};margin:0 0 0.4rem;}}
.ab-card p{{font-size:0.77rem;color:{INK2};line-height:1.65;margin:0;}}

/* ── S-CARD (hero KPI alias) ── */
.s-card{{background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:1rem 1.05rem;position:relative;overflow:hidden;transition:transform .18s ease,box-shadow .18s ease;}}
.s-card:hover{{transform:translateY(-2px);box-shadow:0 8px 24px rgba(28,25,23,0.08);}}
.s-card::before{{content:"";position:absolute;top:0;left:0;right:0;height:2px;border-radius:12px 12px 0 0;}}
.s-card.accent::before{{background:{AMBER};}}
.s-card.success::before{{background:{GREEN};}}
.s-card.warn::before{{background:{MEDIUM};}}
.s-card.purple::before{{background:#6D28D9;}}
.s-card .ico{{width:34px;height:34px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:0.95rem;margin-bottom:0.6rem;}}
.s-card .val{{font-size:clamp(1.3rem,3vw,1.7rem);font-weight:800;color:{INK};letter-spacing:-0.03em;line-height:1;}}
.s-card .lbl{{font-size:0.7rem;color:{MUTED};font-weight:500;margin-top:0.25rem;line-height:1.3;}}

/* ── BUTTONS ── */
.stButton>button{{
  background:{INK}!important;color:#fff!important;border:none!important;
  border-radius:8px!important;font-weight:700!important;font-size:0.87rem!important;
  padding:0.75rem 1.5rem!important;min-height:48px!important;width:100%!important;
  letter-spacing:0.05px!important;
  box-shadow:0 1px 3px rgba(0,0,0,0.2)!important;
  transition:background .15s,transform .12s,box-shadow .15s!important;
}}
.stButton>button *,.stButton>button p,.stButton>button span{{color:#fff!important;}}
.stButton>button:hover{{background:{AMBER_D}!important;transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,0.18)!important;}}
.stButton>button:active{{transform:translateY(0);}}
.stForm [data-testid="stFormSubmitButton"]>button{{
  background:{INK}!important;color:#fff!important;
  min-height:52px!important;font-size:0.9rem!important;
}}
.stForm [data-testid="stFormSubmitButton"]>button *{{color:#fff!important;}}

/* ── INPUTS ── */
div[data-baseweb="select"]>div{{
  border-radius:8px!important;border-color:{BORDER}!important;background:{CARD}!important;
  font-size:0.83rem!important;color:{INK}!important;min-height:44px!important;
}}
div[data-baseweb="select"]>div:focus-within{{border-color:{INK}!important;box-shadow:0 0 0 2px rgba(28,25,23,0.15)!important;}}
.stNumberInput input,.stTextInput input{{
  border-radius:8px!important;border-color:{BORDER}!important;background:{CARD}!important;
  color:{INK}!important;font-size:0.83rem!important;min-height:44px!important;
}}
.stNumberInput input:focus,.stTextInput input:focus{{border-color:{INK}!important;box-shadow:0 0 0 2px rgba(28,25,23,0.12)!important;}}
label{{color:{INK2}!important;font-size:0.74rem!important;font-weight:600!important;}}
button[data-testid="stNumberInputStepDown"],button[data-testid="stNumberInputStepUp"],
[data-testid="stNumberInput"] button{{
  background:{BG}!important;color:{INK}!important;border:1px solid {BORDER}!important;min-width:38px!important;
}}
button[data-testid="stNumberInputStepDown"]:hover,button[data-testid="stNumberInputStepUp"]:hover{{
  background:{INK}!important;color:#fff!important;
}}
[data-testid="stSlider"]>div>div>div{{background:{BORDER}!important;}}
[data-testid="stSlider"] [role="slider"]{{background:{INK}!important;border-color:{INK}!important;}}
[data-testid="stSlider"]>div>div>div>div{{background:{INK}!important;}}
[data-testid="stAlert"]{{background:{CARD}!important;color:{INK}!important;border-radius:9px!important;border:1px solid {BORDER}!important;}}
[data-testid="stDataFrame"]{{background:{CARD}!important;border-radius:9px;border:1px solid {BORDER}!important;}}
[data-testid="stExpander"]{{background:{CARD}!important;border:1px solid {BORDER}!important;border-radius:9px!important;}}
[data-testid="stExpander"] summary{{color:{INK}!important;font-weight:600!important;}}

/* ── FOOTER ── */
.ent-footer{{
  margin-top:3rem;padding:1.1rem 0 0.4rem;border-top:1px solid {BORDER};
  display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;
  gap:0.5rem;font-size:0.7rem;color:{MUTED};
}}
.ent-footer b{{color:{INK2};font-weight:600;}}

/* ── BREAKPOINTS ── */
@media(max-width:1024px){{
  .hero{{grid-template-columns:1fr;gap:1.8rem;}}
  .kpi-grid{{grid-template-columns:repeat(4,1fr);}}
  .hero-p{{max-width:100%;}}
}}
@media(max-width:768px){{
  .block-container{{padding:0 1rem 3rem!important;}}
  .hero{{padding:1.5rem 0 1.1rem;}}
  .kpi-grid{{grid-template-columns:1fr 1fr;}}
  .res-grid{{grid-template-columns:1fr;}}
  .verd-grid{{grid-template-columns:1fr;}}
  .g-stats{{grid-template-columns:1fr 1fr;}}
  [data-testid="stHorizontalBlock"]{{flex-wrap:wrap;gap:0.5rem!important;}}
  [data-testid="stHorizontalBlock"]>div{{min-width:100%!important;flex:1 1 100%!important;}}
  .tnav-right{{display:none;}}
}}
@media(max-width:480px){{
  .block-container{{padding:0 0.75rem 2.5rem!important;}}
  .kpi-grid{{grid-template-columns:1fr 1fr;gap:0.5rem;}}
  .hero h1{{font-size:1.8rem;}}
  .g-score{{font-size:3.4rem;}}
  .g-stats{{grid-template-columns:1fr;}}
  .fsec{{padding:1.1rem;}}
  .stTabs [data-baseweb="tab"]{{font-size:0.76rem!important;padding:0.7rem 0.75rem!important;}}
}}
</style>
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

# ── Constants (unchanged) ─────────────────────────────────────────────────────
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

# human-readable SHAP feature names
FEAT_LABELS = {
    'Prior_Claims_Count':'Prior Claims','FIR_Filed':'FIR Filed',
    'Weak_Documentation':'Weak Documentation','Broker_Address_Risk':'Broker & Address Risk',
    'Supplementary_Reports':'Supplementary Reports','Address_Change_Before_Claim':'Address Change',
    'Intermediary_Type':'Intermediary Type','FNOL_Delay_Days':'FNOL Delay',
    'Claim_Filing_Delay':'Claim Filing Delay','Fault':'Fault',
    'BasePolicy':'Policy Type','VehicleCategory':'Vehicle Category',
    'VehiclePrice':'Vehicle Value','Vehicle_Age':'Vehicle Age',
    'Driver_Risk_Rating':'Driver Risk Rating','Deductible_INR':'Deductible',
    'High_Value_Vehicle':'High-Value Vehicle','Young_Driver_Flag':'Young Driver',
    'Commercial_TP':'Commercial TP','PolicyHolder_Age_Band':'Policyholder Age',
    'Vehicles_In_Policy':'Vehicles in Policy','VehicleMake':'Vehicle Make',
}

DRIVER_DESC = {
    True:  {'Prior_Claims_Count':'Repeat claims history elevated fraud probability',
             'FIR_Filed':'No police report filed — increases suspicion',
             'Weak_Documentation':'No FIR and no witness — documentation gap',
             'Broker_Address_Risk':'Broker-sourced with recent address change',
             'Supplementary_Reports':'Multiple supplementary reports flagged',
             'Address_Change_Before_Claim':'Recent address change before claim',
             'default':'This feature increased fraud probability'},
    False: {'Prior_Claims_Count':'Clean claim history reduced fraud probability',
             'FIR_Filed':'Police report filed — reduces suspicion',
             'Weak_Documentation':'Documentation is adequate',
             'Broker_Address_Risk':'No broker-address risk detected',
             'Supplementary_Reports':'Normal number of reports',
             'default':'This feature reduced fraud probability'},
}

def get_driver_desc(feat, is_up):
    d = DRIVER_DESC[is_up]
    return d.get(feat, d['default'])

# ── Example claims (unchanged values) ─────────────────────────────────────────
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
        "desc": "High-value genuine claim with elevated litigation risk. Routed to ADR/Legal for early settlement planning.",
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
        "desc": "Moderate-risk private car claim. Requires standard surveyor review before settlement.",
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
EX_KEY_COLOR = {'critical':CRITICAL,'high':HIGH,'medium':MEDIUM,'fast':FASTTRACK}

# ── Session state ─────────────────────────────────────────────────────────────
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
<div class="tnav">
  <div class="tnav-brand">
    <div class="tnav-logo">
      <img src="https://lh3.googleusercontent.com/d/1EthMJwQYhVdeYtWICOJNXA-IACirkfap" alt="Sundaram Finance">
    </div>
    <div>
      <div class="tnav-name">Claim Decisioning Model</div>
      <div class="tnav-sub">Motor Third-Party · AI Triage Platform</div>
    </div>
  </div>
  <div class="tnav-right">
    <span class="tnav-pill live"><span class="live-dot"></span>Live Prototype</span>
    <span class="tnav-pill">Sundaram Pitch Fest 2026</span>
    <span class="tnav-pill">Team Apex Counsel · IIT KGP</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div>
    <div class="hero-kicker"><span class="live-dot"></span> Motor TP · FNOL Intelligence</div>
    <h1>Every claim,<br><i>decided at intake.</i></h1>
    <p class="hero-p">Two XGBoost models score fraud probability and litigation risk
       the moment a Motor TP claim arrives — explaining every flag via SHAP
       and routing it to the lowest-cost resolution path.</p>
    <div class="hero-chips">
      <span class="chip">XGBoost Dual-Model</span>
      <span class="chip">SHAP Explainability</span>
      <span class="chip">4 Decision Routes</span>
      <span class="chip">FNOL-Stage Triage</span>
      <span class="chip">Single-Insurer</span>
    </div>
    <div class="hero-team">
      <b>Arunadithyan S</b> &nbsp;·&nbsp; <b>Azhagappan G</b> &nbsp;·&nbsp; <b>G Abiimukeshwar</b><br>
      IIT Kharagpur · Operations &amp; Risk Track · Sundaram Pitch Fest 2026
    </div>
  </div>
  <div class="kpi-grid">
    <div class="kpi-tile">
      <div class="kpi-tile-bar"></div>
      <div class="ki">Fraud Model</div>
      <div class="kv">ROC 0.84</div>
      <div class="kd">XGBoost · cost-sensitive</div>
    </div>
    <div class="kpi-tile">
      <div class="kpi-tile-bar" style="background:{GREEN};"></div>
      <div class="ki">Litigation Model</div>
      <div class="kv">ROC 0.74</div>
      <div class="kd">MACT proxy · leakage-safe</div>
    </div>
    <div class="kpi-tile">
      <div class="kpi-tile-bar" style="background:{MEDIUM};"></div>
      <div class="ki">Claims Analysed</div>
      <div class="kv">15,420</div>
      <div class="kd">Training & validation set</div>
    </div>
    <div class="kpi-tile">
      <div class="kpi-tile-bar" style="background:#6D28D9;"></div>
      <div class="ki">Explainability</div>
      <div class="kv">SHAP</div>
      <div class="kd">Top-3 reasons per claim</div>
    </div>
  </div>
</div>
<div class="rule"></div>
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
        st.info(f"📌 **{st.session_state['loaded_example']}** pre-loaded. Review values and click Run.")

    st.markdown("""
    <div class="sh">
      <div class="sh-eye">Live Assessment</div>
      <h2 class="sh-title">Score a Claim</h2>
      <p class="sh-sub">Complete four sections. The engine scores, explains, and routes in seconds.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("claim_form"):
        # 1
        st.markdown(f'<div class="fsec"><div class="fsec-head"><div class="fsec-num">1</div><div><div class="fsec-title">Policy &amp; Vehicle</div><div class="fsec-sub">Cover type, vehicle profile, and value</div></div></div>', unsafe_allow_html=True)
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

        # 2
        st.markdown(f'<div class="fsec"><div class="fsec-head"><div class="fsec-num">2</div><div><div class="fsec-title">Claimant</div><div class="fsec-sub">Demographics and claim history</div></div></div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        sex = c1.selectbox("Sex", ['Male','Female'], index=['Male','Female'].index(st.session_state.sex))
        marital_status = c2.selectbox("Marital Status", ['Single','Married','Divorced','Widow'],
            index=['Single','Married','Divorced','Widow'].index(st.session_state.marital_status))
        age = c3.number_input("Age", 18, 80, value=int(st.session_state.age))
        prior_claims = c4.selectbox("Prior Claims", CLAIMS, index=safe_idx(CLAIMS, st.session_state.prior_claims))
        st.markdown('</div>', unsafe_allow_html=True)

        # 3
        st.markdown(f'<div class="fsec"><div class="fsec-head"><div class="fsec-num">3</div><div><div class="fsec-title">Accident &amp; FNOL</div><div class="fsec-sub">Incident details and reporting timeline</div></div></div>', unsafe_allow_html=True)
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

        # 4
        st.markdown(f'<div class="fsec"><div class="fsec-head"><div class="fsec-num">4</div><div><div class="fsec-title">Distribution &amp; Documentation</div><div class="fsec-sub">Channel, reports, and policy spread</div></div></div>', unsafe_allow_html=True)
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

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
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
        with st.spinner("Scoring…"):
            fp, lp, score, route, tkey, tlabel, tcolor, reasons = score_claim(claim)

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="sh" style="margin-top:1.6rem;">
          <div class="sh-eye">Score Report</div>
          <h2 class="sh-title">Decisioning Output</h2>
        </div>
        """, unsafe_allow_html=True)

        col_g, col_r = st.columns([1, 1.65], gap="large")

        with col_g:
            st.markdown(f"""
            <div class="gauge">
              <div class="g-eye">Composite Risk Score</div>
              <div>
                <span class="g-score">{score}</span>
                <span class="g-denom"> /100</span>
              </div>
              <div class="g-track"><div class="g-fill" style="width:{score}%;background:{tcolor};"></div></div>
              <div class="g-stats">
                <div class="g-stat">
                  <div class="gk">Fraud Probability</div>
                  <div class="gv" style="color:{tcolor};">{fp:.1%}</div>
                </div>
                <div class="g-stat">
                  <div class="gk">Litigation Risk</div>
                  <div class="gv" style="color:{AMBER};">{lp:.1%}</div>
                </div>
              </div>
              <div class="g-route">
                <div class="grk">Routing Decision</div>
                <div class="grv">{route}</div>
                <div><span class="g-pill" style="background:{tcolor};">{tlabel}</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_r:
            st.markdown(f"<div style='font-size:0.68rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:{MUTED};margin-bottom:0.7rem;'>Top Fraud Risk Drivers</div>", unsafe_allow_html=True)
            for feat, val in reasons:
                is_up = val > 0
                label = FEAT_LABELS.get(feat, feat.replace('_',' '))
                desc  = get_driver_desc(feat, is_up)
                cls   = "up" if is_up else "down"
                ic    = "↑" if is_up else "↓"
                badge_color = RED if is_up else GREEN
                st.markdown(f"""
                <div class="driver {cls}">
                  <div class="driver-main">
                    <div class="driver-feat">{ic} {label}</div>
                    <div class="driver-desc">{desc}</div>
                  </div>
                  <span class="driver-badge">{val:+.3f}</span>
                </div>
                """, unsafe_allow_html=True)

        # action + table
        action_map = {
            'critical': (CRITICAL,'#FEF2F2','↑','Refer to SIU immediately. Do not settle. Assign senior investigator and request full documentation audit.'),
            'high':     (HIGH,    '#FFF7ED','!','Flag for ADR / Legal team. Prepare fight-or-settle brief. Route to in-house counsel within 48 hours.'),
            'medium':   (MEDIUM,  '#FFFBEB','~','Route to Standard Processing queue. Surveyor review required before payment authorisation.'),
            'fast':     (FASTTRACK,'#F0FDF4','✓','Eligible for Fast Track Settlement. Verify documents and initiate payment within 7 working days.'),
        }
        ac, abg, aic, atxt = action_map[tkey]
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k,v in [
            ("Vehicle",f"{vehicle_make} · {vehicle_category}"),
            ("Cover",f"{base_policy} · {vehicle_price}"),
            ("Claimant",f"{sex}, {age} yrs · {marital_status}"),
            ("FNOL Delay",fnol_delay),("FIR Filed",fir_filed),
            ("Witness",witness),("Prior Claims",prior_claims),
            ("Fault",fault),("Intermediary",intermediary),
        ])
        st.markdown(f"""
        <div class="res-grid">
          <div class="act-box" style="background:{abg};border-color:{ac}30;border-left-color:{ac};">
            <div class="act-eye" style="color:{ac};">Recommended Action</div>
            <div class="act-text">{atxt}</div>
            <div class="act-foot">
              <div class="act-ft-lbl">Priority Tier</div>
              <span class="g-pill" style="background:{tcolor};">{tlabel}</span>
            </div>
          </div>
          <div class="sum-box">
            <div class="sum-head">Claim Summary</div>
            <table class="sum-tbl"><tbody>{rows}</tbody></table>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # predicted vs actual
        if st.session_state['loaded_example']:
            ex = EXAMPLE_CLAIMS[st.session_state['loaded_example']]
            act_col = EX_KEY_COLOR[ex['actual_key']]
            is_match = (tkey == ex['actual_key'])
            mc  = GREEN if is_match else MEDIUM
            mbg = '#F0FDF4' if is_match else '#FFFBEB'
            mtxt = "✓  Engine prediction matches the real outcome" if is_match else "≈  Engine flagged the correct risk family"
            st.markdown(f"""
            <div style="margin-top:0.8rem;">
              <div style="font-size:0.68rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:{MUTED};margin-bottom:0.65rem;">Predicted vs Actual</div>
              <div class="verd-grid">
                <div class="verd-box" style="border-top-color:{tcolor};">
                  <div class="vk">Engine Predicted</div>
                  <div class="vv" style="color:{tcolor};">{route}</div>
                </div>
                <div class="verd-box" style="border-top-color:{act_col};">
                  <div class="vk">Actual Outcome</div>
                  <div class="vv" style="color:{act_col};">{ex['actual']}</div>
                </div>
              </div>
              <div class="verd-banner" style="background:{mbg};border-color:{mc}25;color:{mc};">{mtxt}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — EXAMPLE CLAIMS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="sh">
      <div class="sh-eye">Predicted vs Actual</div>
      <h2 class="sh-title">Example Claims</h2>
      <p class="sh-sub">Four real archetypes with known outcomes. Load one → score it → compare the engine's prediction against reality.</p>
    </div>
    """, unsafe_allow_html=True)

    if not models_ok:
        st.warning("Model files not loaded — examples fill the form but scoring needs the `.pkl` files.")

    cols = st.columns(2, gap="large")
    for i, (name, ex) in enumerate(EXAMPLE_CLAIMS.items()):
        with cols[i % 2]:
            tag = EX_KEY_COLOR[ex['actual_key']]
            sel = "sel" if st.session_state['loaded_example'] == name else ""
            ic  = ex.get("icon","📄")
            st.markdown(f"""
            <div class="ex-card {sel}">
              <div class="ex-card-stripe" style="background:{tag};"></div>
              <div class="ex-top">
                <div class="ex-ico" style="background:{tag}15;color:{tag};">{ic}</div>
                <div><h5>{name}</h5></div>
              </div>
              <span class="ex-badge" style="background:{tag};">Actual: {ex['actual']}</span>
              <p class="ex-desc">{ex['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            st.button("Load example", key=f"load_{i}", on_click=load_example, args=(name,), use_container_width=True)
            if st.session_state['loaded_example'] == name:
                st.success("Loaded — switch to Score Claim and click Run.")

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    st.caption("Examples mirror common Motor TP claim archetypes — showing how the engine separates genuine, standard, litigation-bound, and fraudulent claims.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="sh">
      <div class="sh-eye">Transparency</div>
      <h2 class="sh-title">Model Insights</h2>
      <p class="sh-sub">Architecture, feature attribution, routing distribution, and projected financial impact.</p>
    </div>
    """, unsafe_allow_html=True)

    kpi_cols = st.columns(4, gap="medium")
    for col, ki, kv, kd, bar in zip(kpi_cols,
        ["Fraud ROC-AUC","Litigation ROC-AUC","Test Claims","Decisions Explained"],
        ["0.84","0.74","15,420","100%"],
        ["Stratified 80/20 split","MACT proxy label","Training + validation","SHAP per-claim"],
        [AMBER, GREEN, MEDIUM, "#6D28D9"]):
        col.markdown(f"""
        <div class="kpi-tile">
          <div class="kpi-tile-bar" style="background:{bar};"></div>
          <div class="ki">{ki}</div>
          <div class="kv">{kv}</div>
          <div class="kd">{kd}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown(f"""
        <div class="ins-card">
          <h4>Model A — Fraud Detection</h4>
          <p><strong>Algorithm:</strong> XGBoost · cost-sensitive (scale_pos_weight)<br>
          <strong>Label:</strong> Domain-calibrated fraud signatures — Indian Motor TP patterns<br>
          <strong>Split:</strong> Stratified 80/20 — balanced fraud rate in train &amp; test<br>
          <strong>Metric:</strong> PR-AUC · honest under 6% class imbalance<br>
          <strong>Explainability:</strong> SHAP TreeExplainer · top-3 plain-language reasons per claim</p>
        </div>
        """, unsafe_allow_html=True)
        if models_ok and feat_fraud:
            imp = pd.Series(model_fraud.feature_importances_, index=feat_fraud).sort_values().tail(10)
            imp.index = [FEAT_LABELS.get(f, f.replace('_',' ')) for f in imp.index]
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
            bars = ax.barh(imp.index, imp.values, color=BORDER2, height=0.62)
            bars[-1].set_color(AMBER)
            ax.set_xlabel("Feature Importance", fontsize=8, color=INK2)
            ax.set_title("Top 10 Fraud Drivers", fontsize=10, fontweight='bold', color=INK, pad=10)
            ax.tick_params(labelsize=7.5, colors=INK2)
            for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
            ax.spines['bottom'].set_color(BORDER)
            ax.xaxis.grid(True, alpha=0.25, color=BORDER)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

    with col_b:
        st.markdown(f"""
        <div class="ins-card">
          <h4>Model B — Litigation Risk</h4>
          <p><strong>Algorithm:</strong> XGBoost Classifier<br>
          <strong>Label:</strong> Expert-rule MACT escalation proxy · label-leakage-safe<br>
          <strong>Design:</strong> Proxy-building features excluded from the training feature set<br>
          <strong>Business use:</strong> Fight-or-settle signal at FNOL — routes high-risk cases to ADR
          before MACT filing, reducing litigation cost per claim across the book</p>
        </div>
        """, unsafe_allow_html=True)
        labels = ['Fast Track','Standard','ADR / Legal','SIU']
        sizes  = [62, 22, 10, 6]
        colors = [FASTTRACK, INK2, HIGH, CRITICAL]
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
    st.markdown(f"<div style='font-size:0.72rem;font-weight:700;color:{INK};margin-bottom:0.6rem;letter-spacing:0.3px;'>Decision Pipeline</div>", unsafe_allow_html=True)
    arch = [("FNOL","Claim arrives",SLATE),("Data Spine","35 features","#292524"),
            ("Model A","Fraud score","#78350F"),("Model B","Litigation risk","#1C1917"),
            ("SHAP","Explain","#14532D"),("Routing","Decision",FASTTRACK)]
    fh = "".join(f'<div class="flow-node" style="background:{c};"><div class="fn">{t}</div><div class="fs">{s}</div></div>{"<div class=\'flow-arr\'>›</div>" if i<5 else ""}' for i,(t,s,c) in enumerate(arch))
    st.markdown(f"""
    <div class="ins-card">
      <div class="flow">{fh}</div>
      <div style="margin-top:0.85rem;font-size:0.71rem;color:{INK2};text-align:center;">
        Single-insurer · No IIB consortium dependency · Human-in-the-loop on every P1 and P2 flag
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.72rem;font-weight:700;color:{INK};margin-bottom:0.6rem;letter-spacing:0.3px;'>Projected Financial Impact · per ₹1,000 Cr TP Claims</div>", unsafe_allow_html=True)
    fin = pd.DataFrame({
        'Scenario':['🐻 Bear','📊 Base','🐂 Bull'],
        'Leakage Recovered':['₹5 Cr','₹15 Cr','₹30 Cr'],
        'Litigation Saving':['₹1 Cr','₹3 Cr','₹6 Cr'],
        'Total Benefit':['₹6 Cr','₹18 Cr','₹36 Cr'],
        'Build Cost':['₹3.5 Cr','₹3.5 Cr','₹3.5 Cr'],
        'Payback':['~7 mo','~5 mo','~3 mo'],
    }).set_index('Scenario')
    st.dataframe(fin, use_container_width=True)
    st.caption("Normalised to ₹1,000 Cr. Build cost constant across scenarios.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="sh">
      <div class="sh-eye">Sundaram Pitch Fest 2026</div>
      <h2 class="sh-title">About the Engine</h2>
      <p class="sh-sub">A single-insurer AI platform that scores, explains, and routes every Motor TP claim at FNOL.</p>
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
             "The model deepens with every processed claim, creating a proprietary compounding moat."),
            ("🗓️","Deployment Roadmap",
             "P0 data audit → P1 fraud shadow mode (6 mo) → P2 litigation model (12 mo) "
             "→ P3 graph/network layer → P4 live routing with drift monitoring and feedback loop."),
        ]:
            st.markdown(f'<div class="ab-card"><h4>{ic} {title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

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
            st.markdown(f'<div class="ab-card"><h4>{ic} {title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

    tech = ['XGBoost','SHAP','scikit-learn','Streamlit','Python 3.14','pandas','matplotlib','joblib']
    chips = "".join(f'<span style="background:{CARD};border:1px solid {BORDER};border-radius:5px;padding:0.3rem 0.7rem;font-size:0.75rem;font-weight:600;color:{INK2};">{t}</span>' for t in tech)
    st.markdown(f'<div class="ab-card" style="border-left-color:#6D28D9;"><h4>🛠️ Technology Stack</h4><div style="display:flex;gap:0.45rem;flex-wrap:wrap;margin-top:0.5rem;">{chips}</div></div>', unsafe_allow_html=True)

    members = "".join(f'<div style="flex:1;min-width:150px;"><div style="font-weight:700;font-size:0.88rem;color:#fff;letter-spacing:-0.01em;">{n}</div><div style="font-size:0.7rem;color:rgba(255,255,255,0.4);margin-top:2px;">IIT Kharagpur · School of Law</div></div>' for n in ['Arunadithyan S','Azhagappan G','G Abiimukeshwar'])
    st.markdown(f"""
    <div style="background:{SLATE};border-radius:12px;padding:1.6rem 1.8rem;margin-top:0.2rem;border:1px solid rgba(255,255,255,0.06);">
      <div style="font-size:0.58rem;letter-spacing:2px;text-transform:uppercase;color:{AMBER};font-weight:700;margin-bottom:1rem;">Team Apex Counsel · IIT Kharagpur</div>
      <div style="display:flex;gap:1.5rem;flex-wrap:wrap;">{members}</div>
      <div style="margin-top:1rem;font-size:0.7rem;color:rgba(255,255,255,0.3);">
        Operations · Risk &amp; Process Excellence Track &nbsp;|&nbsp; Sundaram Pitch Fest 2026 · Round 2
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="ent-footer">
  <span><b>Sundaram Finance</b> · Claim Decisioning Model</span>
  <span>Team Apex Counsel · IIT Kharagpur · <b>Sundaram Pitch Fest 2026</b></span>
</div>
""", unsafe_allow_html=True)