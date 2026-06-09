

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib, json, os, re
from datetime import datetime
import io
from groq import Groq


ARTIFACT_DIR = "model_artifacts"
_required = ["model.pkl", "label_encoders.pkl", "model_metadata.pkl",
             "intent_classifier.pkl", "entity_patterns.pkl",
             "model_metrics.json", "feature_importance.json"]

if not all(os.path.exists(os.path.join(ARTIFACT_DIR, f)) for f in _required):
    with st.spinner("⚙️ First-time setup: training models... (~30–60 sec)"):
        import setup_model
        df  = setup_model.load_and_clean()
        syn = setup_model.generate_synthetic(df)
        model, encoders, fcols, metrics, imps = setup_model.train_model(syn)
        intent_pipe, entity_patterns = setup_model.train_nlp()
        setup_model.save_all(model, encoders, fcols, metrics, imps,
                             intent_pipe, entity_patterns)
    st.success("✅ Setup complete! Reloading...")
    st.rerun()



# --- STOCHASTIC PROCESSES (MARKOV CHAINS) ---
def calculate_markov_transition(user_rank, closing_rank, volatility_factor=0.08):
    """
    Calculates the probability of seat upgradation across counseling rounds
    using a Stochastic Markov Chain transition model with an exponential decay kernel.
    State 0: Waitlisted -> State 1: Allotted.
    """
    margin = closing_rank - user_rank
    
    if margin >= 5000:
        return 0.99  # Absorbing state reached (Highly Safe)
    elif margin >= 0:
        return 0.85  # High probability of retention (Moderate)
    else:
        # Calculate transition probability P(0 -> 1) for reach colleges
        distance = abs(margin)
        # The volatility factor acts as system noise/entropy in the allocation rounds
        transition_prob = np.exp(-distance / (volatility_factor * closing_rank))
        # Cap minimum probability at 0.01 (1%) for edge cases
        return max(0.01, round(transition_prob, 3))



# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CollegeSetu AI — JoSAA + CSAB Counseling DSS",
    page_icon="🎓", layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACT_DIR      = "model_artifacts"
PREDICTIVE_BUFFER = 10_000

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root{
    --bg:#0a0c0a; --bg-2:#0d100d;
    --surface:rgba(255,255,255,0.025);
    --surface-2:rgba(255,255,255,0.05);
    --border:rgba(255,255,255,0.07);
    --border-hi:rgba(163,230,53,0.38);
    --text:#e7ecef; --muted:#8b97a3; --faint:#5b6670;
    --accent:#a3e635; --accent-bright:#bef264; --accent-dim:rgba(163,230,53,0.12);
    --red:#f87171; --red-bg:rgba(248,113,113,0.12);
    --amber:#fb923c; --amber-bg:rgba(251,146,60,0.12);
    --blue:#60a5fa; --blue-bg:rgba(96,165,250,0.12);
    --green:#4ade80;
    --mono:'JetBrains Mono',ui-monospace,monospace;
    --sans:'Manrope',-apple-system,BlinkMacSystemFont,sans-serif;
}

/* ── Base ─────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"]{ font-family:var(--sans); }
[data-testid="stAppViewContainer"]{
    background:
        radial-gradient(900px 520px at 100% -6%, rgba(163,230,53,0.10), transparent 60%),
        radial-gradient(820px 600px at -8% 112%, rgba(74,222,128,0.06), transparent 55%),
        linear-gradient(180deg,#0a0c0a 0%, #0c0f0c 100%);
    background-attachment:fixed;
    color:var(--text);
    min-height:100vh;
}
[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer, .stDeployButton{ visibility:hidden; height:0; }

[data-testid="stAppViewContainer"] h1,[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,[data-testid="stAppViewContainer"] h6{
    font-family:var(--sans); letter-spacing:-0.02em; color:#fff;
}
[data-testid="stAppViewContainer"] h4{ font-weight:800; font-size:1.5rem; }
[data-testid="stAppViewContainer"] h5{ font-weight:700; font-size:1rem; color:#cdd6df; }
code, kbd{ font-family:var(--mono)!important; color:var(--accent-bright); }

/* ── Hero banner ──────────────────────────────────────────────── */
.hero{
    position:relative; overflow:hidden;
    background:linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.012));
    border:1px solid var(--border);
    padding:2.4rem 2.6rem; border-radius:20px; margin-bottom:1.7rem; color:#fff;
    box-shadow:0 30px 80px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
}
.hero::before{ content:''; position:absolute; inset:0; pointer-events:none;
    background:radial-gradient(620px 260px at 88% -45%, rgba(163,230,53,0.22), transparent 60%); }
.hero::after{ content:''; position:absolute; left:0; top:18px; bottom:18px; width:3px; border-radius:3px;
    background:linear-gradient(180deg,var(--accent),transparent); box-shadow:0 0 18px var(--accent); }
.hero>*{ position:relative; z-index:1; }
.hero h1{ font-size:2.3rem; font-weight:800; margin:.2rem 0 .55rem; color:#fff!important;
    letter-spacing:-0.03em; line-height:1.12; }
.hero p{ font-size:.95rem; line-height:1.65; margin:0; color:#aeb8c2!important; max-width:820px; }
.hero .meta{ margin-top:1.4rem; padding-top:1.3rem; border-top:1px solid var(--border);
    display:flex; gap:2.2rem; flex-wrap:wrap; font-size:.78rem; font-family:var(--mono); }
.hero .meta span{ color:var(--muted)!important; }
.hero .meta strong{ color:var(--accent)!important; font-weight:700; }
.badge{ display:inline-block; background:var(--accent-dim); color:var(--accent-bright)!important;
    padding:.34rem .9rem; border-radius:30px; font-size:.66rem; font-weight:700; letter-spacing:1.6px;
    text-transform:uppercase; margin-bottom:1rem; border:1px solid rgba(163,230,53,0.3); font-family:var(--mono); }

/* ── KPI cards ────────────────────────────────────────────────── */
.kpi{ position:relative; overflow:hidden; background:var(--surface); backdrop-filter:blur(14px);
    padding:1.4rem 1.2rem; border-radius:16px; color:#fff; text-align:left;
    border:1px solid var(--border); transition:all .22s ease; }
.kpi:hover{ transform:translateY(-3px); border-color:var(--border-hi); background:var(--surface-2); }
.kpi h3{ color:#fff!important; margin:0; font-size:2rem; font-weight:800; letter-spacing:-0.03em; font-family:var(--mono); }
.kpi p{ color:var(--muted)!important; margin:.35rem 0 0; font-size:.66rem; text-transform:uppercase; letter-spacing:1px; font-weight:700; }
.kpi-green::before,.kpi-amber::before,.kpi-red::before,.kpi-teal::before{
    content:''; position:absolute; left:0; top:0; bottom:0; width:3px; }
.kpi-green::before{ background:var(--accent); box-shadow:0 0 14px var(--accent); } .kpi-green h3{ color:var(--accent-bright)!important; }
.kpi-amber::before{ background:var(--amber); } .kpi-amber h3{ color:var(--amber)!important; }
.kpi-red::before{ background:var(--red); }     .kpi-red h3{ color:var(--red)!important; }
.kpi-teal::before{ background:var(--blue); }    .kpi-teal h3{ color:var(--blue)!important; }

/* ── Stat boxes ──────────────────────────────────────────────── */
.stat-box{ text-align:left; padding:1.1rem 1rem; border-radius:14px; background:var(--surface);
    border:1px solid var(--border); transition:all .2s ease; }
.stat-box:hover{ background:var(--surface-2); border-color:var(--border-hi); transform:translateY(-2px); }
.stat-box .val{ font-size:1.4rem; font-weight:700; color:#fff!important; letter-spacing:-0.02em; font-family:var(--mono); }
.stat-box .lbl{ font-size:.64rem; color:var(--muted)!important; text-transform:uppercase; letter-spacing:.9px; margin-top:.25rem; font-weight:700; }

/* ── Result cards (legacy) ───────────────────────────────────── */
.result-card{ background:var(--surface); border:1px solid var(--border); border-radius:14px;
    padding:1.1rem 1.4rem; margin-bottom:.7rem; transition:all .2s ease; display:flex; align-items:center; gap:1.2rem; }
.result-card:hover{ background:var(--surface-2); border-color:var(--border-hi); transform:translateX(3px); }
.result-card.safe{ border-left:3px solid var(--accent); }
.result-card.moderate{ border-left:3px solid var(--amber); }
.result-card.reach{ border-left:3px solid var(--red); }
.result-card .rank-num{ font-size:1.3rem; font-weight:800; color:var(--faint); min-width:36px; text-align:center; font-family:var(--mono); }
.result-card .info{ flex:1; }
.result-card .inst{ color:#fff!important; font-weight:700; font-size:1rem; margin:0 0 .25rem; line-height:1.3; }
.result-card .prog{ color:#bcc6d0!important; font-size:.82rem; margin:0 0 .4rem; line-height:1.35; }
.result-card .tags{ display:flex; gap:.5rem; flex-wrap:wrap; font-size:.72rem; }
.result-card .tag{ background:var(--accent-dim); color:var(--accent-bright)!important; padding:.15rem .55rem;
    border-radius:6px; font-weight:600; border:1px solid rgba(163,230,53,0.25); font-family:var(--mono); }
.result-card .ranks{ text-align:right; min-width:130px; }
.result-card .closing-rank{ color:#fff!important; font-weight:700; font-size:1.1rem; letter-spacing:-0.01em; font-family:var(--mono); }
.result-card .closing-lbl{ color:var(--muted)!important; font-size:.62rem; text-transform:uppercase; letter-spacing:.6px; }
.safety-pill{ display:inline-block; padding:.2rem .7rem; border-radius:20px; font-size:.64rem; font-weight:700;
    text-transform:uppercase; letter-spacing:.6px; margin-top:.4rem; font-family:var(--mono); }
.safety-pill.safe{ background:var(--accent-dim); color:var(--accent-bright)!important; border:1px solid rgba(163,230,53,0.3); }
.safety-pill.moderate{ background:var(--amber-bg); color:var(--amber)!important; border:1px solid rgba(251,146,60,0.3); }
.safety-pill.reach{ background:var(--red-bg); color:var(--red)!important; border:1px solid rgba(248,113,113,0.3); }

/* ── Section divider ─────────────────────────────────────────── */
.section-title{ color:var(--muted)!important; font-size:.72rem; font-weight:800; text-transform:uppercase;
    letter-spacing:1.6px; margin:1.7rem 0 .9rem; display:flex; align-items:center; gap:.7rem; font-family:var(--mono); }
.section-title::before{ content:''; width:3px; height:16px; background:var(--accent); border-radius:2px; box-shadow:0 0 10px var(--accent); }
.section-title .count{ background:var(--accent-dim); color:var(--accent-bright)!important; padding:.15rem .55rem;
    border-radius:10px; font-size:.66rem; border:1px solid rgba(163,230,53,0.25); margin-left:auto; }

/* ── NLP intent badge (chat) ─────────────────────────────────── */
.nlp-badge{ display:inline-block; padding:.22rem .65rem; border-radius:6px; font-size:.64rem; font-weight:700;
    color:#fff; letter-spacing:.5px; text-transform:uppercase; font-family:var(--mono); }

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0b0e0b 0%, #0a0c0a 100%); border-right:1px solid var(--border); }
[data-testid="stSidebar"] *{ color:var(--text)!important; }
[data-testid="stSidebar"] h2{ color:#fff!important; font-weight:800; }
[data-testid="stSidebar"] [data-testid="stMetricValue"]{ color:var(--accent-bright)!important; font-weight:800; font-family:var(--mono); }
[data-testid="stSidebar"] [data-testid="stMetricLabel"]{ color:var(--muted)!important; font-size:.66rem; text-transform:uppercase; letter-spacing:.8px; }
[data-testid="stSidebar"] hr{ border-color:var(--border); margin:1rem 0; }

/* ── Tabs ────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{ gap:6px; background:var(--surface); padding:6px; border-radius:14px; border:1px solid var(--border); }
.stTabs [data-baseweb="tab"]{ padding:10px 16px; font-weight:700; font-size:.82rem; border-radius:10px; color:var(--muted)!important; background:transparent; }
.stTabs [data-baseweb="tab"][aria-selected="true"]{ background:var(--accent)!important; color:#0a0c0a!important; box-shadow:0 6px 18px rgba(163,230,53,0.28); }
.stTabs [data-baseweb="tab"] p{ color:inherit!important; font-weight:700; }

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button{ border-radius:11px; font-weight:700; background:var(--surface); border:1px solid var(--border); color:var(--text); transition:all .18s ease; }
.stButton > button:hover{ border-color:var(--border-hi); background:var(--surface-2); }
.stButton > button[kind="primary"]{ background:var(--accent)!important; border:none!important; color:#0a0c0a!important;
    box-shadow:0 6px 20px rgba(163,230,53,0.28); font-weight:800; }
.stButton > button[kind="primary"]:hover{ transform:translateY(-1px); box-shadow:0 10px 28px rgba(163,230,53,0.42); background:var(--accent-bright)!important; }
.stDownloadButton > button{ background:var(--accent-dim)!important; border:1px solid rgba(163,230,53,0.35)!important;
    color:var(--accent-bright)!important; border-radius:11px; font-weight:700; }
.stDownloadButton > button:hover{ background:rgba(163,230,53,0.2)!important; border-color:var(--accent)!important; }

/* ── Inputs ──────────────────────────────────────────────────── */
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input{
    background:var(--surface)!important; border:1px solid var(--border)!important; color:var(--text)!important; border-radius:10px; }
[data-testid="stTextInput"] input:focus,[data-testid="stNumberInput"] input:focus{
    border-color:var(--accent)!important; box-shadow:0 0 0 2px var(--accent-dim)!important; }
[data-baseweb="select"] > div{ background:var(--surface)!important; border:1px solid var(--border)!important; border-radius:10px!important; }
[data-baseweb="select"] > div:hover{ border-color:var(--border-hi)!important; }
[data-testid="stWidgetLabel"] label, label[data-testid="stWidgetLabel"]{ color:var(--muted)!important; font-weight:600; font-size:.8rem; }
.stMultiSelect [data-baseweb="tag"]{ background:var(--accent-dim)!important; border:1px solid rgba(163,230,53,0.3)!important; }
.stMultiSelect [data-baseweb="tag"] span{ color:var(--accent-bright)!important; }

/* ── Radio (segmented) ───────────────────────────────────────── */
.stRadio [role="radiogroup"]{ gap:8px; flex-wrap:wrap; }
.stRadio [role="radiogroup"] label{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:.4rem .85rem; transition:all .18s ease; }
.stRadio [role="radiogroup"] label:hover{ border-color:var(--border-hi); }

/* ── Expander / alerts / dataframe / chat ────────────────────── */
[data-testid="stExpander"]{ border:1px solid var(--border); border-radius:14px; background:var(--surface); overflow:hidden; }
[data-testid="stExpander"] summary{ color:var(--text)!important; font-weight:600; }
[data-testid="stAlert"]{ border-radius:12px; border:1px solid var(--border); background:var(--surface); }
[data-testid="stMetricValue"]{ font-family:var(--mono); }
[data-testid="stDataFrame"]{ border:1px solid var(--border); border-radius:12px; }
[data-testid="stChatMessage"]{ background:var(--surface); border:1px solid var(--border); border-radius:14px; }
[data-testid="stChatInput"] textarea{ background:var(--surface)!important; color:var(--text)!important; }

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-track{ background:transparent; }
::-webkit-scrollbar-thumb{ background:rgba(163,230,53,0.18); border-radius:6px; }
::-webkit-scrollbar-thumb:hover{ background:rgba(163,230,53,0.32); }

/* ── Print/PDF ───────────────────────────────────────────────── */
@media print{ [data-testid="stSidebar"],[data-testid="stHeader"],button,.stButton{ display:none!important; } }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# INSTITUTE → STATE MAP  (complete, exact names from CSAB CSVs)
# ═══════════════════════════════════════════════════════════════════════════════
# ── NIT → Home State map (NITs ONLY — IIITs and GFTIs never have HS quota) ──
NIT_STATE = {
    # Every Indian state/UT mapped to its NIT exactly as named in CSAB CSV
    "National Institute of Technology, Silchar":                     "Assam",
    "National Institute of Technology, Andhra Pradesh":              "Andhra Pradesh",
    "National Institute of Technology Arunachal Pradesh":            "Arunachal Pradesh",
    "National Institute of Technology Patna":                        "Bihar",
    "National Institute of Technology Raipur":                       "Chhattisgarh",
    "National Institute of Technology Delhi":                        "Delhi",
    "National Institute of Technology Goa":                          "Goa",
    "Sardar Vallabhbhai National Institute of Technology, Surat":    "Gujarat",
    "National Institute of Technology, Kurukshetra":                 "Haryana",
    "National Institute of Technology Hamirpur":                     "Himachal Pradesh",
    "National Institute of Technology, Srinagar":                    "Jammu & Kashmir",
    "National Institute of Technology, Jamshedpur":                  "Jharkhand",
    "National Institute of Technology Karnataka, Surathkal":         "Karnataka",
    "National Institute of Technology Calicut":                      "Kerala",
    "Maulana Azad National Institute of Technology Bhopal":          "Madhya Pradesh",
    "Visvesvaraya National Institute of Technology, Nagpur":         "Maharashtra",
    "National Institute of Technology, Manipur":                     "Manipur",
    "National Institute of Technology Meghalaya":                    "Meghalaya",
    "National Institute of Technology, Mizoram":                     "Mizoram",
    "National Institute of Technology Nagaland":                     "Nagaland",
    "National Institute of Technology, Rourkela":                    "Odisha",
    "National Institute of Technology Puducherry":                   "Puducherry",
    "Dr. B R Ambedkar National Institute of Technology, Jalandhar":  "Punjab",
    "Malaviya National Institute of Technology Jaipur":              "Rajasthan",
    "National Institute of Technology Sikkim":                       "Sikkim",
    "National Institute of Technology, Tiruchirappalli":             "Tamil Nadu",
    "National Institute of Technology, Warangal":                    "Telangana",
    "National Institute of Technology  Agartala":                    "Tripura",
    "National Institute of Technology Agartala":                     "Tripura",
    "Motilal Nehru National Institute of Technology Allahabad":      "Uttar Pradesh",
    "National Institute of Technology, Uttarakhand":                 "Uttarakhand",
    "National Institute of Technology Durgapur":                     "West Bengal",
    # UTs without own NIT — mapped to nearest NIT per JOSAA rules
    # Chandigarh → NIT Jalandhar (Punjab)
    # Dadra & NH/DD → SVNIT Surat (Gujarat)
    # Lakshadweep → NIT Calicut (Kerala)
    # Andaman → NIT Durgapur (West Bengal)
    # Ladakh → NIT Srinagar (J&K)
}

# UT → effective state for NIT quota lookup
_UT_STATE = {
    "Chandigarh":                              "Punjab",
    "Ladakh":                                  "Jammu & Kashmir",
    "Lakshadweep":                             "Kerala",
    "Andaman & Nicobar Islands":               "West Bengal",
    "Andaman and Nicobar Islands":             "West Bengal",
    "Dadra & Nagar Haveli and Daman & Diu":    "Gujarat",
    "Dadra and Nagar Haveli and Daman and Diu":"Gujarat",
}

ALL_STATES = sorted({
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh",
    "Delhi","Goa","Gujarat","Haryana","Himachal Pradesh","Jammu & Kashmir",
    "Jharkhand","Karnataka","Kerala","Ladakh","Madhya Pradesh","Maharashtra",
    "Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Puducherry","Punjab",
    "Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura","Uttar Pradesh",
    "Uttarakhand","West Bengal","Chandigarh",
})


def is_eligible(institute: str, home_state: str, quota: str) -> bool:
    q = quota.strip()

    # Exclude DASA quotas (foreign-national, not for domestic counseling)
    if q in ("DASA-CIWG", "DASA-Non CIWG", "DASA CIWG", "DASA Non CIWG"):
        return False

    # All India quota — open to everyone
    if q == "All India":
        return True

    # Home State / Other State quota — ONLY applicable to NITs
    # IIITs and GFTIs never have HS/OS quota — skip those rows entirely
    is_nit = "National Institute of Technology" in institute
    if not is_nit:
        # Non-NIT with HS/OS quota row — not eligible for any student
        return False

    # Resolve UT → effective state for NIT quota purposes
    effective = _UT_STATE.get(home_state, home_state)

    # Look up which state this NIT belongs to
    nit_state = NIT_STATE.get(institute)
    if nit_state is None:
        # NIT not in our map — include conservatively
        return q in ("Home State", "Other State")

    if q == "Home State":  return effective == nit_state
    if q == "Other State": return effective != nit_state

    return False


# ─── DOMAIN CLASSIFIER ────────────────────────────────────────────────────────
def get_domain(p: str) -> str:
    p = str(p).lower()
    if any(k in p for k in ["computer","software","information","data","ai","artificial"]): return "CS_IT"
    if any(k in p for k in ["electrical","electronics","ece","eee","vlsi"]):                return "EE_ECE"
    if any(k in p for k in ["mechanical","production","industrial","auto"]):                return "Mechanical"
    if any(k in p for k in ["civil","architecture","planning","structural"]):               return "Civil"
    if any(k in p for k in ["chemical","bio","pharma","food","biotech"]):                   return "Chemical_Bio"
    if any(k in p for k in ["mining","metallur","material","ceramic"]):                     return "Materials"
    if any(k in p for k in ["math","physics","science","chemistry"]):                       return "Sciences"
    return "Other"


def get_itype(name: str) -> str:
    name = str(name)
    if "National Institute of Technology" in name: return "NIT"
    if "Indian Institute of Information"  in name or "IIIT" in name: return "IIIT"
    return "GFTI"


# ─── BRANCH CATEGORY CLASSIFIER (Tech / Core / Other) ─────────────────────────
# Used by the Tab-1 branch filter:
#   "Tech"  → CSE, IT, ECE, EE, AI/ML, Data Science, MnC, VLSI, Cyber, Robotics …
#   "Core"  → Civil, Mechanical, Chemical, Metallurgy, Materials, Mining, Aero …
#   "Other" → pure sciences / design / generic B.Tech (only shown under "All")
_TECH_KW = [
    "computer", "computing", "computation", "software", "informatics",
    "information technology", "information  technology", "information science",
    "electronic", "communication", "telecommunication", "electrical",
    "vlsi", "microelectronic", "integrated circuit",
    "artificial intelligence", "artificial inelligence", "machine learning",
    "data science", "data analytics", "data engineering",
    "scientific computing", "ai and", "and ai", "ai &", "& ai", "in ai", "robotics and ai",
    "cyber", "robotic", "mechatron", "automation", "internet of things",
    "instrumentation", "statistics and data", "quantitative economics & data",
]
_CORE_KW = [
    "civil", "mechanical", "chemical", "metallurg", "material", "mining",
    "production", "industrial", "manufactur", "aerospace", "aeronaut", "aviation",
    "automobile", "automotive", "marine", "naval", "ocean", "textile", "fibre", "fiber",
    "carpet", "handloom", "ceramic", "agricul", "petroleum", "petro", "polymer", "plastic",
    "rubber", "paint", "food", "dairy", "leather", "biotech", "bio technology", "biomedical",
    "bio medical", "biochemical", "bioengineering", "bio engineering", "biological",
    "bioscience", "bioinformatics", "environment", "energy", "thermal", "structural",
    "construction", "geotechnical", "geolog", "geophys", "earth science", "mineral",
    "space science", "printing", "packaging",
]
_OTHER_KW = ["bachelor of design", "animation", "vfx", " mba ", " bba", "bachelor of business"]


def get_branch_category(program: str) -> str:
    p = str(program).lower()
    if any(k in p for k in _OTHER_KW):
        return "Other"
    if any(k in p for k in _TECH_KW):
        return "Tech"
    if any(k in p for k in _CORE_KW):
        return "Core"
    return "Other"


# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data
def load_csab() -> pd.DataFrame:
    """
    Load and merge all 3 CSAB rounds from csab_cutoffs.csv.
    Expects columns: Institute, Program, Quota, Category, Gender,
                     Opening Rank, Closing Rank, Closed_In_Round, Institute_Type
    Falls back to legacy single-file if merged file missing.
    """
    path = "csab_cutoffs.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Normalize column names
    rn = {}
    if "Academic Program Name" in df.columns: rn["Academic Program Name"] = "Program"
    if "Seat Type"             in df.columns: rn["Seat Type"]             = "Category"
    if "Institute Type"        in df.columns: rn["Institute Type"]        = "Institute_Type"
    if "Opening Rank (int)"    in df.columns: rn["Opening Rank (int)"]    = "Opening Rank"
    if "Closing Rank (int)"    in df.columns: rn["Closing Rank (int)"]    = "Closing Rank"
    df = df.rename(columns=rn)

    if "Program" not in df.columns and "Academic Program Name" in df.columns:
        df["Program"] = df["Academic Program Name"]

    # Drop duplicate rank columns if present
    for extra in ["Opening Rank.1", "Closing Rank.1"]:
        if extra in df.columns: df.drop(columns=[extra], inplace=True)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    for rc in ["Opening Rank", "Closing Rank"]:
        df[rc] = pd.to_numeric(df[rc], errors="coerce")
    df.dropna(subset=["Opening Rank","Closing Rank"], inplace=True)
    df["Opening Rank"] = df["Opening Rank"].astype(int)
    df["Closing Rank"] = df["Closing Rank"].astype(int)

    if "Institute_Type" not in df.columns:
        df["Institute_Type"] = df["Institute"].apply(get_itype)
    if "Closed_In_Round" not in df.columns:
        df["Closed_In_Round"] = 3

    # Remove DASA-CIWG and DASA-Non CIWG seats (foreign-national quotas)
    if "Quota" in df.columns:
        df = df[~df["Quota"].isin(["DASA-CIWG", "DASA-Non CIWG", "DASA CIWG", "DASA Non CIWG"])].copy()

    # --- NEW: Remove Architecture and Planning Courses ---
    if "Program" in df.columns:
        df = df[~df["Program"].str.contains('Architecture|Planning|B.Arch|B.Plan', case=False, na=False)].copy()

    df["Program_Domain"] = df["Program"].apply(get_domain)
    return df


@st.cache_data
def load_uptac() -> pd.DataFrame:
    path = "uptac_cutoffs.csv"
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    
    rn = {}
    if "Seat"               in df.columns: rn["Seat"]               = "Category"
    if "Opening Rank (int)" in df.columns: rn["Opening Rank (int)"] = "Opening Rank"
    if "Closing Rank (int)" in df.columns: rn["Closing Rank (int)"] = "Closing Rank"
    if "Course"             in df.columns: rn["Course"]             = "Program" # Sometimes UPTAC uses Course
    df = df.rename(columns=rn)
    
    if "Quota" in df.columns:
        df["Quota"] = df["Quota"].replace({"HS":"Home State","AI":"All India"})
        
    for rc in ["Opening Rank","Closing Rank"]:
        df[rc] = pd.to_numeric(df[rc], errors="coerce")
    df.dropna(subset=["Opening Rank","Closing Rank"], inplace=True)
    df["Opening Rank"] = df["Opening Rank"].astype(int)
    df["Closing Rank"] = df["Closing Rank"].astype(int)

    # --- NEW: Remove Architecture and Planning Courses ---
    if "Program" in df.columns:
        df = df[~df["Program"].str.contains('Architecture|Planning|B.Arch|B.Plan', case=False, na=False)].copy()

    return df


@st.cache_data
def load_iit() -> pd.DataFrame:
    path = "iit_cutoffs.csv"
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    
    rn = {}
    if "Academic Program Name" in df.columns: rn["Academic Program Name"] = "Program"
    if "Seat Type"             in df.columns: rn["Seat Type"]             = "Category"
    df = df.rename(columns=rn)
    
    for rc in ["Opening Rank","Closing Rank"]:
        df[rc] = pd.to_numeric(df[rc], errors="coerce")
    df.dropna(subset=["Opening Rank","Closing Rank"], inplace=True)
    df["Opening Rank"] = df["Opening Rank"].astype(int)
    df["Closing Rank"] = df["Closing Rank"].astype(int)
    
    # Remove DASA-CIWG and DASA-Non CIWG seats from IIT data
    if "Quota" in df.columns:
        df = df[~df["Quota"].isin(["DASA-CIWG", "DASA-Non CIWG", "DASA CIWG", "DASA Non CIWG"])].copy()

    # --- NEW: Remove Architecture and Planning Courses ---
    if "Program" in df.columns:
        df = df[~df["Program"].str.contains('Architecture|Planning|B.Arch|B.Plan', case=False, na=False)].copy()

    df["Program_Domain"] = df["Program"].apply(get_domain)
    return df


# --- Execute Loaders ---
csab_df  = load_csab()
uptac_df = load_uptac()
iit_df   = load_iit()

# ─── MODEL LOADING ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    model    = joblib.load(os.path.join(ARTIFACT_DIR,"model.pkl"))
    encoders = joblib.load(os.path.join(ARTIFACT_DIR,"label_encoders.pkl"))
    metadata = joblib.load(os.path.join(ARTIFACT_DIR,"model_metadata.pkl"))
    return model, encoders, metadata

@st.cache_data
def load_metrics():
    p = os.path.join(ARTIFACT_DIR,"model_metrics.json")
    return json.load(open(p)) if os.path.exists(p) else {}

@st.cache_resource
def load_nlp():
    return (joblib.load(os.path.join(ARTIFACT_DIR,"intent_classifier.pkl")),
            joblib.load(os.path.join(ARTIFACT_DIR,"entity_patterns.pkl")))

model, encoders, model_meta = load_model_artifacts()
ml_metrics = load_metrics()
intent_clf, entity_patterns = load_nlp()


def safe_enc(col, val):
    le = encoders.get(col)
    if le is None: return 0
    return int(le.transform([val])[0]) if val in le.classes_ else 0


# ─── COLLEGE DATABASE ─────────────────────────────────────────────────────────
@st.cache_data
def get_college_db():
    db = {
        "National Institute of Technology Karnataka, Surathkal": {
            "short":"NIT Surathkal","est":1960,"nirf_rank":12,"campus_acres":295,
            "city":"Mangalore, Karnataka","highest_ctc":"1.27 Cr","avg_ctc":"21.2 LPA",
            "median_ctc":"18.5 LPA","lowest_ctc":"6 LPA","placement_pct":92,
            "top_recruiters":"Google, Microsoft, Amazon, Goldman Sachs, Qualcomm","hostel":"Yes"},
        "National Institute of Technology, Tiruchirappalli": {
            "short":"NIT Trichy","est":1964,"nirf_rank":9,"campus_acres":800,
            "city":"Tiruchirappalli, Tamil Nadu","highest_ctc":"1.2 Cr","avg_ctc":"19.8 LPA",
            "median_ctc":"16.5 LPA","lowest_ctc":"5.5 LPA","placement_pct":95,
            "top_recruiters":"Google, Microsoft, TCS, Infosys, Samsung","hostel":"Yes"},
        "National Institute of Technology, Warangal": {
            "short":"NIT Warangal","est":1959,"nirf_rank":13,"campus_acres":260,
            "city":"Warangal, Telangana","highest_ctc":"1.1 Cr","avg_ctc":"18.5 LPA",
            "median_ctc":"16 LPA","lowest_ctc":"5 LPA","placement_pct":90,
            "top_recruiters":"Microsoft, Amazon, Google, Flipkart, Goldman Sachs","hostel":"Yes"},
        "National Institute of Technology, Rourkela": {
            "short":"NIT Rourkela","est":1961,"nirf_rank":16,"campus_acres":640,
            "city":"Rourkela, Odisha","highest_ctc":"90 LPA","avg_ctc":"15.6 LPA",
            "median_ctc":"14 LPA","lowest_ctc":"5 LPA","placement_pct":88,
            "top_recruiters":"Amazon, Microsoft, TCS, Tata Steel, Goldman Sachs","hostel":"Yes"},
        "Sardar Vallabhbhai National Institute of Technology, Surat": {
            "short":"SVNIT Surat","est":1961,"nirf_rank":25,"campus_acres":130,
            "city":"Surat, Gujarat","highest_ctc":"65 LPA","avg_ctc":"13.5 LPA",
            "median_ctc":"12 LPA","lowest_ctc":"4.5 LPA","placement_pct":85,
            "top_recruiters":"TCS, Infosys, L&T, Amazon, Reliance","hostel":"Yes"},
        "Visvesvaraya National Institute of Technology, Nagpur": {
            "short":"VNIT Nagpur","est":1960,"nirf_rank":19,"campus_acres":200,
            "city":"Nagpur, Maharashtra","highest_ctc":"80 LPA","avg_ctc":"15 LPA",
            "median_ctc":"14 LPA","lowest_ctc":"4.5 LPA","placement_pct":87,
            "top_recruiters":"Google, Microsoft, Amazon, TCS, Wipro","hostel":"Yes"},
        "Malaviya National Institute of Technology Jaipur": {
            "short":"MNIT Jaipur","est":1963,"nirf_rank":20,"campus_acres":317,
            "city":"Jaipur, Rajasthan","highest_ctc":"75 LPA","avg_ctc":"14.2 LPA",
            "median_ctc":"13 LPA","lowest_ctc":"4.5 LPA","placement_pct":86,
            "top_recruiters":"Amazon, Microsoft, Samsung, TCS, Infosys","hostel":"Yes"},
        "Maulana Azad National Institute of Technology Bhopal": {
            "short":"MANIT Bhopal","est":1960,"nirf_rank":28,"campus_acres":650,
            "city":"Bhopal, Madhya Pradesh","highest_ctc":"52 LPA","avg_ctc":"11.5 LPA",
            "median_ctc":"10 LPA","lowest_ctc":"4 LPA","placement_pct":82,
            "top_recruiters":"TCS, Infosys, Cognizant, Capgemini, L&T","hostel":"Yes"},
        "Motilal Nehru National Institute of Technology Allahabad": {
            "short":"MNNIT Allahabad","est":1961,"nirf_rank":22,"campus_acres":222,
            "city":"Prayagraj, Uttar Pradesh","highest_ctc":"1 Cr","avg_ctc":"15 LPA",
            "median_ctc":"14 LPA","lowest_ctc":"5 LPA","placement_pct":88,
            "top_recruiters":"Microsoft, Amazon, Samsung, Goldman Sachs, Flipkart","hostel":"Yes"},
        "Dr. B R Ambedkar National Institute of Technology, Jalandhar": {
            "short":"NIT Jalandhar","est":1987,"nirf_rank":48,"campus_acres":180,
            "city":"Jalandhar, Punjab","highest_ctc":"44 LPA","avg_ctc":"9.5 LPA",
            "median_ctc":"8 LPA","lowest_ctc":"3.6 LPA","placement_pct":78,
            "top_recruiters":"TCS, Infosys, Wipro, HCL, Cognizant","hostel":"Yes"},
        "National Institute of Technology Calicut": {
            "short":"NIT Calicut","est":1961,"nirf_rank":15,"campus_acres":250,
            "city":"Kozhikode, Kerala","highest_ctc":"1 Cr","avg_ctc":"16.5 LPA",
            "median_ctc":"15 LPA","lowest_ctc":"5 LPA","placement_pct":90,
            "top_recruiters":"Google, Microsoft, Amazon, Samsung, Oracle","hostel":"Yes"},
        "National Institute of Technology Durgapur": {
            "short":"NIT Durgapur","est":1960,"nirf_rank":35,"campus_acres":186,
            "city":"Durgapur, West Bengal","highest_ctc":"54 LPA","avg_ctc":"11.2 LPA",
            "median_ctc":"10 LPA","lowest_ctc":"4 LPA","placement_pct":83,
            "top_recruiters":"TCS, Amazon, Infosys, Cognizant, Wipro","hostel":"Yes"},
        "National Institute of Technology Patna": {
            "short":"NIT Patna","est":2004,"nirf_rank":40,"campus_acres":125,
            "city":"Patna, Bihar","highest_ctc":"60 LPA","avg_ctc":"11 LPA",
            "median_ctc":"10 LPA","lowest_ctc":"4 LPA","placement_pct":80,
            "top_recruiters":"Microsoft, Amazon, Samsung, TCS, Flipkart","hostel":"Yes"},
        "National Institute of Technology, Silchar": {
            "short":"NIT Silchar","est":1967,"nirf_rank":50,"campus_acres":627,
            "city":"Silchar, Assam","highest_ctc":"48 LPA","avg_ctc":"9 LPA",
            "median_ctc":"8 LPA","lowest_ctc":"3.5 LPA","placement_pct":75,
            "top_recruiters":"Amazon, TCS, Infosys, Cognizant, ONGC","hostel":"Yes"},
        "National Institute of Technology, Kurukshetra": {
            "short":"NIT Kurukshetra","est":1963,"nirf_rank":38,"campus_acres":290,
            "city":"Kurukshetra, Haryana","highest_ctc":"50 LPA","avg_ctc":"11 LPA",
            "median_ctc":"10 LPA","lowest_ctc":"4 LPA","placement_pct":82,
            "top_recruiters":"Samsung, Amazon, TCS, Infosys, Maruti","hostel":"Yes"},
        "National Institute of Technology Hamirpur": {
            "short":"NIT Hamirpur","est":1986,"nirf_rank":55,"campus_acres":320,
            "city":"Hamirpur, Himachal Pradesh","highest_ctc":"44 LPA","avg_ctc":"9 LPA",
            "median_ctc":"8 LPA","lowest_ctc":"3.5 LPA","placement_pct":75,
            "top_recruiters":"TCS, Infosys, Wipro, HCL, Samsung","hostel":"Yes"},
        "National Institute of Technology Raipur": {
            "short":"NIT Raipur","est":1956,"nirf_rank":45,"campus_acres":100,
            "city":"Raipur, Chhattisgarh","highest_ctc":"40 LPA","avg_ctc":"9 LPA",
            "median_ctc":"8 LPA","lowest_ctc":"3.5 LPA","placement_pct":76,
            "top_recruiters":"TCS, Infosys, Cognizant, Capgemini, BHEL","hostel":"Yes"},
        "Indian Institute of Information Technology, Allahabad": {
            "short":"IIIT Allahabad","est":1999,"nirf_rank":30,"campus_acres":200,
            "city":"Prayagraj, UP","highest_ctc":"1.2 Cr","avg_ctc":"20 LPA",
            "median_ctc":"18 LPA","lowest_ctc":"6 LPA","placement_pct":92,
            "top_recruiters":"Google, Microsoft, Amazon, Uber, Goldman Sachs","hostel":"Yes"},
        "Punjab Engineering College, Chandigarh": {
            "short":"PEC Chandigarh","est":1921,"nirf_rank":56,"campus_acres":145,
            "city":"Chandigarh","highest_ctc":"52 LPA","avg_ctc":"12.5 LPA",
            "median_ctc":"11 LPA","lowest_ctc":"4.5 LPA","placement_pct":84,
            "top_recruiters":"Google, Amazon, Samsung, TCS, Infosys","hostel":"Yes"},
        "Indian Institute of Engineering Science and Technology, Shibpur": {
            "short":"IIEST Shibpur","est":1856,"nirf_rank":24,"campus_acres":86,
            "city":"Howrah, West Bengal","highest_ctc":"70 LPA","avg_ctc":"13 LPA",
            "median_ctc":"12 LPA","lowest_ctc":"4.5 LPA","placement_pct":87,
            "top_recruiters":"Google, Microsoft, Amazon, Goldman Sachs, Samsung","hostel":"Yes"},
    }
    for inst in csab_df["Institute"].unique():
        if inst not in db:
            db[inst] = {"short":str(inst).split(",")[0][:30],"est":"—","nirf_rank":"—",
                        "campus_acres":"—","city":"—","highest_ctc":"—","avg_ctc":"—",
                        "median_ctc":"—","lowest_ctc":"—","placement_pct":"—",
                        "top_recruiters":"—","hostel":"—"}
    return db

college_db = get_college_db()


# ─── REUSABLE PDF GENERATOR ───────────────────────────────────────────────────
def generate_results_pdf(
    rdf: pd.DataFrame,
    report_title: str,
    profile: dict,           # e.g. {"Rank": 50000, "Category": "OPEN", "Gender": "Gender-Neutral", "Home State": "Assam"}
    summary_counts: dict,    # e.g. {"Total": 12, "Safe": 5, "Moderate": 4, "Reach": 3}
    table_columns: list,     # list of column names to include in PDF
) -> bytes:
    """
    Generates a professionally styled PDF report for any tab's results.
    Returns raw PDF bytes ready for st.download_button.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable)
    from reportlab.lib.enums import TA_CENTER
    import io

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=15*mm, rightMargin=15*mm,
                            topMargin=14*mm, bottomMargin=14*mm)

    # Color palette
    DARK   = colors.HexColor("#0a0c0a")
    PURPLE = colors.HexColor("#1a2e05")
    INDIGO = colors.HexColor("#4d7c0f")
    VIOLET = colors.HexColor("#65a30d")
    GREEN  = colors.HexColor("#4d7c0f")
    AMBER  = colors.HexColor("#c2620c")
    RED    = colors.HexColor("#dc2626")
    LIGHT  = colors.HexColor("#f7fee7")
    GREY   = colors.HexColor("#64748b")
    WHITE  = colors.white

    styles = getSampleStyleSheet()
    def sty(name="Normal", **kw):
        return ParagraphStyle(name, parent=styles[name], **kw)

    title_sty   = sty(fontSize=18, fontName="Helvetica-Bold", textColor=WHITE, spaceAfter=2)
    sub_sty     = sty(fontSize=9,  fontName="Helvetica",       textColor=colors.HexColor("#c4b5fd"), spaceAfter=0)
    badge_sty   = sty(fontSize=8,  fontName="Helvetica-Bold", textColor=WHITE, spaceAfter=6)
    section_sty = sty(fontSize=12, fontName="Helvetica-Bold", textColor=PURPLE, spaceBefore=10, spaceAfter=4)
    small_sty   = sty(fontSize=7,  fontName="Helvetica", textColor=GREY)
    cell_sty    = sty(fontSize=7.5, fontName="Helvetica", textColor=DARK, wordWrap="CJK", leading=10)
    cell_bold   = sty(fontSize=7.5, fontName="Helvetica-Bold", textColor=DARK, wordWrap="CJK", leading=10)

    ts = datetime.now().strftime("%d %b %Y, %I:%M %p")
    story = []

    # ── Header ──────────────────────────────────────────────────────────
    hdr_data = [
        [Paragraph("AVRICUS AI · COUNSELING REPORT", badge_sty), ""],
        [Paragraph(report_title, title_sty), ""],
        [Paragraph(f"Generated on {ts}", sub_sty), ""],
    ]
    hdr_tbl = Table(hdr_data, colWidths=["*", 30*mm])
    hdr_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), PURPLE),
        ("TOPPADDING",   (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",(0,-1),(-1,-1), 14),
        ("LEFTPADDING",  (0,0), (-1,-1), 14),
        ("RIGHTPADDING", (0,0), (-1,-1), 14),
    ]))
    story.append(hdr_tbl)
    story.append(Spacer(1, 6*mm))

    # ── Student profile ─────────────────────────────────────────────────
    if profile:
        story.append(Paragraph("Student Profile", section_sty))
        story.append(HRFlowable(width="100%", thickness=1.5, color=INDIGO, spaceAfter=4))
        prof_items = list(profile.items())
        # Build profile as 2-col layout (label | value | label | value)
        rows = []
        for i in range(0, len(prof_items), 2):
            row = [
                Paragraph(f"<b>{prof_items[i][0]}</b>", cell_bold),
                Paragraph(str(prof_items[i][1]), cell_sty),
            ]
            if i+1 < len(prof_items):
                row += [
                    Paragraph(f"<b>{prof_items[i+1][0]}</b>", cell_bold),
                    Paragraph(str(prof_items[i+1][1]), cell_sty),
                ]
            else:
                row += [Paragraph("", cell_sty), Paragraph("", cell_sty)]
            rows.append(row)
        prof_tbl = Table(rows, colWidths=[35*mm, 55*mm, 35*mm, 45*mm])
        prof_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), LIGHT),
            ("TOPPADDING",    (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("RIGHTPADDING",  (0,0), (-1,-1), 8),
            ("LINEABOVE",     (0,0), (-1,0),  2, INDIGO),
            ("ROWBACKGROUNDS",(0,0), (-1,-1), [LIGHT, colors.HexColor("#ede9fe")]),
        ]))
        story.append(prof_tbl)
        story.append(Spacer(1, 5*mm))

    # ── Summary ─────────────────────────────────────────────────────────
    if summary_counts:
        story.append(Paragraph("Summary", section_sty))
        story.append(HRFlowable(width="100%", thickness=1.5, color=INDIGO, spaceAfter=4))
        sum_cells = []
        color_map = {"Total": PURPLE, "Safe": colors.HexColor("#059669"),
                     "Moderate": colors.HexColor("#d97706"), "Reach": colors.HexColor("#dc2626")}
        for label, val in summary_counts.items():
            sum_cells.append(Paragraph(f"<b>{val}</b><br/>{label}", small_sty))
        sum_tbl = Table([sum_cells], colWidths=["*"]*len(sum_cells))
        ts_styles = [
            ("TEXTCOLOR",     (0,0), (-1,-1), WHITE),
            ("FONTNAME",      (0,0), (-1,-1), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("INNERGRID",     (0,0), (-1,-1), 1, WHITE),
        ]
        for i, label in enumerate(summary_counts.keys()):
            ts_styles.append(("BACKGROUND", (i,0), (i,0), color_map.get(label, PURPLE)))
        sum_tbl.setStyle(TableStyle(ts_styles))
        story.append(sum_tbl)
        story.append(Spacer(1, 5*mm))

    # ── Results table ───────────────────────────────────────────────────
    story.append(Paragraph(f"Matching Colleges ({len(rdf)} found)", section_sty))
    story.append(HRFlowable(width="100%", thickness=1.5, color=INDIGO, spaceAfter=4))

    # Build header row
    thead = [Paragraph(f"<b>#</b>", cell_bold)] + [
        Paragraph(f"<b>{c}</b>", cell_bold) for c in table_columns
    ]
    tdata = [thead]
    safety_colors = {"Safe": GREEN, "Moderate": AMBER, "Reach": RED}

    for i, (_, row) in enumerate(rdf.iterrows(), 1):
        line = [Paragraph(str(i), cell_sty)]
        for c in table_columns:
            val = row.get(c, "—")
            if c == "Safety":
                clean = str(val).replace("🟢 ","").replace("🟡 ","").replace("🔴 ","").strip()
                sc = safety_colors.get(clean, GREY)
                line.append(Paragraph(
                    f"<font color='#{sc.hexval()[2:]}'><b>{clean}</b></font>", cell_sty))
            elif "Rank" in c and isinstance(val, (int, float)):
                line.append(Paragraph(f"{int(val):,}", cell_sty))
            elif c == "Margin" and isinstance(val, (int, float)):
                line.append(Paragraph(f"{int(val):+,}", cell_sty))
            else:
                line.append(Paragraph(str(val)[:70], cell_sty))
        tdata.append(line)

    # Distribute column widths
    n_cols = len(table_columns) + 1
    if n_cols <= 6:
        col_w = [8*mm] + [(180/(n_cols-1))*mm] * (n_cols-1)
    else:
        # Wider Institute/Program columns
        col_w = [8*mm, 50*mm, 45*mm] + [(85/(n_cols-3))*mm] * (n_cols-3)
        col_w = col_w[:n_cols]

    res_tbl = Table(tdata, colWidths=col_w, repeatRows=1)
    res_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), PURPLE),
        ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 4),
        ("RIGHTPADDING",  (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#faf5ff")]),
        ("INNERGRID",     (0,0), (-1,-1), 0.4, colors.HexColor("#e9d5ff")),
        ("BOX",           (0,0), (-1,-1), 0.8, colors.HexColor("#c4b5fd")),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
    ]))
    story.append(res_tbl)
    story.append(Spacer(1, 6*mm))

    # ── Footer ──────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#e2e8f0"), spaceAfter=4))
    story.append(Paragraph(
        "This report is for guidance only. Always verify with official JoSAA / CSAB / state counselling "
        "portals before making final choices.  ©  2026 Avricus AI.",
        small_sty))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ─── NLP ──────────────────────────────────────────────────────────────────────
INTENT_COLORS = {
    "college_recommendation":"#6366f1","branch_comparison":"#ec4899",
    "cutoff_inquiry":"#06b6d4","counseling_process":"#10b981",
    "career_guidance":"#f43f5e","category_reservation":"#eab308",
    "admission_probability":"#8b5cf6","greeting":"#f97316",
}

def classify_intent(text):
    intent     = intent_clf.predict([text])[0]
    confidence = max(intent_clf.predict_proba([text])[0])
    entities   = {}
    rm = re.findall(r'\b(\d{3,7})\b', text)
    if rm: entities["rank"] = int(rm[0])
    tl = text.lower()
    for cat, kws in entity_patterns["category_keywords"].items():
        if any(k in tl for k in kws): entities["category"] = cat; break
    for gen, kws in entity_patterns["gender_keywords"].items():
        if any(k in tl for k in kws): entities["gender"] = gen; break
    return intent, confidence, entities


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 CollegeSetu")
    st.caption("powered by Avricus AI · v4.1")
    st.divider()

    st.markdown("##### 📊 Platform Coverage")
    c1, c2 = st.columns(2)
    c1.metric("Cutoffs", f"{len(csab_df):,}")
    c2.metric("Institutes", csab_df["Institute"].nunique())
    st.divider()

    st.markdown("##### 🧭 Modules")
    st.caption("📋  JoSAA + CSAB Allocation")
    st.caption("🏛️  IIT Cutoffs")
    st.caption("🗺️  State Counselling")
    st.caption("📈  Admission Predictor")
    st.caption("💬  AI Counselor")
    st.caption("⚖️  Compare Colleges")
    st.divider()

    st.markdown("##### 🌐 Need Help?")
    st.caption("Use the **AI Counselor** tab for personalised guidance, or visit the official portals for verification.")
    st.divider()

    st.caption("© 2026 CollegeSetu · All rights reserved.")


# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="badge">✨ Avricus AI · Smart Counseling Platform</div>
    <h1>Find Your Perfect Engineering College</h1>
    <p>Personalised admission guidance powered by intelligent cutoff analysis across
    <strong style="color:#fff;">JoSAA + CSAB</strong>, <strong style="color:#fff;">IITs</strong>, and
    <strong style="color:#fff;">state counselling</strong> — with an AI counselor ready to answer your questions.</p>
    <div class="meta">
        <span><strong>{len(csab_df):,}</strong> cutoffs indexed</span>
        <span><strong>{csab_df['Institute'].nunique()}</strong> institutes</span>
        <span><strong>{len(iit_df):,}</strong> IIT records</span>
        <span><strong>3</strong> counselling systems</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋  JoSAA + CSAB Allocation",
    "🏛️  IITs",
    "🗺️  State Counselling",
    "📈  ML Forecasting",
    "💬  AI Counselor",
    "⚖️  College Comparison",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — CSAB ALLOCATION ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown("#### JoSAA + CSAB Allocation Engine")
    st.caption(f"**{len(csab_df):,} cutoff records** across **{csab_df['Institute'].nunique()} institutes** — "
               f"JoSAA + CSAB Rounds 1, 2 & 3 intelligently merged. Round 1 closings preserved.")

    with st.expander("ℹ️ How JoSAA + CSAB quota logic works", expanded=False):
        st.markdown("""
        **Home State (HS) quota** — for students whose home state matches the NIT's state.
        Example: Assam student → eligible for **NIT Silchar Home State quota**

        **Other State (OS) quota** — for students from all other states.

        **All India (AI) quota** — open to all students regardless of state.

        The predictor automatically applies the correct quota based on your home state.
        """)

    c1, c2, c3, c4 = st.columns(4)
    with c1: rank   = st.number_input("JEE Main Rank", 1, 1_500_000, 50_000, 500, key="c1_rank")
    with c2:
        state = st.selectbox("Home State", ALL_STATES, key="c1_state",
                             index=sorted(ALL_STATES).index("Assam") if "Assam" in ALL_STATES else 0)
    with c3: cat    = st.selectbox("Category", sorted(csab_df["Category"].unique()), key="c1_cat")
    with c4: gender = st.selectbox("Gender",   sorted(csab_df["Gender"].unique()),   key="c1_gen")

    itype_filter = st.multiselect(
        "Institute Type", ["NIT","IIIT","GFTI"],
        default=["NIT","IIIT","GFTI"], key="c1_itype")
    branch_choice = st.radio(
        "Branch Category", ["All", "Tech", "Core"],
        index=0, horizontal=True, key="c1_branch",
        help="Tech → CSE / IT / ECE / EE / AI-ML / Data Science / MnC / VLSI / Cyber / Robotics.   "
             "Core → Civil / Mechanical / Chemical / Metallurgy / Materials / Mining / Aerospace.   "
             "All → every branch.")
    use_buffer = st.checkbox(
        f"Apply predictive buffer (+{PREDICTIVE_BUFFER:,} ranks for year-over-year seat expansion)",
        value=True, key="c1_buf")

    if st.button("📋 Find JoSAA + CSAB Options", type="primary", key="c1_btn"):
        buf = PREDICTIVE_BUFFER if use_buffer else 0
        results = []
        for _, row in csab_df.iterrows():
            if row.get("Institute_Type","GFTI") not in itype_filter: continue
            if row["Category"] != cat:    continue
            if row["Gender"]   != gender: continue
            if branch_choice != "All" and get_branch_category(row["Program"]) != branch_choice: continue
            if not is_eligible(row["Institute"], state, row["Quota"]): continue
            if rank > row["Closing Rank"] + buf: continue

            margin = row["Closing Rank"] - rank
            safety = "Safe" if margin >= 5000 else ("Moderate" if margin >= 0 else "Reach")
            info   = college_db.get(row["Institute"], {})
            
            # ─── ECE STOCHASTIC CALCULATION (MARKOV KERNEL) ───
            markov_prob = calculate_markov_transition(rank, row["Closing Rank"])
            
            results.append({
                "Institute":      row["Institute"],
                "Program":        row["Program"],
                "Quota":          row["Quota"],
                "Category":       cat,
                "Closing Rank":   row["Closing Rank"],
                "Margin":         margin,
                "Safety":         safety,
                "Markov_Prob":    markov_prob, 
                "Round Closed":   int(row.get("Closed_In_Round", 3)),
                "NIRF":           info.get("nirf_rank","—"),
                "Avg CTC":        info.get("avg_ctc","—"),
            })

        if results:
            rdf = pd.DataFrame(results).sort_values("Closing Rank").reset_index(drop=True)

            s = sum(1 for r in results if r["Safety"] == "Safe")
            m = sum(1 for r in results if r["Safety"] == "Moderate")
            r = sum(1 for r in results if r["Safety"] == "Reach")

            # ── Summary KPIs ─────────────────────────────────────────────
            st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f'<div class="kpi"><h3>{len(results)}</h3><p>Total Options</p></div>',       unsafe_allow_html=True)
            k2.markdown(f'<div class="kpi kpi-green"><h3>{s}</h3><p>🟢 Safe</p></div>',             unsafe_allow_html=True)
            k3.markdown(f'<div class="kpi kpi-amber"><h3>{m}</h3><p>🟡 Moderate</p></div>',         unsafe_allow_html=True)
            k4.markdown(f'<div class="kpi kpi-red"><h3>{r}</h3><p>🔴 Reach</p></div>',              unsafe_allow_html=True)

            # ── Top picks (cards) ───────────────────────────────────────
            top_n = min(10, len(rdf))
            st.markdown(f'<div class="section-title">Top Recommendations <span class="count">{top_n} of {len(rdf)}</span></div>',
                        unsafe_allow_html=True)

            # ── Inject Compact UI CSS ──
            st.markdown("""
            <style>
            .compact-card {
                display: flex;
                align-items: center;
                background: rgba(255, 255, 255, 0.025);
                border: 1px solid rgba(255, 255, 255, 0.07);
                border-radius: 12px;
                padding: 13px 18px;
                margin-bottom: 10px;
                gap: 16px;
                transition: all 0.2s ease;
            }
            .compact-card:hover { background: rgba(255, 255, 255, 0.05); border-color: rgba(163,230,53,0.38); transform: translateX(3px); }
            .compact-card.safe { border-left: 3px solid #a3e635; }
            .compact-card.moderate { border-left: 3px solid #fb923c; }
            .compact-card.reach { border-left: 3px solid #f87171; }

            .cc-rank { font-size: 1.1rem; font-weight: 800; color: #5b6670; min-width: 30px; font-family:'JetBrains Mono',monospace; }
            .cc-info { flex-grow: 1; display: flex; flex-direction: column; gap: 3px; }

            /* Forces the full name to display and wrap beautifully */
            .cc-inst {
                font-size: 1rem; font-weight: 700; color: #ffffff;
                line-height: 1.3; white-space: normal; word-wrap: break-word;
            }
            .cc-prog { font-size: 0.85rem; color: #9aa6b1; line-height: 1.25; }

            .cc-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 5px; }
            .cc-tag { font-size: 0.68rem; padding: 2px 7px; border-radius: 6px; background: rgba(163,230,53,0.12); color: #bef264; border:1px solid rgba(163,230,53,0.25); font-family:'JetBrains Mono',monospace; }

            .cc-stats { text-align: right; min-width: 95px; }
            .cc-close { font-size: 1.1rem; font-weight: 700; color: #ffffff; font-family:'JetBrains Mono',monospace; }
            .cc-close-lbl { font-size: 0.62rem; color: #8b97a3; text-transform: uppercase; letter-spacing:.5px; margin-bottom: 4px; }

            .cc-pill { display: inline-block; font-size: 0.66rem; font-weight: 700; padding: 2px 9px; border-radius: 12px; text-transform:uppercase; letter-spacing:.4px; font-family:'JetBrains Mono',monospace; }
            .cc-pill.safe { background: rgba(163,230,53,0.14); color: #bef264; border:1px solid rgba(163,230,53,0.3); }
            .cc-pill.moderate { background: rgba(251,146,60,0.14); color: #fb923c; border:1px solid rgba(251,146,60,0.3); }
            .cc-pill.reach { background: rgba(248,113,113,0.14); color: #f87171; border:1px solid rgba(248,113,113,0.3); }
            </style>
            """, unsafe_allow_html=True)

            for i, row in rdf.head(top_n).iterrows():
                safety_class = "safe" if "Safe" in str(row["Safety"]) else ("moderate" if "Moderate" in str(row["Safety"]) else "reach")
                safety_text  = "Safe" if safety_class == "safe" else ("Moderate" if safety_class == "moderate" else "Reach")
                
                # FIX: We bypass 'college_db' short names entirely to force the full, official name
                inst_full = row["Institute"] 
                
                nirf = row.get("NIRF", "—")
                ctc  = row.get("Avg CTC", "—")
                
                prob_percentage = f"{row['Markov_Prob'] * 100:.1f}%"
                
                tags_html = (
                    f'<span class="cc-tag">{row["Quota"]}</span>'
                    f'<span class="cc-tag">Round {row["Round Closed"]}</span>'
                    f'<span class="cc-tag" style="background:rgba(59,130,246,0.15); color:#60a5fa; border:1px solid rgba(59,130,246,0.3);">⚡ {prob_percentage} Upgradation</span>'
                    + (f'<span class="cc-tag">NIRF #{nirf}</span>' if nirf != "—" else "")
                    + (f'<span class="cc-tag">Avg {ctc}</span>' if ctc != "—" else "")
                )
                
                st.markdown(f"""
                <div class="compact-card {safety_class}">
                    <div class="cc-rank">#{i+1}</div>
                    <div class="cc-info">
                        <div class="cc-inst">{inst_full}</div>
                        <div class="cc-prog">{row["Program"]}</div>
                        <div class="cc-tags">{tags_html}</div>
                    </div>
                    <div class="cc-stats">
                        <div class="cc-close">{int(row["Closing Rank"]):,}</div>
                        <div class="cc-close-lbl">Closing Rank</div>
                        <div class="cc-pill {safety_class}">{safety_text}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            

            # ── Full table (collapsible) ──────────────────────────────
            with st.expander(f"📋 View complete list ({len(rdf)} colleges)", expanded=False):
                rdf_display = rdf.copy()
                rdf_display["Safety"] = rdf_display["Safety"].map(
                    {"Safe":"🟢 Safe","Moderate":"🟡 Moderate","Reach":"🔴 Reach"}).fillna(rdf_display["Safety"])
                rdf_display.insert(0, "#", range(1, len(rdf_display)+1))
                st.dataframe(rdf_display, column_config={
                    "Closing Rank": st.column_config.NumberColumn(format="%d"),
                    "Margin":       st.column_config.NumberColumn(format="%+d"),
                }, use_container_width=True, hide_index=True)

            # ── PDF EXPORT ────────────────────────────────────────────────────
            st.markdown('<div class="section-title">Export Report</div>', unsafe_allow_html=True)

            cdl1, cdl2, _ = st.columns([1, 1, 3])
            with cdl1:
                pdf_bytes = generate_results_pdf(
                    rdf,
                    report_title="JoSAA + CSAB Allocation Report",
                    profile={
                        "JEE Main Rank": f"{rank:,}",
                        "Category":      cat,
                        "Gender":        "Female" if "Female" in gender else "Gender-Neutral",
                        "Home State":    state,
                    },
                    summary_counts={"Total": s+m+r, "Safe": s, "Moderate": m, "Reach": r},
                    table_columns=["Institute","Program","Quota","Closing Rank","Safety","NIRF","Avg CTC"],
                )
                st.download_button(
                    "📄 Download PDF",
                    pdf_bytes,
                    f"JoSAA_CSAB_Report_{state}_{cat}_{rank}.pdf",
                    "application/pdf",
                    key="pdf_dl",
                    use_container_width=True,
                )
            with cdl2:
                st.download_button(
                    "📊 Download CSV",
                    rdf.to_csv(index=False),
                    "josaa_csab_results.csv",
                    "text/csv",
                    key="csv_dl",
                    use_container_width=True,
                )
        else:
            st.warning("No matching options found. Try adjusting rank, category, or enabling the predictive buffer.")
            st.info("""
**Why might an institute be missing?**
- JoSAA + CSAB Special Round only has **seats left vacant after JoSAA main rounds**.
  Some categories at popular institutes fill completely in JoSAA — no seats left for CSAB.
- Try selecting **"OPEN"** category to see broader results first.
- Home State quota rows only appear when the data includes them for that specific round.
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — IIT CUTOFF EXPLORER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("#### IIT Cutoff Explorer")
    st.caption("JEE Advanced cutoffs — JOSAA Round 6 (2024). All India quota only. Ranks are JEE Advanced ranks.")

    with st.expander("ℹ️ Important Note", expanded=False):
        st.markdown("""
        - IITs use **JEE Advanced rank**, not JEE Main rank
        - All IIT seats are **All India (AI) quota** — no Home State quota exists for IITs
        - Eligibility: you must have cleared **both JEE Main and JEE Advanced**
        - This data is JOSAA 2024 Round 6 — reference data for guidance
        """)

    if iit_df.empty:
        st.info("Place `iit_cutoffs.csv` in the project folder to enable IIT tab.")
        st.markdown("**Expected columns:** `Institute, Academic Program Name, Quota, Seat Type, Gender, Opening Rank, Closing Rank`")
    else:
        i1, i2, i3 = st.columns(3)
        with i1: iit_rank = st.number_input("JEE Advanced Rank", 1, 100000, 5000, 100, key="iit_rank")
        with i2: iit_cat  = st.selectbox("Category", sorted(iit_df["Category"].unique()), key="iit_cat")
        with i3: iit_gen  = st.selectbox("Gender",   sorted(iit_df["Gender"].unique()),   key="iit_gen")

        iit_sel    = st.selectbox("Filter by IIT", ["All IITs"] + sorted(iit_df["Institute"].unique()), key="iit_inst")
        iit_domain = st.multiselect("Branch preference (empty=all)",
            ["CS_IT","EE_ECE","Mechanical","Civil","Chemical_Bio","Materials","Sciences","Other"], key="iit_dom")

        if st.button("🏛️ Find IIT Options", type="primary", key="iit_btn"):
            results = []
            for _, row in iit_df.iterrows():
                if row["Category"] != iit_cat: continue
                if row["Gender"]   != iit_gen: continue
                if iit_sel != "All IITs" and row["Institute"] != iit_sel: continue
                if iit_domain and row.get("Program_Domain","Other") not in iit_domain: continue
                if iit_rank > row["Closing Rank"]: continue
                margin = row["Closing Rank"] - iit_rank
                safety = "Safe" if margin >= 500 else ("Moderate" if margin >= 0 else "Reach")
                results.append({"Institute":row["Institute"],"Program":row["Program"],
                                 "Opening Rank":row["Opening Rank"],"Closing Rank":row["Closing Rank"],
                                 "Margin":margin,"Safety":safety})

            if results:
                rdf = pd.DataFrame(results).sort_values("Closing Rank").reset_index(drop=True)
                s = sum(1 for r in results if r["Safety"]=="Safe")
                m = sum(1 for r in results if r["Safety"]=="Moderate")
                rch = sum(1 for r in results if r["Safety"]=="Reach")

                # Summary
                st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
                k1,k2,k3,k4 = st.columns(4)
                k1.markdown(f'<div class="kpi"><h3>{len(results)}</h3><p>Total Options</p></div>', unsafe_allow_html=True)
                k2.markdown(f'<div class="kpi kpi-green"><h3>{s}</h3><p>🟢 Safe</p></div>', unsafe_allow_html=True)
                k3.markdown(f'<div class="kpi kpi-amber"><h3>{m}</h3><p>🟡 Moderate</p></div>', unsafe_allow_html=True)
                k4.markdown(f'<div class="kpi kpi-red"><h3>{rch}</h3><p>🔴 Reach</p></div>', unsafe_allow_html=True)

                # Top picks as cards
                top_n = min(10, len(rdf))
                st.markdown(f'<div class="section-title">Top Recommendations <span class="count">{top_n} of {len(rdf)}</span></div>',
                            unsafe_allow_html=True)
                for i, row in rdf.head(top_n).iterrows():
                    safety_class = row["Safety"].lower()
                    tags_html = f'<span class="tag">All India</span>'
                    st.markdown(f"""
                    <div class="result-card {safety_class}">
                        <div class="rank-num">{i+1}</div>
                        <div class="info">
                            <div class="inst">{row["Institute"]}</div>
                            <div class="prog">{row["Program"]}</div>
                            <div class="tags">{tags_html}</div>
                        </div>
                        <div class="ranks">
                            <div class="closing-rank">{int(row["Closing Rank"]):,}</div>
                            <div class="closing-lbl">closing rank</div>
                            <div class="safety-pill {safety_class}">{row["Safety"]}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Full list grouped by institute
                with st.expander(f"📋 View all results grouped by IIT ({len(rdf)} programs)", expanded=False):
                    for inst in rdf["Institute"].unique():
                        sub = rdf[rdf["Institute"]==inst]
                        st.markdown(f"**{inst}** — {len(sub)} program(s)")
                        st.dataframe(sub[["Program","Opening Rank","Closing Rank","Margin","Safety"]],
                                     use_container_width=True, hide_index=True)

                # Export
                st.markdown('<div class="section-title">Export Report</div>', unsafe_allow_html=True)
                idl1, idl2, _ = st.columns([1, 1, 3])
                with idl1:
                    iit_pdf = generate_results_pdf(
                        rdf,
                        report_title="IIT Cutoff Report (JEE Advanced)",
                        profile={
                            "JEE Advanced Rank": f"{iit_rank:,}",
                            "Category":          iit_cat,
                            "Gender":            "Female" if "Female" in iit_gen else "Gender-Neutral",
                            "Filter":            iit_sel,
                        },
                        summary_counts={"Total": len(rdf), "Safe": s, "Moderate": m, "Reach": rch},
                        table_columns=["Institute","Program","Opening Rank","Closing Rank","Safety"],
                    )
                    st.download_button(
                        "📄 Download PDF",
                        iit_pdf,
                        f"IIT_Report_{iit_cat}_{iit_rank}.pdf",
                        "application/pdf",
                        key="iit_pdf_dl",
                        use_container_width=True,
                    )
                with idl2:
                    st.download_button(
                        "📊 Download CSV",
                        rdf.to_csv(index=False),
                        "iit_options.csv",
                        "text/csv",
                        key="iit_csv_dl",
                        use_container_width=True,
                    )
            else:
                st.warning("No IIT options for your rank and filters.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — STATE COUNSELLING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("#### State Counselling")
    st.caption("State-level engineering counsellings using JEE Main rank.")

    sc1, sc2 = st.tabs(["🏛️ UPTAC — Uttar Pradesh", "🔜 IPU & JAC Delhi (Coming Soon)"])

    with sc1:
        with st.expander("ℹ️ What is UPTAC?", expanded=False):
            st.markdown("""
            **UPTAC** is conducted by AKTU for 200+ colleges in Uttar Pradesh.
            - Uses same JEE Main rank as JoSAA + CSAB
            - **Home State (HS)** — UP domicile students only
            - **All India (AI)** — open to all students
            - Much more relaxed cutoffs than JoSAA + CSAB NITs
            """)

        if uptac_df.empty:
            st.info("Place `uptac_cutoffs.csv` in the project folder.")
        else:
            u1, u2, u3 = st.columns(3)
            with u1: u_rank = st.number_input("JEE Main Rank  ", 1, 1_500_000, 200_000, 1000, key="u_rank")
            with u2: u_cat  = st.selectbox("Category  ", sorted(uptac_df["Category"].unique()), key="u_cat")
            with u3: u_gen  = st.selectbox("Gender  ",   sorted(uptac_df["Gender"].unique()),   key="u_gen")
            u_hs = st.checkbox("I have UP domicile (Home State quota)", value=False, key="u_hs")

            if st.button("🗺️ Find UPTAC Options", type="primary", key="u_btn"):
                results = []
                for _, row in uptac_df.iterrows():
                    if row["Category"] != u_cat: continue
                    if row["Gender"]   != u_gen: continue
                    quota = row.get("Quota","All India")
                    if quota == "Home State" and not u_hs: continue
                    if u_rank > row["Closing Rank"] + PREDICTIVE_BUFFER: continue
                    margin = row["Closing Rank"] - u_rank
                    safety = "Safe" if margin >= 10000 else ("Moderate" if margin >= 0 else "Reach")
                    results.append({"Institute":row["Institute"],"Program":row.get("Program","—"),
                                    "Quota":quota,"Closing Rank":row["Closing Rank"],
                                    "Margin":margin,"Safety":safety})
                if results:
                    rdf = pd.DataFrame(results).sort_values("Closing Rank").reset_index(drop=True)
                    s = sum(1 for r in results if r["Safety"]=="Safe")
                    m = sum(1 for r in results if r["Safety"]=="Moderate")
                    rch = sum(1 for r in results if r["Safety"]=="Reach")

                    # Summary
                    st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
                    k1,k2,k3,k4 = st.columns(4)
                    k1.markdown(f'<div class="kpi kpi-teal"><h3>{len(results)}</h3><p>UPTAC Options</p></div>', unsafe_allow_html=True)
                    k2.markdown(f'<div class="kpi kpi-green"><h3>{s}</h3><p>🟢 Safe</p></div>', unsafe_allow_html=True)
                    k3.markdown(f'<div class="kpi kpi-amber"><h3>{m}</h3><p>🟡 Moderate</p></div>', unsafe_allow_html=True)
                    k4.markdown(f'<div class="kpi kpi-red"><h3>{rch}</h3><p>🔴 Reach</p></div>', unsafe_allow_html=True)

                    # Top picks as cards
                    top_n = min(10, len(rdf))
                    st.markdown(f'<div class="section-title">Top Recommendations <span class="count">{top_n} of {len(rdf)}</span></div>',
                                unsafe_allow_html=True)
                    for i, row in rdf.head(top_n).iterrows():
                        safety_class = row["Safety"].lower()
                        tags_html = f'<span class="tag">{row["Quota"]}</span>'
                        st.markdown(f"""
                        <div class="result-card {safety_class}">
                            <div class="rank-num">{i+1}</div>
                            <div class="info">
                                <div class="inst">{row["Institute"]}</div>
                                <div class="prog">{row["Program"]}</div>
                                <div class="tags">{tags_html}</div>
                            </div>
                            <div class="ranks">
                                <div class="closing-rank">{int(row["Closing Rank"]):,}</div>
                                <div class="closing-lbl">closing rank</div>
                                <div class="safety-pill {safety_class}">{row["Safety"]}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Full table (collapsible)
                    with st.expander(f"📋 View complete list ({len(rdf)} colleges)", expanded=False):
                        rdf_disp = rdf.copy()
                        rdf_disp["Safety"] = rdf_disp["Safety"].map(
                            {"Safe":"🟢 Safe","Moderate":"🟡 Moderate","Reach":"🔴 Reach"}).fillna(rdf_disp["Safety"])
                        rdf_disp.insert(0, "#", range(1, len(rdf_disp)+1))
                        st.dataframe(rdf_disp, column_config={
                            "Closing Rank": st.column_config.NumberColumn(format="%d"),
                            "Margin":       st.column_config.NumberColumn(format="%+d"),
                        }, use_container_width=True, hide_index=True)

                    # Export
                    st.markdown('<div class="section-title">Export Report</div>', unsafe_allow_html=True)
                    udl1, udl2, _ = st.columns([1, 1, 3])
                    with udl1:
                        u_pdf = generate_results_pdf(
                            rdf,
                            report_title="UPTAC Counselling Report",
                            profile={
                                "JEE Main Rank": f"{u_rank:,}",
                                "Category":      u_cat,
                                "Gender":        "Female" if "Female" in u_gen else "Gender-Neutral",
                                "Domicile":      "UP (Home State eligible)" if u_hs else "Outside UP",
                            },
                            summary_counts={"Total": len(rdf), "Safe": s, "Moderate": m, "Reach": rch},
                            table_columns=["Institute","Program","Quota","Closing Rank","Safety"],
                        )
                        st.download_button(
                            "📄 Download PDF",
                            u_pdf,
                            f"UPTAC_Report_{u_cat}_{u_rank}.pdf",
                            "application/pdf",
                            key="u_pdf_dl",
                            use_container_width=True,
                        )
                    with udl2:
                        st.download_button(
                            "📊 Download CSV",
                            rdf.to_csv(index=False),
                            "uptac_results.csv",
                            "text/csv",
                            key="u_csv_dl",
                            use_container_width=True,
                        )
                else:
                    st.warning("No UPTAC options found.")

    with sc2:
        st.markdown("### Coming Soon")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""<div style="padding:1.5rem;border-radius:14px;border:1px dashed rgba(163,230,53,0.45);background:rgba(255,255,255,0.025);text-align:center;">
            <div style="font-size:2rem;">🎓</div><h3 style="color:#bef264;">IPU CET</h3>
            <p style="opacity:.7;">Guru Gobind Singh Indraprastha University · Delhi NCR</p>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown("""<div style="padding:1.5rem;border-radius:14px;border:1px dashed rgba(96,165,250,0.45);background:rgba(255,255,255,0.025);text-align:center;">
            <div style="font-size:2rem;">🏛️</div><h3 style="color:#60a5fa;">JAC Delhi</h3>
            <p style="opacity:.7;">DTU, NSUT, IGDTUW via JEE Main rank</p>
            </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — ML FORECASTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("#### Admission Probability Predictor")
    st.caption("Tell us about you. We'll show eligible colleges and predict your chances at any program.")

    with st.expander("ℹ️ How this works", expanded=False):
        st.markdown("""
        1. Enter your **rank, category, home state, and gender**
        2. The system instantly finds all **eligible colleges** for your profile (applying the correct quota)
        3. Pick any **institute + program** that interests you
        4. Get an **admission probability score** along with a breakdown of how it was calculated

        The predictor takes historical cutoffs, year-over-year seat changes, and quota eligibility into account
        to estimate your realistic chances.
        """)

    # ── Step 1: Student inputs ───────────────────────────────────────────────
    st.markdown("##### Step 1 — Your Details")
    s1, s2, s3, s4 = st.columns(4)
    with s1: ml_rank   = st.number_input("JEE Main Rank", 1, 1_500_000, 50_000, 500, key="ml_rank")
    with s2:
        ml_state = st.selectbox("Home State", ALL_STATES, key="ml_state",
                                index=sorted(ALL_STATES).index("Assam") if "Assam" in ALL_STATES else 0)
    with s3: ml_cat    = st.selectbox("Category", sorted(csab_df["Category"].unique()), key="ml_cat")
    with s4: ml_gender = st.selectbox("Gender",   sorted(csab_df["Gender"].unique()),   key="ml_gender")

    # ── Step 2: Auto-filter eligible institutes ──────────────────────────────
    st.markdown("##### Step 2 — Pick Institute & Program")

    # Find all eligible rows for this student
    eligible_rows = []
    for _, row in csab_df.iterrows():
        if row["Category"] != ml_cat:    continue
        if row["Gender"]   != ml_gender: continue
        if not is_eligible(row["Institute"], ml_state, row["Quota"]): continue
        if ml_rank > row["Closing Rank"] + PREDICTIVE_BUFFER: continue
        eligible_rows.append(row)

    if eligible_rows:
        elig_df = pd.DataFrame(eligible_rows)
        # Sort Home State first, then by closing rank
        quota_order = {"Home State": 0, "All India": 1, "Other State": 2}
        elig_df["_qo"] = elig_df["Quota"].map(quota_order).fillna(3)
        elig_df = elig_df.sort_values(["_qo", "Closing Rank"]).drop(columns=["_qo"])

        # Build institute options with quota label
        elig_df["_inst_label"] = elig_df.apply(
            lambda r: f"{r['Institute']}  [{r['Quota']}]", axis=1)
        inst_options = elig_df["_inst_label"].unique().tolist()

        # Remove duplicate institute+quota combos for the selectbox
        seen, inst_options_dedup = set(), []
        for opt in inst_options:
            if opt not in seen:
                seen.add(opt)
                inst_options_dedup.append(opt)

        st.info(f"✅ **{len(elig_df['Institute'].unique())} eligible institutes** found for your profile.")

        sel_inst_label = st.selectbox("Select Institute", inst_options_dedup, key="ml_inst")

        # Extract institute name and quota from label
        # Format: "Institute Name  [Quota]"
        import re as _re
        _match = _re.match(r"^(.*)\s+\[([^\]]+)\]$", sel_inst_label)
        if _match:
            sel_inst_name  = _match.group(1).strip()
            sel_inst_quota = _match.group(2).strip()
        else:
            sel_inst_name  = sel_inst_label
            sel_inst_quota = "All India"

        # Programs for this institute + quota + category + gender
        prog_rows = elig_df[
            (elig_df["Institute"] == sel_inst_name) &
            (elig_df["Quota"]     == sel_inst_quota)
        ]
        prog_options = sorted(prog_rows["Program"].unique())
        sel_prog = st.selectbox("Select Program", prog_options, key="ml_prog")

    else:
        st.warning("No eligible institutes found for your rank/category/state. "
                   "Try increasing your rank or changing category.")
        st.stop()

    # ── Step 3: Predict ───────────────────────────────────────────────────────
    st.markdown("##### Step 3 — Predict")

    if st.button("🔮 Predict Admission Probability", type="primary", key="ml_btn"):
        try:
            itype  = csab_df[csab_df["Institute"] == sel_inst_name]["Institute_Type"].iloc[0] \
                     if "Institute_Type" in csab_df.columns else "GFTI"
            domain = csab_df[csab_df["Program"] == sel_prog]["Program_Domain"].iloc[0] \
                     if "Program_Domain" in csab_df.columns \
                        and len(csab_df[csab_df["Program"] == sel_prog]) > 0 \
                     else "Other"

            match = csab_df[
                (csab_df["Institute"] == sel_inst_name) &
                (csab_df["Program"]   == sel_prog) &
                (csab_df["Quota"]     == sel_inst_quota) &
                (csab_df["Category"]  == ml_cat) &
                (csab_df["Gender"]    == ml_gender)
            ]
            rc  = int(match.iloc[0]["Closing Rank"]) if not match.empty else ml_rank
            ro  = int(match.iloc[0]["Opening Rank"])  if not match.empty else ml_rank
            rs  = rc - ro
            ci  = 1 / (1 + np.log1p(rc))
            rr  = ml_rank / max(rc, 1)
            dfc = rc - ml_rank

            enc = [
                safe_enc("Institute_Type",  itype),
                safe_enc("Institute",       sel_inst_name),
                safe_enc("Program",         sel_prog),
                safe_enc("Quota",           sel_inst_quota),
                safe_enc("Category",        ml_cat),
                safe_enc("Gender",          ml_gender),
                safe_enc("Program_Domain",  domain),
            ]
            features = np.array([enc + [ml_rank, rs, ci, rr, dfc]])
            ap = model.predict_proba(features)[0][1] * 100

            # ── Result display ────────────────────────────────────────────────
            col_g, col_d = st.columns([1, 2])
            with col_g:
                color = "#a3e635" if ap >= 70 else ("#fb923c" if ap >= 40 else "#f87171")
                fig   = go.Figure(go.Indicator(
                    mode="gauge+number", value=ap,
                    number={"suffix": "%", "font": {"size": 44, "color": color}},
                    gauge={
                        "axis":    {"range": [0, 100], "tickcolor": "#8b97a3"},
                        "bar":     {"color": color},
                        "bgcolor": "rgba(0,0,0,0)",
                        "steps": [
                            {"range": [0,  40], "color": "rgba(248,113,113,.10)"},
                            {"range": [40, 70], "color": "rgba(251,146,60,.10)"},
                            {"range": [70,100], "color": "rgba(163,230,53,.12)"},
                        ],
                    }
                ))
                fig.update_layout(height=260, margin=dict(t=30,b=0,l=30,r=30),
                                   font=dict(color="#e7ecef", family="JetBrains Mono"),
                                   paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            with col_d:
                st.markdown("**Prediction Summary**")
                info = college_db.get(sel_inst_name, {})
                st.markdown(f"🏛️ **{info.get('short', sel_inst_name)}**")
                st.markdown(f"📚 {sel_prog[:70]}")
                st.markdown(f"🏷️ Quota: **{sel_inst_quota}** · Category: **{ml_cat}** · {ml_gender}")
                st.markdown(f"📍 Home State: **{ml_state}**")
                st.markdown("")
                if not match.empty:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Opening Rank", f"{ro:,}")
                    m2.metric("Closing Rank", f"{rc:,}")
                    m3.metric("Your Rank",    f"{ml_rank:,}")
                    m4.metric("Gap",          f"{dfc:+,}")
                st.markdown("")
                if ap >= 70:
                    st.success("✅ **High likelihood** of getting this seat.")
                elif ap >= 40:
                    st.warning("⚠️ **Moderate chance** — realistic but not guaranteed.")
                else:
                    st.error("❌ **Low probability** — keep as an aspirational choice.")

                # Explain the score
                with st.expander("📊 How was this calculated?"):
                    st.markdown(f"""
                    | Factor | Your Value |
                    |---------|-------|
                    | Gap from cutoff | {dfc:+,} ranks |
                    | Your rank vs closing | {rr:.2f}x |
                    | Cutoff spread | {rs:,} ranks |
                    | Institute tier | {itype} |
                    | Branch category | {domain.replace('_', ' ')} |
                    | Quota applied | {sel_inst_quota} |

                    *Your probability is calculated by comparing your rank against historical cutoffs,
                    factoring in quota eligibility, branch demand, and year-over-year seat expansion trends.*
                    """)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — AI COUNSELOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.markdown("#### Chat with Your AI Counselor")
    st.caption("Ask anything about colleges, branches, cutoffs, or counselling — get personalised guidance instantly.")

    def _get_groq_key():
        # 1) .streamlit/secrets.toml  ->  GROQ_API_KEY = "..."
        try:
            if "GROQ_API_KEY" in st.secrets:
                return str(st.secrets["GROQ_API_KEY"]).strip()
        except Exception:
            pass
        # 2) .streamlit/secrets.toml  ->  [groq] \n api_key = "..."
        try:
            return str(st.secrets["groq"]["api_key"]).strip()
        except Exception:
            pass
        # 3) environment variable fallback
        return os.environ.get("GROQ_API_KEY", "").strip()

    GROQ_API_KEY = _get_groq_key()
    if not GROQ_API_KEY:
        st.warning("The AI Counselor is currently unavailable. Add a Groq API key to enable it.")
        with st.expander("For developers / admins"):
            st.markdown("Add your key to **`.streamlit/secrets.toml`** in the project root:")
            st.code('GROQ_API_KEY = "gsk_your_key_here"', language="toml")
            st.markdown("…or set the `GROQ_API_KEY` environment variable. "
                        "Get a free key at [console.groq.com](https://console.groq.com).")
        st.stop()

    client = Groq(api_key=GROQ_API_KEY)

    # ── Build live, data-grounded context for the counselor ──
    top12   = csab_df.groupby("Institute")["Closing Rank"].min().nsmallest(12)
    top_ctx = "\n".join([f"  - {i}: best closing rank ~{r:,}" for i, r in top12.items()])
    cats    = ", ".join(sorted(csab_df["Category"].dropna().unique()))

    SYSTEM = f"""You are "Avricus AI", a sharp, friendly, highly knowledgeable senior engineering-admissions counselor for Indian students (JEE Main / JEE Advanced).

YOUR LIVE DATA (this platform):
- JoSAA + CSAB: {len(csab_df):,} cutoff records across {csab_df['Institute'].nunique()} institutes (NITs, IIITs, GFTIs).
- IITs: {len(iit_df):,} JEE-Advanced cutoff records.
- UPTAC: {len(uptac_df):,} Uttar-Pradesh college records.
- Categories present in data: {cats}.
- Strongest institutes by best closing rank:
{top_ctx}

WHAT YOU KNOW COLD (use confidently):
- Counselling flow: JoSAA runs the main joint allotment for IITs + NITs + IIITs + GFTIs over several rounds. CSAB Special is the mop-up round that fills NIT/IIIT/GFTI seats left vacant after JoSAA. State counselling (e.g. UPTAC in UP) is separate, on its own merit list.
- Quotas: NITs/IIITs/GFTIs use the All-India (AI) quota; NITs ALSO split seats into Home-State (HS, ~50%) vs Other-State (OS) — HS applies only at the NIT in the student's own home state. IIITs and GFTIs have NO HS/OS split.
- Reservation (central institutes): OPEN, EWS ~10%, OBC-NCL ~27%, SC ~15%, ST ~7.5%, plus PwD ~5% horizontal. Female-supernumerary seats improve girls' chances.
- Ranks: lower number = better. "Safe" = rank beats the closing rank by a comfortable margin; "Moderate" = just inside the cutoff; "Reach" = above the cutoff (unlikely but possible via later rounds).
- Branch families: TECH = CSE, IT, ECE, EE, AI/ML, Data Science, Mathematics & Computing, VLSI, Cyber, Robotics. CORE = Civil, Mechanical, Chemical, Metallurgy, Materials, Mining, Aerospace, Production.

HOW TO ANSWER:
- Be genuinely helpful on ANY question — college shortlists, branch-vs-branch, placements, cutoffs, eligibility, fees, hostels, study strategy, career scope, comparisons, or general doubts. Never refuse a reasonable question.
- If you have the rank/category/home-state, give specific, ranked, data-backed suggestions. If something essential is missing, make a sensible assumption, answer anyway, and mention what would sharpen it.
- Be concise and well-structured (short paragraphs or tight bullets). Use real institute names and numbers where you can.
- Be honest about uncertainty: cutoffs shift year to year — tell students to confirm on the official portal for the current year. Never promise guaranteed admission.
- Warm, encouraging, never condescending. Assume Indian context (LPA, CTC, JEE, JoSAA).
Sign off every reply with: "— Avricus AI"."""

    # Human-readable topic labels (hide internal NLP intent names from users)
    TOPIC_LABELS = {
        "college_recommendation": "🎯 College Suggestions",
        "branch_comparison":      "🔀 Branch Comparison",
        "cutoff_inquiry":         "📊 Cutoff Lookup",
        "counseling_process":     "📋 Counselling Help",
        "career_guidance":        "🚀 Career Guidance",
        "category_reservation":   "🏷️ Category & Quota",
        "admission_probability":  "🔮 Admission Chances",
        "greeting":               "👋 Hello",
    }

    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"]=="user" and "ii" in msg:
                c = INTENT_COLORS.get(msg["ii"]["intent"],"#666")
                label = TOPIC_LABELS.get(msg["ii"]["intent"], "💬 General")
                st.markdown(f'<span class="nlp-badge" style="background:{c};">{label}</span>',
                            unsafe_allow_html=True)

    if prompt := st.chat_input("Ask about colleges, cutoffs, branches, or admission chances..."):
        intent, conf, ents = classify_intent(prompt)
        ii = {"intent":intent,"confidence":conf,"entities":ents}
        st.session_state.messages.append({"role":"user","content":prompt,"ii":ii})
        with st.chat_message("user"):
            st.markdown(prompt)
            c = INTENT_COLORS.get(intent,"#666")
            label = TOPIC_LABELS.get(intent, "💬 General")
            st.markdown(f'<span class="nlp-badge" style="background:{c};">{label}</span>',
                        unsafe_allow_html=True)
        # Internally enriched prompt (with intent + entities) helps the LLM, but isn't shown to the user
        enriched  = f"[Context · Topic:{intent} · Entities:{json.dumps(ents)}]\n[USER] {prompt}"
        api_msgs  = [{"role":"system","content":SYSTEM}]
        for m in st.session_state.messages[-20:]:
            api_msgs.append({"role":m["role"],"content":m["content"]})
        api_msgs[-1]["content"] = enriched
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp  = client.chat.completions.create(model="llama-3.3-70b-versatile",
                                messages=api_msgs, temperature=0.6, max_tokens=2048)
                    reply = resp.choices[0].message.content
                except Exception as e:
                    reply = f"Sorry, something went wrong: {e}"
            st.markdown(reply)
        st.session_state.messages.append({"role":"assistant","content":reply})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6 — COLLEGE COMPARISON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab6:
    st.markdown("#### College Comparison")
    cmp1, cmp2 = st.tabs(["📖 Institute Profile", "⚖️ Head-to-Head"])
    all_cmp = sorted(csab_df["Institute"].unique())

    with cmp1:
        sel = st.selectbox("Select Institute", all_cmp, key="prof_sel")
        info = college_db.get(sel, {})
        st.markdown(f"### {info.get('short', sel)}")
        st.caption(f"📍 {info.get('city','—')}")
        st.markdown("")
        if info.get("est") != "—":
            s1,s2,s3,s4,s5,s6 = st.columns(6)
            for col, lbl, val in [
                (s1,"Est.",info["est"]),  (s2,"NIRF",f"#{info['nirf_rank']}"),
                (s3,"Campus",f"{info['campus_acres']} ac"), (s4,"Placement",f"{info['placement_pct']}%"),
                (s5,"Highest CTC",info["highest_ctc"]),    (s6,"Avg CTC",info["avg_ctc"]),
            ]:
                col.markdown(f'<div class="stat-box"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                             unsafe_allow_html=True)
            st.markdown("")
            st.markdown(f"**Top Recruiters:** {info['top_recruiters']}")
            st.markdown(f"**Hostel:** {info['hostel']}")
        else:
            st.info("Detailed profile not yet available for this institute.")
        st.markdown("")
        st.markdown("##### JoSAA + CSAB Cutoffs (OPEN · Gender-Neutral)")
        sub = csab_df[(csab_df["Institute"]==sel)&(csab_df["Category"]=="OPEN")&
                      (csab_df["Gender"]=="Gender-Neutral")].sort_values("Closing Rank")
        if sub.empty: sub = csab_df[csab_df["Institute"]==sel].sort_values("Closing Rank").head(15)
        cols_show = ["Program","Quota","Opening Rank","Closing Rank"]
        if "Closed_In_Round" in sub.columns: cols_show.append("Closed_In_Round")
        st.dataframe(sub[cols_show], use_container_width=True, hide_index=True)

    with cmp2:
        c1c, c2c = st.columns(2)
        with c1c: inst_a = st.selectbox("College A", all_cmp, key="cmp_a")
        with c2c: inst_b = st.selectbox("College B", all_cmp, index=min(1,len(all_cmp)-1), key="cmp_b")

        if st.button("⚖️ Compare", type="primary", key="cmp_btn"):
            ia, ib = college_db.get(inst_a,{}), college_db.get(inst_b,{})
            na, nb = ia.get("short", inst_a[:25]), ib.get("short", inst_b[:25])
            fields = [("Established","est"),("NIRF Rank","nirf_rank"),("City","city"),
                      ("Campus (acres)","campus_acres"),("Highest CTC","highest_ctc"),
                      ("Average CTC","avg_ctc"),("Median CTC","median_ctc"),
                      ("Placement %","placement_pct"),("Top Recruiters","top_recruiters"),("Hostel","hostel")]
            rows = [{"Parameter":l, na:str(ia.get(k,"—")), nb:str(ib.get(k,"—"))} for l,k in fields]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=420)

            vis = []
            for l,k in [("NIRF Rank","nirf_rank"),("Campus (ac)","campus_acres"),("Placement %","placement_pct")]:
                va,vb = ia.get(k,0), ib.get(k,0)
                if isinstance(va,(int,float)) and isinstance(vb,(int,float)):
                    vis.append({"Metric":l, na:va, nb:vb})
            if vis:
                vdf = pd.DataFrame(vis)
                fig = go.Figure()
                fig.add_trace(go.Bar(name=na, x=vdf["Metric"], y=vdf[na], marker_color="#a3e635"))
                fig.add_trace(go.Bar(name=nb, x=vdf["Metric"], y=vdf[nb], marker_color="#60a5fa"))
                fig.update_layout(barmode="group", height=340,
                                   font=dict(color="#e7ecef", family="JetBrains Mono"),
                                   legend=dict(font=dict(color="#e7ecef")),
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Cutoff Comparison (OPEN · Gender-Neutral)")
            for inst, label in [(inst_a, na),(inst_b, nb)]:
                sub = csab_df[(csab_df["Institute"]==inst)&(csab_df["Category"]=="OPEN")&
                              (csab_df["Gender"]=="Gender-Neutral")].sort_values("Closing Rank")
                st.markdown(f"**{label}**")
                if not sub.empty:
                    st.dataframe(sub[["Program","Quota","Opening Rank","Closing Rank"]],
                                 use_container_width=True, hide_index=True)
                else:
                    st.caption("No OPEN/Gender-Neutral data available.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div style="margin-top:3rem;padding:1.8rem 0 1rem;border-top:1px solid rgba(255,255,255,0.08);
            text-align:center;color:#94a3b8;font-size:.8rem;line-height:1.7;">
    <div style="font-weight:700;color:#e2e8f0;font-size:.95rem;margin-bottom:.3rem;">
        🎓 Avricus AI
    </div>
    <div style="opacity:.85;">
        Smart counseling, simplified. Find your perfect college — backed by data.
    </div>
    <div style="margin-top:.6rem;font-size:.72rem;opacity:.6;">
        © 2026 Avricus AI · All rights reserved · This platform provides guidance only.
        Always verify with official JoSAA / CSAB / state counselling portals.
    </div>
</div>
""", unsafe_allow_html=True)
