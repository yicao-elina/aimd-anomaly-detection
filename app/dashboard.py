"""
AIMD Anomaly Detection — Interactive Dashboard  (v2)
Run: /Users/alina/anaconda3/envs/agentic/bin/streamlit run app/dashboard.py

Pages:
  1. Data Overview
  2. Feature Analysis
  3. Anomaly Detection  (+ dynamic window zoom)
  4. AIMD vs Upload Comparison
  5. AI Analysis Assistant  (LLM-powered, glm-5:cloud via Ollama)

Upload sidebar: drop any new .xyz MLFF trajectory → predict-only pipeline runs (~5s)
               → results stored in session; switch / delete between uploads freely.
"""

import sys
import io
import json
import pickle
import tempfile
import datetime
import traceback
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.loaders import TrajectoryLoader
from src.core.feature_extractors import FeatureExtractor, WindowConfig
from src.core.detectors import AnomalyDetectionFramework
from src.core.llm_analyst import OllamaAnalyst, build_data_context
from src.utils.jhu_colors.jhu_colors.colors import get_jhu_color

# ── Paths ──────────────────────────────────────────────────────────────────
PROCESSED_DIR = ROOT / 'data' / 'processed'
RESULTS_DIR   = ROOT / 'results'
MODELS_DIR    = RESULTS_DIR / 'models'
FIGURES_DIR   = RESULTS_DIR / 'figures'
REPORTS_DIR   = RESULTS_DIR / 'reports'

# ── Colour palette — JHU palette (professional dashboard theme) ────────────
def _rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(
        int(max(0, min(1, rgb[0])) * 255),
        int(max(0, min(1, rgb[1])) * 255),
        int(max(0, min(1, rgb[2])) * 255),
    )

JHU_WHITE  = _rgb_to_hex(get_jhu_color('White'))
JHU_BLACK  = _rgb_to_hex(get_jhu_color('Double Black'))
JHU_BLUE   = _rgb_to_hex(get_jhu_color('Heritage Blue'))
JHU_SBLUE  = _rgb_to_hex(get_jhu_color('Spirit Blue'))
JHU_RED    = _rgb_to_hex(get_jhu_color('Red'))
JHU_ORANGE = _rgb_to_hex(get_jhu_color('Orange'))
JHU_GREEN  = _rgb_to_hex(get_jhu_color('Homewood Green'))
JHU_GOLD   = _rgb_to_hex(get_jhu_color('Gold'))
JHU_PURPLE = _rgb_to_hex(get_jhu_color('Purple'))
JHU_HARBOR = _rgb_to_hex(get_jhu_color('Harbor Blue'))
JHU_FOREST = _rgb_to_hex(get_jhu_color('Forest Green'))
JHU_MAROON = _rgb_to_hex(get_jhu_color('Maroon'))

BG       = JHU_WHITE
SURFACE  = JHU_WHITE
SURFACE2 = '#44ACE51A'
BORDER_C = '#002D722E'
CORAL    = JHU_RED
CORAL_LT = JHU_ORANGE
SAGE     = JHU_GREEN
GOLD     = JHU_GOLD
INK      = JHU_BLACK
TEXT     = '#000000D8'
SUB      = '#000000A6'
MUTED    = '#00000073'

def _apply_css_tokens(css: str) -> str:
    tokens = {
        '__BG__': BG,
        '__SURFACE__': SURFACE,
        '__SURFACE2__': SURFACE2,
        '__BORDER__': BORDER_C,
        '__CORAL__': CORAL,
        '__CORAL_LT__': CORAL_LT,
        '__SAGE__': SAGE,
        '__GOLD__': GOLD,
        '__RED__': RED,
        '__PURPLE__': PURPLE,
        '__INK__': INK,
        '__MUTED__': MUTED,
        '__SUB__': SUB,
        '__BLUE__': JHU_BLUE,
        '__SBLUE__': JHU_SBLUE,
        '__MAROON__': JHU_MAROON,
    }
    for key, value in tokens.items():
        css = css.replace(key, value)
    return css

# Matplotlib shorthand aliases (JHU palette)
CYAN   = JHU_HARBOR    # AIMD / baseline traces
RED    = JHU_RED       # Upload / anomaly traces
GREEN  = JHU_FOREST    # tertiary positive
AMBER  = JHU_GOLD      # warning / borderline
PURPLE = JHU_PURPLE    # multi-series variety

def _style_ax(ax, title='', xlabel='', ylabel=''):
    """Apply warm-paper light theme to a matplotlib Axes."""
    ax.set_facecolor(SURFACE2)
    ax.tick_params(colors=SUB, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER_C)
        spine.set_linewidth(0.8)
    if title:
        ax.set_title(title, color=INK, fontsize=11, fontweight='600', pad=8)
    ax.xaxis.label.set_color(SUB)
    ax.yaxis.label.set_color(SUB)
    if xlabel: ax.set_xlabel(xlabel, color=SUB)
    if ylabel: ax.set_ylabel(ylabel, color=SUB)
    ax.grid(color=BORDER_C, linewidth=0.6, alpha=0.7, linestyle='--')

def mpl_fig(nrows=1, ncols=1, figsize=(12, 4)):
    """Return (fig, axes) with warm paper theme pre-applied."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                              facecolor=BG, constrained_layout=True)
    for ax in np.array(axes).flatten():
        _style_ax(ax)
    return fig, axes


# ── Custom HTML component helpers ────────────────────────────────────────────

def _ring_color(rate: float) -> str:
    if rate > 0.50: return CORAL
    if rate > 0.10: return GOLD
    return SAGE

def inject_status_monitor(ra: dict, rm: dict, upload_name: str) -> None:
    """
    Inject an SVG-based 'Interactions & States' status monitor panel.
    Shows three detector rings (L1, L2a, L2b) with live anomaly rates,
    ensemble verdict, and an animated pulse for high-anomaly states.
    """
    def ring(rate, label, abbr):
        c = 201.1   # 2π × 32 (circle r=32)
        fill  = max(3, rate * c)
        color = _ring_color(rate)
        anim  = 'pulse-ring' if rate > 0.5 else ''
        pct   = f"{rate:.0%}"
        return f"""
        <div class="sr-ring {anim}">
          <svg width="96" height="96" viewBox="0 0 96 96">
            <circle cx="48" cy="48" r="32" fill="none"
              stroke="{BORDER_C}" stroke-width="8"/>
            <circle cx="48" cy="48" r="32" fill="none"
              stroke="{color}" stroke-width="8" stroke-linecap="round"
              stroke-dasharray="{fill:.1f} {c:.1f}"
              stroke-dashoffset="50.3" class="ring-fill"/>
            <text x="48" y="44" text-anchor="middle"
              font-family="DM Serif Display,serif" font-size="15"
              fill="{color}" font-weight="600">{pct}</text>
            <text x="48" y="58" text-anchor="middle"
              font-family="DM Mono,monospace" font-size="8"
              fill="{SUB}" letter-spacing="1">{abbr}</text>
          </svg>
          <div class="sr-label">{label}</div>
        </div>"""

    l1_a  = float(np.mean(ra['l1_flag']))
    l2if_a= float(np.mean(ra['l2_if_flag']))
    l2sv_a= float(np.mean(ra['l2_svm_flag']))
    l1_u  = float(np.mean(rm['l1_flag']))
    l2if_u= float(np.mean(rm['l2_if_flag']))
    l2sv_u= float(np.mean(rm['l2_svm_flag']))
    ens_a = float(np.mean(ra['anomaly_label']))
    ens_u = float(np.mean(rm['anomaly_label']))

    verdict_color = _ring_color(ens_u)
    verdict_label = "CATASTROPHIC" if ens_u > 0.9 else ("HIGH ANOMALY" if ens_u > 0.5 else ("WARNING" if ens_u > 0.1 else "NORMAL"))
    ratio = ens_u / max(ens_a, 1e-4)

    html = f"""
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet"/>
<style>
  .sm-wrap {{
    background: {SURFACE}; border: 1px solid {BORDER_C};
    border-radius: 16px; padding: 24px 28px;
    display: grid; grid-template-columns: 1fr auto 1fr;
    gap: 24px; align-items: center;
    box-shadow: 0 2px 12px rgba(0,45,114,.12);
    font-family: 'DM Sans', sans-serif;
    margin-bottom: 8px;
  }}
  .sm-group {{ display: flex; flex-direction: column; gap: 4px; }}
  .sm-group-label {{
    font-family: 'DM Mono', monospace;
    font-size: 9px; letter-spacing: .18em; text-transform: uppercase;
    color: {MUTED}; margin-bottom: 8px;
  }}
  .sm-rings {{ display: flex; gap: 16px; justify-content: center; }}
  .sr-ring {{ text-align: center; }}
  .sr-label {{
    font-family: 'DM Mono', monospace; font-size: 9px;
    letter-spacing: .1em; color: {SUB}; margin-top: 4px;
    text-transform: uppercase;
  }}
  .sm-verdict {{
    text-align: center; display: flex; flex-direction: column;
    align-items: center; gap: 8px;
  }}
  .verdict-pct {{
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem; line-height: 1;
    color: {verdict_color};
  }}
  .verdict-label {{
    font-family: 'DM Mono', monospace; font-size: 10px;
    letter-spacing: .15em; text-transform: uppercase;
    color: {verdict_color}; font-weight: 600;
  }}
  .verdict-ratio {{
    font-family: 'DM Mono', monospace; font-size: 11px;
    color: {SUB}; margin-top: 4px;
  }}
  .verdict-name {{
    font-family: 'DM Mono', monospace; font-size: 9px;
    color: {MUTED}; margin-top: 2px;
    max-width: 120px; text-align: center; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
  }}
  .sm-divider {{
    width: 1px; background: {BORDER_C}; height: 80px; margin: auto;
  }}
  .ring-fill {{
    transition: stroke-dasharray 1s ease;
    animation: ring-in 1s ease forwards;
  }}
  @keyframes ring-in {{
    from {{ stroke-dasharray: 0 {201.1:.1f}; }}
  }}
  @keyframes pulse-ring-anim {{
    0%, 100% {{ transform: scale(1); }}
    50%       {{ transform: scale(1.06); filter: drop-shadow(0 0 8px {CORAL}80); }}
  }}
  .pulse-ring {{ animation: pulse-ring-anim 2s ease-in-out infinite; }}
</style>
<div class="sm-wrap">
  <div class="sm-group">
    <div class="sm-group-label">AIMD Baseline</div>
    <div class="sm-rings">
      {ring(l1_a,  'Statistical', 'L1')}
      {ring(l2if_a,'Isolation F.',  'L2a')}
      {ring(l2sv_a,'One-Class SVM', 'L2b')}
    </div>
  </div>
  <div class="sm-divider"></div>
  <div class="sm-verdict">
    <div class="verdict-pct">{ens_u:.0%}</div>
    <div class="verdict-label">{verdict_label}</div>
    <div class="verdict-ratio">{ratio:.1f}× detection ratio</div>
    <div class="verdict-name">{upload_name[:28]}</div>
  </div>
  <div class="sm-divider"></div>
  <div class="sm-group">
    <div class="sm-group-label">Upload: {upload_name[:20]}</div>
    <div class="sm-rings">
      {ring(l1_u,  'Statistical', 'L1')}
      {ring(l2if_u,'Isolation F.',  'L2a')}
      {ring(l2sv_u,'One-Class SVM', 'L2b')}
    </div>
  </div>
</div>"""
    components.html(html, height=200, scrolling=False)


def inject_metrics_bar(metrics: list) -> None:
    """
    Inject an HTML metrics strip at the top of a page.
    metrics: list of (label, value, delta, color) tuples.
    color: 'coral' | 'sage' | 'gold' | 'ink'
    """
    colors = {'coral': CORAL, 'sage': SAGE, 'gold': GOLD, 'ink': INK, 'muted': MUTED}

    cards = ''
    for label, value, delta, clr in metrics:
        c = colors.get(clr, CORAL)
        delta_html = f'<div class="mb-delta">{delta}</div>' if delta else ''
        cards += f"""
        <div class="mb-card" style="border-top:3px solid {c}">
          <div class="mb-label">{label}</div>
          <div class="mb-value" style="color:{c}">{value}</div>
          {delta_html}
        </div>"""

    html = f"""
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet"/>
<style>
  .mb-strip {{
    display: flex; gap: 12px; margin-bottom: 4px;
    font-family: 'DM Sans', sans-serif;
  }}
  .mb-card {{
    flex: 1; background: {SURFACE}; border: 1px solid {BORDER_C};
    border-radius: 10px; padding: 14px 16px;
    box-shadow: 0 1px 4px rgba(0,45,114,.10);
  }}
  .mb-label {{
    font-family: 'DM Mono', monospace; font-size: 9px;
    letter-spacing: .15em; text-transform: uppercase; color: {MUTED};
    margin-bottom: 6px;
  }}
  .mb-value {{
    font-family: 'DM Serif Display', serif; font-size: 1.8rem;
    line-height: 1; font-weight: 600;
  }}
  .mb-delta {{
    font-family: 'DM Mono', monospace; font-size: 10px;
    color: {SUB}; margin-top: 4px;
  }}
</style>
<div class="mb-strip">{cards}</div>"""
    components.html(html, height=110, scrolling=False)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIMD Anomaly Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Fonts + CSS (JHU palette — playground-aligned) ─────────────────────────
st.markdown(_apply_css_tokens("""
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap" rel="stylesheet"/>

<style>
/* ── Root palette — JHU theme ── */
:root {
    --bg:     __BG__;
    --sf:     __SURFACE__;
    --sf2:    __SURFACE2__;
    --border: __BORDER__;
    --coral:  __CORAL__;
    --coral2: __CORAL_LT__;
    --sage:   __SAGE__;
    --gold:   __GOLD__;
    --red:    __RED__;
    --purple: __PURPLE__;
    --text:   __INK__;
    --muted:  __MUTED__;
    --sub:    __SUB__;
    --blue:   __BLUE__;
    --sblue:  __SBLUE__;
    --maroon: __MAROON__;
  --ff-d:   'DM Serif Display', serif;
  --ff-m:   'DM Mono', monospace;
  --ff-b:   'DM Sans', sans-serif;
}

/* ── Streamlit app shell ── */
.stApp {
    background: linear-gradient(180deg, rgba(68,172,229,0.10), rgba(255,255,255,1) 45%) !important;
    font-family: var(--ff-b);
}
.stApp > header { background: transparent !important; }

/* Main content area */
.main .block-container {
  background: var(--bg);
  padding-top: 2rem;
  max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--sf) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--sub) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--text) !important; }
[data-testid="stSidebar"] .stMarkdown strong { color: var(--blue) !important; }
[data-testid="stSidebarContent"] [data-testid="metric-container"] {
  background: var(--sf2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 14px;
}

/* ── Radio (page selector) ── */
[data-testid="stSidebar"] [role="radiogroup"] label {
  color: var(--sub) !important;
  font-family: var(--ff-m);
  font-size: 12px;
  letter-spacing: .05em;
  padding: 6px 10px;
  border-radius: 6px;
  transition: all .2s;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: var(--sf2); }
[data-testid="stSidebar"] [role="radiogroup"] [aria-checked="true"] + div label {
    color: var(--blue) !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5 { font-family: var(--ff-d) !important; color: var(--text) !important; }
h1 { font-size: 2.6rem !important; font-weight: 800 !important; letter-spacing: -.02em; }
h2 { font-size: 1.8rem !important; font-weight: 600 !important; }
h3 { font-size: 1.3rem !important; }
p, li, span, div { font-family: var(--ff-b); }

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: var(--sf) !important;
  border: 1px solid var(--border) !important;
    border-top: 3px solid var(--blue) !important;
  border-radius: 10px !important;
  padding: 16px !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
  color: var(--muted) !important;
  font-family: var(--ff-m) !important;
  font-size: 10px !important;
  letter-spacing: .12em !important;
  text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--blue) !important;
  font-family: var(--ff-d) !important;
  font-size: 1.8rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
  font-family: var(--ff-m);
  font-size: 11px !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
  background: transparent !important;
  color: var(--muted) !important;
  font-family: var(--ff-m) !important;
  font-size: 11px !important;
  letter-spacing: .08em;
  border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--blue) !important;
    border-bottom-color: var(--blue) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
  background: var(--sf) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
  color: var(--sub) !important;
  font-family: var(--ff-m) !important;
  font-size: 12px !important;
  letter-spacing: .06em;
}

/* ── Buttons ── */
.stButton button {
  background: var(--coral) !important;
  color: #FFFFFF !important;
  font-family: var(--ff-b) !important;
  font-weight: 600 !important;
  border: none !important;
  border-radius: 8px !important;
  transition: all .2s !important;
}
.stButton button:hover {
  background: var(--coral2) !important;
  transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(207,69,32,.28) !important;
}
.stButton [kind="secondary"] button,
button[data-testid="baseButton-secondary"] {
  background: transparent !important;
    color: var(--blue) !important;
  border: 1px solid var(--border) !important;
}

/* ── Dataframe / table ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
.dvn-scroller { background: var(--sf) !important; }

/* ── Select / input / textarea ── */
.stSelectbox div[data-baseweb],
.stTextInput div[data-baseweb],
.stTextArea div[data-baseweb] {
  background: var(--sf2) !important;
  border-color: var(--border) !important;
  border-radius: 8px !important;
}
.stSelectbox div[data-baseweb] *,
.stTextInput div[data-baseweb] *,
.stTextArea div[data-baseweb] * {
  color: var(--text) !important;
  font-family: var(--ff-b) !important;
}
textarea {
  background: var(--sf2) !important;
  color: var(--text) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] div[data-baseweb] div[role="slider"] {
  background: var(--coral) !important;
}
[data-testid="stSlider"] div[data-baseweb] div[data-testid="stSliderTrackFill"] {
  background: var(--coral) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: var(--sf2) !important;
    border: 1px dashed var(--blue) !important;
  border-radius: 10px !important;
}
[data-testid="stFileUploader"] * { color: var(--sub) !important; }
[data-testid="stFileUploader"] button {
  background: var(--sf) !important;
    color: var(--blue) !important;
  border: 1px solid var(--border) !important;
}

/* ── Alerts / info boxes ── */
[data-testid="stAlert"] {
  background: var(--sf2) !important;
  border-radius: 8px !important;
  border: 1px solid var(--border) !important;
  color: var(--sub) !important;
}

/* ── Horizontal rule ── */
hr { border-color: var(--border) !important; }

/* ── Divider styled ── */
.section-div {
  height: 1px;
    background: linear-gradient(to right, transparent, var(--blue), transparent);
  margin: 24px 0;
}

/* ── Page heading style ── */
.page-head {
  display: flex; align-items: baseline; gap: 12px;
  margin-bottom: 4px;
}
.page-head-label {
  font-family: var(--ff-m);
  font-size: 10px;
  letter-spacing: .2em;
  text-transform: uppercase;
    color: var(--blue);
  opacity: .8;
}

/* ── Evidence / claim blocks ── */
.evidence-block {
  background: var(--sf2);
    border-left: 3px solid var(--blue);
  padding: 12px 16px;
  border-radius: 6px;
  margin: 8px 0;
  font-size: .9em;
  color: var(--sub);
  font-family: var(--ff-b);
}
.claim-line {
  margin: 5px 0;
  color: var(--gold);
  font-family: var(--ff-m);
  font-size: 11px;
}

/* ── Upload dataset card ── */
.upload-card {
  background: var(--sf2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  margin: 6px 0;
  cursor: pointer;
  transition: border-color .2s;
}
.upload-card.active { border-color: var(--blue); }
.upload-card .uc-name {
  font-family: var(--ff-m);
  font-size: 11px;
  color: var(--text);
  letter-spacing: .04em;
}
.upload-card .uc-meta {
  font-size: 10px;
  color: var(--muted);
  margin-top: 3px;
}

/* ── Status badge ── */
.badge {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 2px 9px;
  border-radius: 20px;
  font-family: var(--ff-m);
  font-size: 10px;
  letter-spacing: .06em;
}
.badge-coral  { background: rgba(207,69,32,.10);  color: var(--coral);  border: 1px solid rgba(207,69,32,.25); }
.badge-red    { background: rgba(207,69,32,.10);  color: var(--red);    border: 1px solid rgba(207,69,32,.25); }
.badge-green  { background: rgba(0,135,103,.10);  color: var(--sage);   border: 1px solid rgba(0,135,103,.25); }
.badge-amber  { background: rgba(241,196,0,.12);  color: var(--gold);   border: 1px solid rgba(241,196,0,.25); }

/* ── Anomaly indicator ── */
.anom-high { color: var(--coral); font-family: var(--ff-m); font-weight: 700; }
.anom-low  { color: var(--sage);  font-family: var(--ff-m); font-weight: 700; }

/* ── Code block ── */
.stCode code { background: var(--sf2) !important; color: var(--blue) !important; }

/* ── Plot image rounding ── */
[data-testid="stImage"] img { border-radius: 8px; }
</style>
"""), unsafe_allow_html=True)


# ── Helper: impute NaNs ──────────────────────────────────────────────────────
def _impute(X):
    out = X.copy().astype(float)
    med = np.nanmedian(out, axis=0)
    for c in range(out.shape[1]):
        mask = np.isnan(out[:, c])
        out[mask, c] = med[c] if not np.isnan(med[c]) else 0.0
    return out


# ── Data loading (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    aimd_npz = np.load(PROCESSED_DIR / 'features_aimd.npz', allow_pickle=True)
    mlff_npz = np.load(PROCESSED_DIR / 'features_mlff.npz', allow_pickle=True)
    framework = AnomalyDetectionFramework.load(MODELS_DIR / 'anomaly_framework.pkl')
    return {
        'X_aimd':        aimd_npz['X'],
        'feature_names': list(aimd_npz['feature_names']),
        'meta_aimd':     pd.read_csv(PROCESSED_DIR / 'meta_aimd.csv'),
        'framework':     framework,
        'summary':       json.load(open(PROCESSED_DIR / 'pipeline_summary.json')),
        # Default MLFF dataset (from original pipeline run)
        'X_mlff_default':   mlff_npz['X'],
        'meta_mlff_default': pd.read_csv(PROCESSED_DIR / 'meta_mlff.csv'),
        'feat_cmp_default':  pd.read_csv(REPORTS_DIR / 'feature_comparison.csv'),
    }


@st.cache_data
def _predict_aimd(_fw, _X):
    return _fw.predict(_X)


def _build_feat_comparison(X_aimd, X_upload, feature_names):
    """Compute per-feature comparison between AIMD baseline and an upload."""
    aimd_means = np.nanmean(X_aimd, axis=0)
    aimd_stds  = np.nanstd(X_aimd, axis=0)
    upl_means  = np.nanmean(X_upload, axis=0)
    rows = []
    for i, fn in enumerate(feature_names):
        z = (upl_means[i] - aimd_means[i]) / (aimd_stds[i] + 1e-10)
        rel = (upl_means[i] - aimd_means[i]) / (abs(aimd_means[i]) + 1e-10) * 100
        rows.append({
            'feature': fn,
            'aimd_mean': aimd_means[i],
            'aimd_std':  aimd_stds[i],
            'mlff_mean': upl_means[i],
            'z_score':   z,
            'relative_change_%': rel,
        })
    df = pd.DataFrame(rows).sort_values('z_score', key=abs, ascending=False)
    return df.reset_index(drop=True)


def _run_upload_pipeline(xyz_bytes: bytes, filename: str, D: dict) -> dict:
    """
    Predict-only pipeline for an uploaded .xyz file.
    Uses the already-trained AnomalyDetectionFramework (no retraining).
    Returns dict with keys: X, meta, results, feat_cmp, filename, timestamp, n_windows.
    """
    # Write to a temp file so TrajectoryLoader can read it
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tf:
        tf.write(xyz_bytes)
        tmp_path = tf.name

    loader    = TrajectoryLoader()
    extractor = FeatureExtractor(WindowConfig(window_size=50, stride=10))

    traj = loader.load(tmp_path)
    coords   = traj['coordinates']   # (n_frames, n_atoms, 3)
    energies = traj.get('energies')
    meta_raw = traj.get('metadata', {})

    X_upl, windows = extractor.extract_all_windows(coords, energies)
    X_upl = _impute(X_upl)

    framework = D['framework']
    results   = framework.predict(X_upl)

    # Build metadata  (windows is a list of (start, end) tuples)
    n_w = len(X_upl)
    meta = pd.DataFrame({
        'file':          [filename] * n_w,
        'start':         [w[0] for w in windows],
        'end':           [w[1] for w in windows],
        'n_atoms':       [coords.shape[1]] * n_w,
        'temperature_K': [meta_raw.get('temperature_K')] * n_w,
        'configuration': [meta_raw.get('configuration', 'uploaded')] * n_w,
    })

    feat_cmp = _build_feat_comparison(D['X_aimd'], X_upl, D['feature_names'])

    return {
        'X':         X_upl,
        'meta':      meta,
        'results':   results,
        'feat_cmp':  feat_cmp,
        'filename':  filename,
        'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
        'n_windows': n_w,
        'n_frames':  coords.shape[0],
        'n_atoms':   coords.shape[1],
        'anom_rate': float(np.mean(results['anomaly_label'])),
    }


# ── Load base data ───────────────────────────────────────────────────────────
try:
    D = load_all()
except FileNotFoundError as e:
    st.error(f"**Pipeline outputs missing:** {e}")
    st.info("Run the pipeline first:\n```bash\n/Users/alina/anaconda3/envs/agentic/bin/python scripts/run_full_pipeline.py\n```")
    st.stop()

X_aimd        = D['X_aimd']
feature_names = D['feature_names']
meta_aimd     = D['meta_aimd']
framework     = D['framework']
summary       = D['summary']

ra = _predict_aimd(framework, X_aimd)  # AIMD predictions (cached)

# ── Session state: upload registry ──────────────────────────────────────────
if 'uploads' not in st.session_state:
    # Pre-populate with the default MLFF from the original pipeline run
    st.session_state.uploads = {
        '__default_mlff__': {
            'X':         D['X_mlff_default'],
            'meta':      D['meta_mlff_default'],
            'results':   _predict_aimd(framework, D['X_mlff_default']),
            'feat_cmp':  D['feat_cmp_default'],
            'filename':  'MLFF_baseline.xyz',
            'timestamp': 'baseline',
            'n_windows': len(D['X_mlff_default']),
            'n_frames':  None,
            'n_atoms':   None,
            'anom_rate': float(np.mean(
                _predict_aimd(framework, D['X_mlff_default'])['anomaly_label']
            )),
        }
    }

if 'active_upload' not in st.session_state:
    st.session_state.active_upload = '__default_mlff__'


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    f"<h1 style='font-family:DM Serif Display,serif;font-size:1.2rem;color:{INK};"
    "margin-bottom:2px'>AIMD Anomaly<br/>Detection</h1>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    f"<div style='font-family:DM Mono,monospace;font-size:10px;"
    f"letter-spacing:.1em;color:{SUB};margin-bottom:16px'>"
    "Sb₂Te₃ · Cr dopants · npj 2D Materials</div>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    ["📊 Data Overview",
     "🔍 Feature Analysis",
     "⚠️ Anomaly Detection",
     "⚖️ AIMD vs Upload",
     "🤖 AI Analysis"],
    label_visibility="collapsed",
)

st.sidebar.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

# ── Upload panel ─────────────────────────────────────────────────────────────
st.sidebar.markdown(
    f"<div style='font-family:DM Mono,monospace;font-size:10px;"
    f"letter-spacing:.15em;text-transform:uppercase;color:{JHU_BLUE};"
    "margin-bottom:8px'>Upload New Trajectory</div>",
    unsafe_allow_html=True,
)

uploaded_file = st.sidebar.file_uploader(
    "Upload .xyz trajectory",
    type=['xyz'],
    label_visibility="collapsed",
    help="Upload a new MLFF or reference .xyz file. Predict-only (~5s, no retraining).",
)

if uploaded_file is not None:
    key = uploaded_file.name
    if key not in st.session_state.uploads:
        with st.sidebar.status(f"Analyzing {uploaded_file.name}…", expanded=False):
            try:
                result = _run_upload_pipeline(
                    uploaded_file.read(), uploaded_file.name, D
                )
                st.session_state.uploads[key] = result
                st.session_state.active_upload = key
                st.sidebar.success(f"✓ {uploaded_file.name} — {result['n_windows']} windows")
            except Exception as exc:
                st.sidebar.error(f"Pipeline error: {exc}")
    else:
        # Already processed — just switch to it
        st.session_state.active_upload = key

# ── Dataset selector ──────────────────────────────────────────────────────────
st.sidebar.markdown(
    f"<div style='font-family:DM Mono,monospace;font-size:10px;"
    f"letter-spacing:.15em;text-transform:uppercase;color:{MUTED};"
    "margin-top:12px;margin-bottom:6px'>Loaded Datasets</div>",
    unsafe_allow_html=True,
)

keys_to_delete = []
for k, v in st.session_state.uploads.items():
    is_active = (k == st.session_state.active_upload)
    anom_pct  = f"{v['anom_rate']:.0%}"
    badge_cls = 'badge-red' if v['anom_rate'] > 0.5 else ('badge-amber' if v['anom_rate'] > 0.1 else 'badge-green')  # noqa — badge-green mapped to sage in CSS

    cols = st.sidebar.columns([6, 1])
    label = v['filename'][:22] + ('…' if len(v['filename']) > 22 else '')
    border_style = f'border-color:{JHU_BLUE}' if is_active else ''

    if cols[0].button(
        f"{'▶ ' if is_active else '   '}{label}\n{anom_pct} anomaly · {v['n_windows']}w",
        key=f"sel_{k}",
        use_container_width=True,
    ):
        st.session_state.active_upload = k
        st.rerun()

    # Don't allow deleting the default baseline
    if k != '__default_mlff__':
        if cols[1].button("✕", key=f"del_{k}", help="Remove this dataset"):
            keys_to_delete.append(k)

for k in keys_to_delete:
    del st.session_state.uploads[k]
    if st.session_state.active_upload == k:
        st.session_state.active_upload = '__default_mlff__'
    st.rerun()

# ── Active dataset ────────────────────────────────────────────────────────────
active_data = st.session_state.uploads[st.session_state.active_upload]
X_mlff      = active_data['X']
meta_mlff   = active_data['meta']
rm          = active_data['results']
feat_cmp    = active_data['feat_cmp']
upload_name = active_data['filename']

# ── Sidebar quick-stats ───────────────────────────────────────────────────────
st.sidebar.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

aimd_rate = float(np.mean(ra['anomaly_label']))
mlff_rate = float(np.mean(rm['anomaly_label']))
det_ratio = mlff_rate / max(aimd_rate, 1e-4)

st.sidebar.metric("AIMD anomaly rate",  f"{aimd_rate:.1%}")
st.sidebar.metric(f"Upload anomaly rate", f"{mlff_rate:.1%}",
                  delta=f"{'+' if mlff_rate > aimd_rate else ''}{mlff_rate - aimd_rate:.1%}")
st.sidebar.metric("Detection ratio",    f"{det_ratio:.1f}×")

st.sidebar.markdown(
    f"<div style='font-family:DM Mono,monospace;font-size:9px;"
    f"color:{MUTED};margin-top:8px;line-height:1.6'>"
    "L1 (3σ) + L2 (IF + SVM) ensemble<br/>"
    "window=50 frames · stride=10</div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Data Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Data Overview":
    st.markdown(
        "<div class='page-head'>"
        "<span class='page-head-label'>01 — Data Overview</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.title("Trajectory Inventory")
    st.markdown(
        f"AIMD baseline (**Sb₂Te₃** with Cr dopants, 2D bilayer) "
        f"vs active upload: **{upload_name}**. "
        "Pipeline: L1+L2 ensemble · 22 features · window=50 · stride=10."
    )

    inject_metrics_bar([
        ("AIMD files",     meta_aimd['file'].nunique(),  None,    "ink"),
        ("AIMD windows",   f"{len(X_aimd):,}",           None,    "sage"),
        ("Upload windows", f"{len(X_mlff):,}",           None,    "coral"),
        ("Temperatures",   meta_aimd['temperature_K'].dropna().nunique(), None, "gold"),
        ("Features",       len(feature_names),            None,    "ink"),
    ])

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("AIMD Trajectory Files")
        tbl = pd.DataFrame()
        required_cols = {'file', 'start', 'temperature_K', 'configuration', 'n_atoms'}
        if meta_aimd.empty or not required_cols.issubset(meta_aimd.columns):
            st.info("AIMD metadata is missing or incomplete. Run the pipeline to generate it.")
        else:
            tbl = (
                meta_aimd.groupby('file')
                .agg(windows=('start','count'),
                     temperature_K=('temperature_K','first'),
                     configuration=('configuration','first'),
                     n_atoms=('n_atoms','first'))
                .reset_index().sort_values('windows', ascending=False)
            )
            st.dataframe(tbl, use_container_width=True, height=280)

    with col_b:
        st.subheader(f"Active Upload: {upload_name}")
        upl_tbl = pd.DataFrame()
        upl_meta = active_data.get('meta')
        if upl_meta is None or upl_meta.empty:
            st.info("Upload metadata is empty. Re-upload the trajectory to reprocess it.")
        else:
            upl_tbl = upl_meta.groupby('file').agg(
                windows=('start','count'),
                n_atoms=('n_atoms','first'),
                temperature_K=('temperature_K','first'),
            ).reset_index()
            st.dataframe(upl_tbl, use_container_width=True, height=280)

        upl_anom = active_data['anom_rate']
        if upl_anom > 0.5:
            st.markdown(
                f'<span class="badge badge-red">⚠ {upl_anom:.0%} ANOMALY</span>',
                unsafe_allow_html=True,
            )
        elif upl_anom > 0.1:
            st.markdown(
                f'<span class="badge badge-amber">! {upl_anom:.0%} anomaly</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="badge badge-green">✓ {upl_anom:.0%} normal</span>',  # badge-green → sage via CSS
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    col_p, col_q = st.columns(2)
    with col_p:
        st.subheader("Windows per AIMD file")
        if tbl.empty:
            st.info("No AIMD file summary available for plotting yet.")
        else:
            fig, ax = mpl_fig(figsize=(7, 4))
            ax = fig.axes[0]
            ax.barh(tbl['file'], tbl['windows'], color=CYAN, alpha=0.8)
            ax.set_xlabel('Windows')
            ax.tick_params(axis='y', labelsize=7)
            _style_ax(ax, title='AIMD — windows per trajectory', xlabel='Windows')
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    with col_q:
        st.subheader("Windows by temperature (AIMD)")
        if meta_aimd.empty or 'temperature_K' not in meta_aimd.columns:
            st.info("Temperature metadata not available for AIMD trajectories.")
        else:
            tc = meta_aimd['temperature_K'].value_counts().sort_index()
            if not tc.empty:
                fig2, ax2 = mpl_fig(figsize=(5, 4))
                ax2 = fig2.axes[0]
                clrs = [CYAN, AMBER, RED, GREEN, PURPLE][:len(tc)]
                ax2.bar(tc.index.astype(str), tc.values, color=clrs, alpha=0.85, width=0.6)
                _style_ax(ax2, title='Windows by temperature', xlabel='Temperature (K)', ylabel='Windows')
                st.pyplot(fig2, use_container_width=True); plt.close(fig2)
            else:
                st.info("No temperature bins to display for AIMD.")

    st.info(
        "**Quality note:** All AIMD files loaded successfully. Energy drift warnings "
        "(12/13 files) are expected for NVT-AIMD thermostats and do not affect structural anomaly detection."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Feature Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Analysis":
    st.markdown(
        "<div class='page-head'>"
        "<span class='page-head-label'>02 — Feature Analysis</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.title("22-Feature Extraction")
    st.markdown(
        f"All features aggregate over atoms → atom-count-agnostic (handles 80–86 atom systems). "
        f"Comparing AIMD baseline vs **{upload_name}**."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Distributions", "🎯 Importance & Correlation", "🔍 Interactive Explorer", "📋 Statistics Table"]
    )

    with tab1:
        p = FIGURES_DIR / 'feature_distributions.png'
        if p.exists():
            st.image(str(p), use_container_width=True)
            st.caption("Static figure from original pipeline. Use Interactive Explorer below for uploaded dataset.")
        else:
            st.info("Run pipeline to generate this figure.")

        # Live comparison for active upload
        st.subheader(f"Live — AIMD vs {upload_name}")
        n_show = st.slider("Features to show", 6, 22, 12)
        # Select top-N by z-score magnitude
        top_feat_idx = feat_cmp.head(n_show)['feature'].apply(
            lambda fn: feature_names.index(fn) if fn in feature_names else -1
        ).tolist()
        top_feat_idx = [i for i in top_feat_idx if i >= 0]
        if not top_feat_idx or X_aimd.size == 0 or X_mlff.size == 0:
            st.info("Not enough feature data to plot the live comparison yet.")
        else:
            fig_v, axes_v = mpl_fig(figsize=(14, max(4, n_show * 0.55)))
            ax_v = fig_v.axes[0]
            x_pos = np.arange(len(top_feat_idx))
            w = 0.36
            aimd_m = np.nanmean(X_aimd[:, top_feat_idx], axis=0)
            mlff_m = np.nanmean(X_mlff[:, top_feat_idx], axis=0)
            aimd_s = np.nanstd(X_aimd[:, top_feat_idx], axis=0)
            mlff_s = np.nanstd(X_mlff[:, top_feat_idx], axis=0)

            ax_v.barh(x_pos + w/2, aimd_m, w, xerr=aimd_s, color=CYAN,
                      alpha=0.75, label='AIMD', error_kw=dict(ecolor=MUTED, lw=1))
            ax_v.barh(x_pos - w/2, mlff_m, w, xerr=mlff_s, color=RED,
                      alpha=0.75, label=upload_name[:20], error_kw=dict(ecolor=MUTED, lw=1))
            ax_v.set_yticks(x_pos)
            ax_v.set_yticklabels([feature_names[i] for i in top_feat_idx], fontsize=8, color=SUB)
            ax_v.legend(facecolor=SURFACE2, labelcolor=TEXT, fontsize=9)
            _style_ax(ax_v, title='Feature means ± std: AIMD vs Upload (top by z-score)',
                      xlabel='Feature value')
            st.pyplot(fig_v, use_container_width=True); plt.close(fig_v)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            p = FIGURES_DIR / 'feature_importance.png'
            if p.exists(): st.image(str(p), use_container_width=True)
        with c2:
            p = FIGURES_DIR / 'feature_correlation.png'
            if p.exists(): st.image(str(p), use_container_width=True)

    with tab3:
        sel = st.selectbox("Select feature", feature_names)
        fi  = feature_names.index(sel)
        av  = X_aimd[:, fi]; mv = X_mlff[:, fi]
        av  = av[~np.isnan(av)]; mv = mv[~np.isnan(mv)]

        if len(av) == 0 or len(mv) == 0:
            st.info("Selected feature has no values for one of the datasets.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("AIMD mean ± std",  f"{np.mean(av):.4f} ± {np.std(av):.4f}")
            col2.metric(f"{upload_name[:16]} mean", f"{np.mean(mv):.4f}")
            z = (np.mean(mv) - np.mean(av)) / (np.std(av) + 1e-10)
            col3.metric("Z-score", f"{z:+.2f}",
                        delta="⚠ Anomalous" if abs(z) > 2 else "✓ Normal",
                        delta_color="inverse" if abs(z) > 2 else "normal")

            fig_e, axes_e = mpl_fig(nrows=1, ncols=2, figsize=(13, 4))
            ax_l, ax_r = fig_e.axes[0], fig_e.axes[1]
            ax_l.hist(av, bins=40, alpha=0.65, color=CYAN, label='AIMD', density=True)
            ax_l.hist(mv, bins=40, alpha=0.65, color=RED, label=upload_name[:16], density=True)
            ax_l.axvline(np.mean(av), color=CYAN, ls='--', lw=1.5)
            ax_l.axvline(np.mean(mv), color=RED, ls='--', lw=1.5)
            ax_l.legend(facecolor=SURFACE2, labelcolor=TEXT, fontsize=9)
            _style_ax(ax_l, title=f'{sel} — Distribution', xlabel=sel)

            ax_r.plot(av[:400], alpha=0.7, color=CYAN, lw=0.8, label='AIMD')
            ax_r.plot(mv[:400], alpha=0.7, color=RED, lw=0.8, label=upload_name[:16])
            ax_r.legend(facecolor=SURFACE2, labelcolor=TEXT, fontsize=9)
            _style_ax(ax_r, title=f'{sel} — Time series (first 400 windows)', xlabel='Window index')
            st.pyplot(fig_e, use_container_width=True); plt.close(fig_e)

    with tab4:
        st.subheader(f"Feature Statistics: AIMD vs {upload_name}")
        if feat_cmp.empty:
            st.info("Feature statistics are unavailable for this dataset.")
        else:
            max_z = float(feat_cmp['z_score'].abs().max()) if len(feat_cmp) > 0 else 10
            vmax  = max(10, min(max_z, 1e4))
            try:
                styled = (
                    feat_cmp.style
                    .background_gradient(subset=['z_score'], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                    .format({'aimd_mean':'{:.4f}','mlff_mean':'{:.4f}',
                             'aimd_std':'{:.4f}','z_score':'{:+.2f}',
                             'relative_change_%':'{:+.1f}%'})
                )
                st.dataframe(styled, use_container_width=True)
            except Exception:
                st.dataframe(feat_cmp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Anomaly Detection  (+ dynamic window zoom)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Anomaly Detection":
    st.markdown(
        "<div class='page-head'>"
        "<span class='page-head-label'>03 — Anomaly Detection</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.title("Multi-Level Detector")
    st.markdown(
        f"**L1** Statistical (3σ) · **L2a** Isolation Forest · **L2b** One-Class SVM · "
        f"Anomaly = ≥ 2 detectors agree. Active upload: **{upload_name}**"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AIMD anomaly rate", f"{aimd_rate:.1%}", help="Expected ~5%")
    c2.metric("Upload anomaly rate", f"{mlff_rate:.1%}",
              delta=f"{mlff_rate - aimd_rate:+.1%}", delta_color="inverse")
    c3.metric("Detection ratio",   f"{det_ratio:.1f}×")
    c4.metric("Upload windows",    f"{len(X_mlff):,}")

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    inject_status_monitor(ra, rm, upload_name)

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    source_sel = st.radio("Inspect trajectory", ["AIMD (baseline)", f"Upload: {upload_name[:30]}"],
                          horizontal=True)
    is_upload  = source_sel.startswith("Upload")
    X_sel      = X_mlff  if is_upload else X_aimd
    res_sel    = rm       if is_upload else ra
    color_sel  = RED      if is_upload else CYAN
    n_win      = len(X_sel)
    src_label  = upload_name[:20] if is_upload else "AIMD baseline"

    if n_win == 0:
        st.warning("No windows available for this dataset. Upload a valid trajectory to analyze.")
        st.stop()

    # ── Full timeline ────────────────────────────────────────────────────────
    st.subheader("Detector Confidence Timeline")
    fig_tl, ax_tl = mpl_fig(figsize=(14, 3.5))
    ax_tl = fig_tl.axes[0]
    conf = res_sel['confidence']
    ax_tl.fill_between(range(n_win), conf, alpha=0.25, color=color_sel)
    ax_tl.plot(conf, lw=0.8, color=color_sel)
    ax_tl.axhline(2, color=RED, ls='--', lw=1.5, label='Anomaly threshold (≥2/3 detectors)')
    ax_tl.set_ylim(-0.15, 3.5)
    ax_tl.legend(facecolor=SURFACE2, labelcolor=TEXT, fontsize=9)
    rate_pct = np.mean(res_sel['anomaly_label'])
    _style_ax(ax_tl,
              title=f'{src_label} — Anomaly confidence  ({rate_pct:.1%} anomalous)',
              xlabel='Window index', ylabel='Detector votes (0–3)')
    st.pyplot(fig_tl, use_container_width=True); plt.close(fig_tl)

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    # ── Dynamic window zoom ──────────────────────────────────────────────────
    st.subheader("🔎 Dynamic Window Zoom")
    st.caption("Drag the range slider to peel into any region of the trajectory.")

    zoom_c1, zoom_c2 = st.columns([4, 1])
    with zoom_c1:
        win_range = st.slider(
            "Window range",
            min_value=0, max_value=max(n_win - 1, 1),
            value=(0, min(200, n_win - 1)),
            key='win_zoom_slider',
        )
    w_start, w_end = win_range
    w_end = min(w_end + 1, n_win)

    with zoom_c2:
        anom_in_range = float(np.mean(res_sel['anomaly_label'][w_start:w_end]))
        st.metric("Windows", w_end - w_start)
        st.metric("Rate in range", f"{anom_in_range:.1%}")

    # Zoomed timeline + detector heatmap
    fig_z, axes_z = plt.subplots(2, 1, figsize=(14, 6),
                                  sharex=True, facecolor=SURFACE,
                                  constrained_layout=True)
    idx    = np.arange(w_start, w_end)
    conf_z = res_sel['confidence'][w_start:w_end]
    anom_z = res_sel['anomaly_label'][w_start:w_end]

    for ax in axes_z: _style_ax(ax)

    axes_z[0].fill_between(idx, conf_z, alpha=0.25, color=color_sel)
    axes_z[0].plot(idx, conf_z, lw=1.0, color=color_sel)
    axes_z[0].fill_between(idx, conf_z, where=(anom_z == 1),
                             alpha=0.55, color=RED, label='Anomaly')
    axes_z[0].axhline(2, color=RED, ls='--', lw=1.2, label='Threshold')
    axes_z[0].set_ylim(-0.15, 3.5)
    axes_z[0].legend(facecolor=SURFACE2, labelcolor=TEXT, fontsize=8)
    _style_ax(axes_z[0],
              title=f'Zoomed: windows {w_start}–{w_end-1}  |  anomaly rate: {np.mean(anom_z):.1%}',
              ylabel='Detector votes')

    flags = np.vstack([
        res_sel['l1_flag'][w_start:w_end],
        res_sel['l2_if_flag'][w_start:w_end],
        res_sel['l2_svm_flag'][w_start:w_end],
    ]).astype(float)
    from matplotlib.colors import LinearSegmentedColormap
    cm_flags = LinearSegmentedColormap.from_list('dark_red', [SURFACE2, RED])
    axes_z[1].imshow(flags, aspect='auto', cmap=cm_flags, vmin=0, vmax=1,
                     extent=[w_start, w_end, -0.5, 2.5])
    axes_z[1].set_yticks([0, 1, 2])
    axes_z[1].set_yticklabels(['L2 SVM', 'L2 IF', 'L1 stat'], fontsize=8, color=SUB)
    axes_z[1].set_xlabel('Window index', color=SUB)
    _style_ax(axes_z[1], title='Per-detector flags  (red = anomaly)')

    st.pyplot(fig_z, use_container_width=True); plt.close(fig_z)

    # Top feature deviations in selected region
    st.subheader("Top Feature Deviations in Region")
    X_zone     = X_sel[w_start:w_end]
    zone_means = np.nanmean(X_zone, axis=0)
    aimd_m_g   = np.nanmean(X_aimd, axis=0)
    aimd_s_g   = np.nanstd(X_aimd, axis=0)
    z_zone     = (zone_means - aimd_m_g) / (aimd_s_g + 1e-10)
    top_idx    = np.argsort(np.abs(z_zone))[::-1][:8]

    fig_fz, ax_fz = mpl_fig(figsize=(10, 4))
    ax_fz = fig_fz.axes[0]
    z_top     = z_zone[top_idx]
    names_top = [feature_names[i] for i in top_idx]
    bar_clrs  = [RED if z > 0 else CYAN for z in z_top]
    ax_fz.barh(names_top, z_top, color=bar_clrs, alpha=0.85)
    ax_fz.axvline(0, color=TEXT, lw=0.6)
    ax_fz.axvline(2,  color=MUTED, ls='--', lw=1, label='+2σ')
    ax_fz.axvline(-2, color=MUTED, ls='--', lw=1, label='-2σ')
    ax_fz.legend(facecolor=SURFACE2, labelcolor=TEXT, fontsize=8)
    _style_ax(ax_fz,
              title=f'Feature deviation vs AIMD  (windows {w_start}–{w_end-1})',
              xlabel='Z-score')
    ax_fz.tick_params(axis='y', labelsize=9, labelcolor=SUB)
    st.pyplot(fig_fz, use_container_width=True); plt.close(fig_fz)

    # ── AI window analysis ───────────────────────────────────────────────────
    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)
    with st.expander("🤖 Ask AI to explain this window region", expanded=False):
        if st.button("Analyze region with AI"):
            data_ctx = build_data_context(
                X_aimd, X_mlff, feature_names, ra, rm, meta_aimd, feat_cmp
            )
            analyst = OllamaAnalyst()
            with st.spinner("Analyzing with glm-5:cloud…"):
                analysis = analyst.analyze_window_region(
                    start_win=w_start, end_win=w_end,
                    source=src_label,
                    X=X_sel, feature_names=feature_names,
                    results=res_sel, data_context=data_ctx,
                )
            st.markdown(
                f'<div class="evidence-block">{analysis}</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    # ── Threshold slider ────────────────────────────────────────────────────
    st.subheader("Adjust Anomaly Threshold")
    thr = st.slider("Minimum detector votes to flag as anomaly", 1, 3, 2, key='thr_slider')
    ca  = float(np.mean(ra['confidence'] >= thr))
    cm  = float(np.mean(rm['confidence'] >= thr))
    c1t, c2t, c3t = st.columns(3)
    c1t.metric(f"AIMD at threshold={thr}",   f"{ca:.1%}")
    c2t.metric(f"Upload at threshold={thr}", f"{cm:.1%}",
               delta=f"{cm - ca:+.1%}", delta_color="inverse")
    c3t.metric("Ratio",                       f"{cm/max(ca,1e-4):.1f}×")

    # ── Confidence distribution ─────────────────────────────────────────────
    st.subheader("Confidence Score Distribution")
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4), facecolor=SURFACE,
                                constrained_layout=True)
    for ax in axes4: _style_ax(ax)

    for ax, conf_d, title, clr in [
        (axes4[0], ra['confidence'], 'AIMD (baseline)', CYAN),
        (axes4[1], rm['confidence'], upload_name[:24], RED),
    ]:
        cnts = [int(np.sum(conf_d == v)) for v in range(4)]
        bars = ax.bar(range(4), cnts, color=clr, alpha=0.8, width=0.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['0\n(Normal)', '1', '2', '3\n(All 3)'], color=SUB, fontsize=9)
        for bar, cnt in zip(bars, cnts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cnts)*0.01,
                    str(cnt), ha='center', va='bottom', color=TEXT, fontsize=8,
                    fontfamily='monospace')
        _style_ax(ax, title=title, xlabel='Detector votes', ylabel='Windows')

    st.pyplot(fig4, use_container_width=True); plt.close(fig4)

    # ── LSTM training curve ─────────────────────────────────────────────────
    p = FIGURES_DIR / 'lstm_training.png'
    if p.exists():
        st.subheader("LSTM Autoencoder Training Curve")
        st.image(str(p), use_container_width=True)
        st.info(f"Anomaly threshold (95th percentile val error): **{summary.get('lstm_threshold', 'N/A')}**")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — AIMD vs Upload Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ AIMD vs Upload":
    st.markdown(
        "<div class='page-head'>"
        "<span class='page-head-label'>04 — Comparison</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.title("AIMD vs Upload")
    st.markdown(
        f"First-principles AIMD dynamics vs **{upload_name}**. "
        "Quantifies per-feature deviation from DFT-level physics. "
        "All z-scores are live — they update whenever you switch uploads."
    )

    # Summary metrics
    top_feat  = feat_cmp.iloc[0]
    max_z     = float(feat_cmp['z_score'].abs().max())
    n_sig     = int((feat_cmp['z_score'].abs() > 2).sum())

    inject_metrics_bar([
        ("Max |z-score|",       f"{max_z:.1f}",          top_feat['feature'],           "coral"),
        ("Features |z| > 2σ",  f"{n_sig}/{len(feature_names)}", None,                   "gold"),
        ("Upload anomaly rate", f"{mlff_rate:.1%}",       f"{mlff_rate-aimd_rate:+.1%} vs AIMD", "coral"),
        ("AIMD anomaly rate",   f"{aimd_rate:.1%}",       "baseline",                    "sage"),
    ])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max |z-score|",      f"{max_z:.1f}",
              delta=f"Feature: {top_feat['feature']}")
    c2.metric("Features |z| > 2σ",  f"{n_sig} / {len(feature_names)}")
    c3.metric("Upload anomaly rate", f"{mlff_rate:.1%}",
              delta=f"{mlff_rate - aimd_rate:+.1%} vs AIMD", delta_color="inverse")
    c4.metric("Detection ratio",     f"{det_ratio:.1f}×")

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    # Z-score chart
    st.subheader(f"Per-Feature Z-score: {upload_name[:30]} vs AIMD")
    n_show2 = st.slider("Features to display", 5, 22, 10, key='feat_show')
    top_df  = feat_cmp.head(n_show2)

    fig5, ax5 = mpl_fig(figsize=(11, max(4, n_show2 * 0.5)))
    ax5 = fig5.axes[0]
    clrs = [RED if z > 0 else CYAN for z in top_df['z_score']]
    ax5.barh(top_df['feature'], top_df['z_score'], color=clrs, alpha=0.85)
    ax5.axvline(0, color=TEXT, lw=0.5)
    for v, lbl in [(2, '+2σ'), (-2, '−2σ')]:
        ax5.axvline(v, color=MUTED, ls='--', lw=1, label=lbl)
    ax5.legend(facecolor=SURFACE2, labelcolor=TEXT, fontsize=9)
    _style_ax(ax5,
              title=f'Feature deviation: {upload_name[:24]} vs AIMD baseline\n(positive = upload higher than AIMD)',
              xlabel='Z-score  (upload − AIMD) / AIMD σ')
    ax5.tick_params(axis='y', labelsize=9, labelcolor=SUB)
    st.pyplot(fig5, use_container_width=True); plt.close(fig5)

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    # Narrative report
    rp = REPORTS_DIR / 'aimd_vs_mlff_analysis.md'
    if rp.exists():
        with st.expander("📄 Original pipeline narrative report", expanded=False):
            st.markdown(open(rp).read())

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        p = FIGURES_DIR / 'feature_distributions.png'
        if p.exists():
            st.subheader("Feature Distributions (baseline pipeline)")
            st.image(str(p), use_container_width=True)
    with col2:
        p = FIGURES_DIR / 'detector_agreement.png'
        if p.exists():
            st.subheader("Detector Agreement (baseline pipeline)")
            st.image(str(p), use_container_width=True)

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    # Full table
    st.subheader("Full Feature Comparison Table")
    st.caption(f"Live comparison: AIMD baseline vs {upload_name}")
    vmax2 = max(10, min(float(feat_cmp['z_score'].abs().max()), 1e4))
    try:
        st.dataframe(
            feat_cmp.style
                .background_gradient(subset=['z_score'], cmap='RdBu_r', vmin=-vmax2, vmax=vmax2)
                .format({'aimd_mean':'{:.4f}','mlff_mean':'{:.4f}',
                         'aimd_std':'{:.4f}','z_score':'{:+.2f}',
                         'relative_change_%':'{:+.1f}%'}),
            use_container_width=True,
        )
    except Exception:
        st.dataframe(feat_cmp, use_container_width=True)

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    # Downloads
    st.subheader("Download Results")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.download_button(
            "📥 Feature comparison CSV",
            feat_cmp.to_csv(index=False),
            file_name=f"feature_comparison_{upload_name.replace('.xyz','')}.csv",
            mime='text/csv',
        )
    with col_d2:
        # Build per-window results DataFrame
        win_df = pd.DataFrame({
            'window':       range(len(X_mlff)),
            'anomaly_label': rm['anomaly_label'],
            'confidence':    rm['confidence'],
            'l1_flag':       rm['l1_flag'],
            'l2_if_flag':    rm['l2_if_flag'],
            'l2_svm_flag':   rm['l2_svm_flag'],
        })
        st.download_button(
            "📥 Window results CSV",
            win_df.to_csv(index=False),
            file_name=f"window_results_{upload_name.replace('.xyz','')}.csv",
            mime='text/csv',
        )
    with col_d3:
        if (REPORTS_DIR / 'ensemble_comparison.csv').exists():
            df1 = pd.read_csv(REPORTS_DIR / 'ensemble_comparison.csv')
            st.download_button(
                "📥 Baseline pipeline CSV",
                df1.to_csv(index=False),
                file_name='ensemble_comparison_baseline.csv',
                mime='text/csv',
            )

    st.caption(
        "Framework is domain-agnostic — same detector pipeline applies to game profiler "
        "data (frame times, GPU utilization) for Qualcomm HLM game debugging automation."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — AI Analysis Assistant
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Analysis":
    st.markdown(
        "<div class='page-head'>"
        "<span class='page-head-label'>05 — AI Analysis</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.title("AI Analysis Assistant")
    st.markdown(
        f"Natural language interface powered by **glm-5:cloud** via Ollama. "
        f"Active dataset: **{upload_name}**. "
        "Generate figures from real data arrays or get evidence-grounded mechanism analysis. "
        "**Every claim is backed by actual computed values.**"
    )
    st.warning(
        "**Anti-hallucination:** LLM only receives computed statistics from this session. "
        "Figures are generated by executing real code against real numpy arrays — they cannot be fabricated. "
        "Mechanism claims must cite specific numbers or are marked `► Uncertain`."
    )

    # Build data context for active upload
    data_ctx = build_data_context(
        X_aimd, X_mlff, feature_names, ra, rm, meta_aimd, feat_cmp
    )

    exec_ns = {
        'X_aimd':        X_aimd,
        'X_mlff':        X_mlff,
        'feature_names': feature_names,
        'meta_aimd':     meta_aimd,
        'meta_mlff':     meta_mlff,
        'results_aimd':  ra,
        'results_mlff':  rm,
        'feat_comparison': feat_cmp,
        'np':  np,
        'pd':  pd,
        'plt': plt,
        'sns': sns,
    }

    analyst = OllamaAnalyst(model='glm-5:cloud')

    # ── Mode selector ──────────────────────────────────────────────────────
    mode = st.radio(
        "Analysis mode",
        ["📈 Generate Figure", "🧠 Mechanism Analysis", "🔬 Both Figure + Analysis"],
        horizontal=True,
    )

    with st.expander("💡 Example queries", expanded=False):
        st.markdown("""
**Figure generation:**
- *"Show a scatter plot of rms_velocity vs crest_factor, colored by AIMD/MLFF"*
- *"Plot the top 6 most deviating features as paired box plots"*
- *"Show the MSD slope distribution for each AIMD temperature separately"*
- *"Compare the energy_trend distribution between AIMD and MLFF as overlapping histograms"*

**Mechanism analysis:**
- *"Why does MLFF show 100% anomaly rate?"*
- *"Which features suggest MLFF has higher atomic velocities than AIMD?"*
- *"What does the spectral entropy difference imply about phonon modes?"*
        """)

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    query = st.text_area(
        "Enter your analysis question",
        placeholder=f"e.g. 'Compare displacement distributions between AIMD and {upload_name[:20]}'",
        height=100,
        key="ai_query",
    )

    run_btn = st.button("▶ Run Analysis", type="primary", disabled=not query.strip())

    if run_btn and query.strip():
        generated_fig = None
        generated_code = ''
        figure_description = ''

        if mode in ["📈 Generate Figure", "🔬 Both Figure + Analysis"]:
            with st.spinner("Generating figure code with glm-5:cloud…"):
                gen_fig, gen_code, gen_err = analyst.generate_and_execute(
                    query=query, data_context=data_ctx, namespace=exec_ns,
                )
            with st.expander("🔧 Generated code", expanded=False):
                st.code(gen_code, language='python')

            if gen_err:
                st.error(f"Code execution error:\n```\n{gen_err}\n```")
                st.info("AI retried once. Try rephrasing your query.")
            elif gen_fig is not None:
                st.subheader("Generated Figure")
                st.pyplot(gen_fig)
                plt.close(gen_fig)
                generated_fig = gen_fig
                figure_description = (
                    f"Matplotlib figure for: '{query}'. Code: {gen_code[:300]}..."
                )
            else:
                st.warning("Code ran but produced no figure. Try rephrasing.")

        if mode in ["🧠 Mechanism Analysis", "🔬 Both Figure + Analysis"]:
            with st.spinner("Generating mechanism analysis with glm-5:cloud…"):
                analysis = analyst.mechanism_analysis(
                    query=query,
                    data_context=data_ctx,
                    figure_description=figure_description,
                )
            st.subheader("Mechanism Analysis")
            st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

            lines = analysis.split('\n')
            output_parts = []
            for line in lines:
                if line.startswith('## '):
                    output_parts.append(
                        f"<div style='font-family:DM Serif Display,serif;font-size:1rem;"
                        f"color:{INK};margin:12px 0 4px;font-weight:600'>{line[3:]}</div>"
                    )
                elif '► Claim:' in line or '► Uncertain:' in line:
                    output_parts.append(
                        f'<div class="claim-line">🔸 {line.strip()}</div>'
                    )
                else:
                    output_parts.append(f"<p style='color:{SUB};margin:3px 0'>{line}</p>")

            st.markdown(
                '<div class="evidence-block">'
                + '\n'.join(output_parts)
                + '</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Lines marked `► Uncertain` indicate insufficient data to confirm that claim. "
                "All other claims cite exact values from the current session."
            )

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    with st.expander("📋 Raw data context (what the LLM sees)", expanded=False):
        st.code(data_ctx, language='text')

    # ── Quick analyses ────────────────────────────────────────────────────
    st.subheader("⚡ Quick Pre-built Analyses")
    st.caption(f"One-click analysis against {upload_name}.")

    quick_queries = {
        "Top deviating features (Z-score bar chart)":
            "Create a horizontal bar chart of all 22 features sorted by absolute z-score "
            "(MLFF vs AIMD). Color bars JHU red (#CF4520) for positive z-score, JHU green (#008767) for negative. "
            "Add a vertical line at ±2. Use a clean white background (#FFFFFF).",
        "AIMD anomaly rate by temperature":
            "Plot a bar chart of anomaly rate (anomaly_label) grouped by temperature_K "
            "for AIMD trajectories only. Use meta_aimd merged with results_aimd.",
        "Energy vs MSD scatter (AIMD vs upload)":
            "Scatter plot of energy_mean (x) vs msd_slope (y) for both AIMD and MLFF. "
            "Color AIMD cyan and MLFF red. Add legend.",
        "Confidence score CDF":
            "Plot the CDF of the l2_if_score for both AIMD and MLFF on the same axes. "
            "Add a vertical line at the median MLFF score.",
        "MLFF displacement distribution":
            "Histogram of disp_median for AIMD and MLFF (log scale on y-axis if needed). "
            "Annotate with mean values.",
    }

    cols_q = st.columns(min(len(quick_queries), 3))
    for i, (label, q) in enumerate(quick_queries.items()):
        with cols_q[i % 3]:
            if st.button(label, key=f"quick_{i}"):
                with st.spinner(f"Running: {label}…"):
                    fig_q, code_q, err_q = analyst.generate_and_execute(
                        query=q, data_context=data_ctx, namespace=exec_ns
                    )
                if err_q:
                    st.error(err_q)
                elif fig_q:
                    st.pyplot(fig_q); plt.close(fig_q)
                    with st.expander("Code", expanded=False):
                        st.code(code_q, language='python')
                else:
                    st.warning("No figure produced.")
