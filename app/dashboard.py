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
import socket
import tempfile
import datetime
import traceback
from pathlib import Path


def _ollama_available() -> bool:
    try:
        s = socket.create_connection(("localhost", 11434), timeout=1)
        s.close()
        return True
    except OSError:
        return False

OLLAMA_AVAILABLE = _ollama_available()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
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
SURFACE2 = '#E8F4F8'  # Light blue background
BORDER_C = '#7FA9C5'  # Medium blue border
CORAL    = JHU_RED
CORAL_LT = JHU_ORANGE
SAGE     = JHU_GREEN
GOLD     = JHU_GOLD
INK      = JHU_BLACK
TEXT     = '#1a1a1a'  # Near-black text
SUB      = '#666666'  # Medium gray for secondary text
MUTED    = '#999999'  # Light gray for muted text

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


def inject_feature_table(df: pd.DataFrame, title: str = '', height: int = 560) -> None:
    """
    Render an interactive, visualized feature-comparison table via components.html().

    Columns: Feature | AIMD mean±σ | Upload mean | Z-score bar | Δ%
    Features:
      - Inline horizontal z-score bar (left=negative, right=positive, severity-colored)
      - Click-to-sort on every column
      - Live filter/search input
      - Row severity tinting (coral/gold/sage) based on |z|
      - Fully matches the warm cream theme (DM Serif Display + DM Mono + DM Sans)
    """
    if df is None or df.empty:
        st.info("Feature statistics are not yet available.")
        return

    # Ensure numeric types
    for col in ['aimd_mean', 'mlff_mean', 'aimd_std', 'z_score', 'relative_change_%']:
        if col in df.columns:
            df = df.copy()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    max_z = max(float(df['z_score'].abs().max()), 1.0) if 'z_score' in df.columns else 10.0

    def _sev(z):
        az = abs(z)
        if az > 5:  return CORAL
        if az > 2:  return GOLD
        return SAGE

    def _sev_cls(z):
        az = abs(z)
        if az > 5:  return 'sev-high'
        if az > 2:  return 'sev-mid'
        return 'sev-low'

    def _z_bar(z):
        az = abs(z)
        pct = min(az / max_z * 100, 100)
        c = _sev(z)
        # Positive z: bar grows right; negative: bar grows left
        if z >= 0:
            bar = (f"<div class='zb-pos' style='width:{pct:.1f}%;background:{c}66;"
                   f"border-right:2px solid {c}'></div>")
        else:
            bar = (f"<div class='zb-neg' style='width:{pct:.1f}%;background:{c}66;"
                   f"border-left:2px solid {c}'></div>")
        label_color = c
        label = f"{z:+.2f}" if abs(z) < 1000 else f"{z:+.0f}"
        return (f"<div class='z-bar-wrap'>"
                f"  <div class='z-bar-track'><div class='z-bar-center'></div>{bar}</div>"
                f"  <span class='z-label' style='color:{label_color}'>{label}</span>"
                f"</div>")

    def _delta_badge(d):
        c = _sev(d / 20)  # scale: ±100% → 5σ equivalent for color
        txt = f"{d:+,.0f}%" if abs(d) >= 1000 else f"{d:+.1f}%"
        return f"<span class='delta-badge' style='color:{c};border-color:{c}40'>{txt}</span>"

    # Sort by |z| descending by default
    df_sorted = df.reindex(df['z_score'].abs().sort_values(ascending=False).index) \
        if 'z_score' in df.columns else df

    uid = str(abs(hash(title + str(len(df)))))[:8]

    rows_html = ''
    for _, row in df_sorted.iterrows():
        feat  = str(row.get('feature', ''))
        am    = float(row.get('aimd_mean', 0))
        as_   = float(row.get('aimd_std', 0))
        mm    = float(row.get('mlff_mean', 0))
        z     = float(row.get('z_score', 0))
        delta = float(row.get('relative_change_%', 0))
        sev   = _sev_cls(z)
        sc    = _sev(z)
        mm_fmt = f"{mm:.4f}" if abs(mm) < 1e4 else f"{mm:,.1f}"
        am_fmt = f"{am:.4f}" if abs(am) < 1e4 else f"{am:,.4f}"
        rows_html += (
            f"<tr class='ft-row {sev}' data-feat='{feat.lower()}' data-z='{z:.4f}'>"
            f"<td class='td-feat'><span class='feat-name'>{feat}</span></td>"
            f"<td class='td-num'><span class='val-main'>{am_fmt}</span>"
            f"<span class='val-std'>±{as_:.4f}</span></td>"
            f"<td class='td-num'><span class='val-main' style='color:{sc}'>{mm_fmt}</span></td>"
            f"<td class='td-bar'>{_z_bar(z)}</td>"
            f"<td class='td-delta'>{_delta_badge(delta)}</td>"
            f"</tr>"
        )

    title_html = f"<div class='ft-title'>{title}</div>" if title else ''

    html = f"""
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: {BG}; }}
  .ft-wrap {{
    font-family: 'DM Sans', sans-serif;
    background: {BG};
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid {BORDER_C};
  }}
  {title_html and f".ft-title {{font-family:'DM Serif Display',serif;font-size:1.1rem;color:{INK};padding:16px 20px 0;}}"}
  .ft-controls {{
    display: flex; align-items: center; gap: 12px;
    padding: 12px 16px; border-bottom: 1px solid {BORDER_C};
    background: {SURFACE};
  }}
  .ft-search-input {{
    flex: 1; font-family: 'DM Mono', monospace; font-size: 11px;
    padding: 6px 12px; border: 1px solid {BORDER_C}; border-radius: 6px;
    background: {SURFACE2}; color: {TEXT}; outline: none;
    letter-spacing: .03em;
    transition: border-color .2s;
  }}
  .ft-search-input:focus {{ border-color: {CORAL}; }}
  .ft-hint {{
    font-family: 'DM Mono', monospace; font-size: 9px;
    letter-spacing: .12em; color: {MUTED}; text-transform: uppercase;
    white-space: nowrap;
  }}
  .ft-table {{
    width: 100%; border-collapse: collapse;
    font-size: 12px; table-layout: fixed;
  }}
  .ft-table thead {{
    background: {SURFACE}; position: sticky; top: 0; z-index: 2;
  }}
  .ft-table th {{
    font-family: 'DM Mono', monospace; font-size: 9px;
    letter-spacing: .12em; text-transform: uppercase;
    color: {MUTED}; padding: 10px 12px; text-align: left;
    border-bottom: 2px solid {BORDER_C};
    cursor: pointer; user-select: none;
    transition: color .15s;
  }}
  .ft-table th:hover {{ color: {CORAL}; }}
  .ft-table th.sorted {{ color: {CORAL}; }}
  .ft-table th:nth-child(1) {{ width: 22%; }}
  .ft-table th:nth-child(2) {{ width: 20%; }}
  .ft-table th:nth-child(3) {{ width: 16%; }}
  .ft-table th:nth-child(4) {{ width: 28%; }}
  .ft-table th:nth-child(5) {{ width: 14%; }}
  .ft-row {{ border-bottom: 1px solid {BORDER_C}; transition: background .12s; }}
  .ft-row:hover {{ background: {SURFACE} !important; }}
  .ft-row.sev-high {{ background: rgba(224,92,58,.04); }}
  .ft-row.sev-mid  {{ background: rgba(200,160,80,.04); }}
  .ft-row.sev-low  {{ background: {BG}; }}
  .ft-table td {{ padding: 9px 12px; vertical-align: middle; }}
  .td-feat .feat-name {{
    font-family: 'DM Mono', monospace; font-size: 11px;
    color: {INK}; letter-spacing: .02em;
  }}
  .td-num {{ text-align: right; }}
  .val-main {{
    font-family: 'DM Mono', monospace; font-size: 11px;
    color: {TEXT}; display: block;
  }}
  .val-std {{
    font-family: 'DM Mono', monospace; font-size: 9px;
    color: {MUTED}; display: block; margin-top: 1px;
  }}
  /* Z-score bar */
  .z-bar-wrap {{
    display: flex; align-items: center; gap: 8px;
  }}
  .z-bar-track {{
    flex: 1; height: 14px; background: {SURFACE2};
    border-radius: 3px; overflow: hidden; position: relative;
  }}
  .z-bar-center {{
    position: absolute; left: 50%; top: 0; bottom: 0;
    width: 1px; background: {BORDER_C}; z-index: 1;
  }}
  .zb-pos {{
    position: absolute; left: 50%; top: 1px; bottom: 1px;
    border-radius: 0 2px 2px 0;
    transition: width .4s ease;
  }}
  .zb-neg {{
    position: absolute; right: 50%; top: 1px; bottom: 1px;
    border-radius: 2px 0 0 2px;
    transition: width .4s ease;
  }}
  .z-label {{
    font-family: 'DM Mono', monospace; font-size: 10px;
    font-weight: 500; min-width: 52px; text-align: right;
    letter-spacing: .02em;
  }}
  /* Δ% badge */
  .td-delta {{ text-align: right; }}
  .delta-badge {{
    font-family: 'DM Mono', monospace; font-size: 10px;
    padding: 2px 7px; border-radius: 4px; border: 1px solid;
    display: inline-block;
  }}
  /* empty state */
  .ft-empty {{
    padding: 24px; text-align: center;
    font-family: 'DM Mono', monospace; font-size: 11px;
    color: {MUTED}; letter-spacing: .08em;
  }}
  /* scrollable body */
  .ft-scroll {{ max-height: {height - 90}px; overflow-y: auto; }}
  .ft-scroll::-webkit-scrollbar {{ width: 6px; }}
  .ft-scroll::-webkit-scrollbar-track {{ background: {SURFACE2}; }}
  .ft-scroll::-webkit-scrollbar-thumb {{ background: {BORDER_C}; border-radius: 3px; }}
</style>

<div class="ft-wrap">
  {title_html}
  <div class="ft-controls">
    <input class="ft-search-input" id="fts-{uid}" type="text"
           placeholder="Filter features…" oninput="ftFilter('{uid}', this.value)" />
    <span class="ft-hint">↕ click header to sort · {len(df)} features</span>
  </div>
  <div class="ft-scroll">
    <table class="ft-table" id="ft-{uid}">
      <thead><tr>
        <th onclick="ftSort('{uid}',0,false)">Feature ↕</th>
        <th onclick="ftSort('{uid}',1,true)">AIMD mean ± σ ↕</th>
        <th onclick="ftSort('{uid}',2,true)">Upload mean ↕</th>
        <th onclick="ftSort('{uid}',3,true)" class="sorted">Z-score ↓</th>
        <th onclick="ftSort('{uid}',4,true)">Δ% ↕</th>
      </tr></thead>
      <tbody id="ftb-{uid}">{rows_html}</tbody>
    </table>
  </div>
</div>

<script>
(function() {{
  var _asc = {{}};

  window.ftFilter = function(uid, q) {{
    q = q.toLowerCase();
    document.querySelectorAll('#ftb-' + uid + ' tr').forEach(function(r) {{
      r.style.display = (r.dataset.feat || '').includes(q) ? '' : 'none';
    }});
  }};

  window.ftSort = function(uid, col, numeric) {{
    var tbody = document.getElementById('ftb-' + uid);
    if (!tbody) return;
    var rows = Array.from(tbody.querySelectorAll('tr'));
    var asc = !_asc[uid + col];
    _asc[uid + col] = asc;

    // Update header highlight
    var ths = document.querySelectorAll('#ft-' + uid + ' thead th');
    ths.forEach(function(th) {{ th.classList.remove('sorted'); }});
    if (ths[col]) ths[col].classList.add('sorted');

    rows.sort(function(a, b) {{
      var av, bv;
      if (col === 3) {{
        av = parseFloat(a.dataset.z || 0);
        bv = parseFloat(b.dataset.z || 0);
        // Sort by abs z descending by default
        if (!_asc[uid + col + '_init']) {{
          _asc[uid + col + '_init'] = true;
          return Math.abs(bv) - Math.abs(av);
        }}
        return asc ? av - bv : bv - av;
      }}
      av = (a.cells[col] ? a.cells[col].textContent : '').trim().replace(/[+±%,]/g,'');
      bv = (b.cells[col] ? b.cells[col].textContent : '').trim().replace(/[+±%,]/g,'');
      var an = parseFloat(av), bn = parseFloat(bv);
      if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
      return asc ? av.localeCompare(bv) : bv.localeCompare(av);
    }});
    rows.forEach(function(r) {{ tbody.appendChild(r); }});
  }};
}})();
</script>
"""
    components.html(html, height=height, scrolling=False)


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


def _parse_lattice_from_comment(comment_line: str):
    """Extract 3×3 lattice matrix from extended-XYZ comment line. Returns (3,3) array or None."""
    import re
    m = re.search(r'Lattice="([^"]+)"', comment_line)
    if m:
        vals = list(map(float, m.group(1).split()))
        if len(vals) == 9:
            return np.array(vals, dtype=float).reshape(3, 3)
    return None


@st.cache_data(show_spinner=False)
def _sample_traj_for_viewer(filepath: str, n_sample: int = 75):
    """
    Load XYZ trajectory, sample n_sample frames evenly.
    Returns (coords_s, species, lattice, orig_indices, n_total_frames).
    orig_indices — array of original frame numbers corresponding to each sampled frame.
    """
    loader = TrajectoryLoader()
    traj   = loader.load(filepath)
    coords   = traj['coordinates']            # (n_frames, n_atoms, 3)
    species  = traj['species']
    n_total  = int(coords.shape[0])
    n_take   = min(n_sample, n_total)
    idx      = np.linspace(0, n_total - 1, n_take, dtype=int)
    coords_s = coords[idx]
    # Parse lattice from first comment line
    lattice = None
    with open(filepath, 'r') as f:
        try:
            int(f.readline().strip())
            lattice = _parse_lattice_from_comment(f.readline())
        except Exception:
            pass
    return coords_s, species, lattice, idx, n_total


def inject_3d_viewer(coords, species, lattice, orig_indices, n_total_frames: int,
                     title: str, height: int = 500) -> None:
    """
    Embed an interactive 3D molecular trajectory viewer using 3Dmol.js.

    coords:         np.array (n_frames, n_atoms, 3) — sampled frames
    species:        list[str] length n_atoms
    lattice:        (3,3) np.array or None — draws unit-cell box
    orig_indices:   array-like of ints, length n_frames — original step numbers
    n_total_frames: int — total frames in the source trajectory (for label)
    """
    import json, uuid
    uid      = uuid.uuid4().hex[:8]
    n_frames = int(coords.shape[0])
    n_atoms  = int(coords.shape[1])

    # ── Compact per-frame position data (3-decimal precision) ─────────────────
    # frames_data[fi][ai] = [x, y, z]
    frames_data = [
        [[round(float(coords[fi, ai, j]), 3) for j in range(3)] for ai in range(n_atoms)]
        for fi in range(n_frames)
    ]
    frames_js    = json.dumps(frames_data)
    sp_list_js   = json.dumps(list(species))
    orig_idx_js  = json.dumps([int(x) for x in orig_indices])

    # ── Unit-cell box JS (12 edges as thin cylinders, added once as shapes) ───
    lattice_js = ""
    if lattice is not None:
        a = lattice.tolist()
        lattice_js = f"""
    (function() {{
      var a1=[{a[0][0]},{a[0][1]},{a[0][2]}],
          a2=[{a[1][0]},{a[1][1]},{a[1][2]}],
          a3=[{a[2][0]},{a[2][1]},{a[2][2]}];
      var corners = [
        [0,0,0],
        a1, a2, a3,
        [a1[0]+a2[0], a1[1]+a2[1], a1[2]+a2[2]],
        [a1[0]+a3[0], a1[1]+a3[1], a1[2]+a3[2]],
        [a2[0]+a3[0], a2[1]+a3[1], a2[2]+a3[2]],
        [a1[0]+a2[0]+a3[0], a1[1]+a2[1]+a3[1], a1[2]+a2[2]+a3[2]]
      ];
      var edges = [[0,1],[0,2],[0,3],[1,4],[1,5],[2,4],[2,6],[3,5],[3,6],[4,7],[5,7],[6,7]];
      edges.forEach(function(e) {{
        viewer.addCylinder({{
          start: {{x:corners[e[0]][0], y:corners[e[0]][1], z:corners[e[0]][2]}},
          end:   {{x:corners[e[1]][0], y:corners[e[1]][1], z:corners[e[1]][2]}},
          radius: 0.06, color: '#778899', fromCap: 1, toCap: 1
        }});
      }});
    }})();"""

    html = f"""
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400&family=DM+Sans:wght@500&display=swap" rel="stylesheet"/>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  #wrap_{uid} {{ font-family:'DM Sans',sans-serif; background:#0d1117;
    border-radius:12px; padding:10px; box-sizing:border-box; }}
  #vdiv_{uid} {{ width:100%; height:{height-76}px; border-radius:8px;
    background:#0d1117; position:relative; overflow:hidden; }}
  .vt_{uid}  {{ font-size:11px; font-weight:600; letter-spacing:.13em;
    text-transform:uppercase; color:#6b8cba; margin-bottom:6px; }}
  .vc_{uid}  {{ display:flex; align-items:center; gap:8px;
    margin-top:8px; flex-wrap:wrap; }}
  .vb_{uid}  {{ background:#1a2744; border:1px solid #2e4470; color:#93b4de;
    font-family:'DM Mono',monospace; font-size:11px; padding:4px 11px;
    border-radius:5px; cursor:pointer; user-select:none; }}
  .vb_{uid}:hover {{ background:#243557; }}
  .vs_{uid}  {{ flex:1; min-width:60px; accent-color:#5b8dd9; cursor:pointer; height:4px; }}
  .vl_{uid}  {{ font-family:'DM Mono',monospace; font-size:10px;
    color:#6b8cba; white-space:nowrap; min-width:110px; }}
  .vsp_{uid} {{ background:#1a2744; border:1px solid #2e4470; color:#6b8cba;
    font-family:'DM Mono',monospace; font-size:10px; padding:2px 6px;
    border-radius:4px; cursor:pointer; }}
</style>
<div id="wrap_{uid}">
  <div class="vt_{uid}">{title}</div>
  <div id="vdiv_{uid}"></div>
  <div class="vc_{uid}">
    <button class="vb_{uid}" id="pbtn_{uid}">▶ Play</button>
    <input  class="vs_{uid}" type="range" id="fslider_{uid}"
            min="0" max="{n_frames-1}" value="0"/>
    <span   class="vl_{uid}" id="flbl_{uid}">step 0 / {n_total_frames}</span>
    <select class="vsp_{uid}" id="fspd_{uid}">
      <option value="200">0.5×</option>
      <option value="120" selected>1×</option>
      <option value="60">2×</option>
      <option value="30">4×</option>
    </select>
    <button class="vb_{uid}" id="rstbtn_{uid}">⟳</button>
  </div>
</div>
<script>
(function() {{
  // ── Data ──────────────────────────────────────────────────────────────────
  var frames   = {frames_js};       // [nFrames][nAtoms][3]
  var spList   = {sp_list_js};      // [nAtoms] element symbols
  var origIdx  = {orig_idx_js};     // [nFrames] original step numbers
  var nFrames  = {n_frames};
  var nTotal   = {n_total_frames};

  // ── State ─────────────────────────────────────────────────────────────────
  var curF     = 0;
  var interval = 120;
  var playing  = false;
  var timer    = null;
  var curModel = null;

  // ── Viewer ────────────────────────────────────────────────────────────────
  var container = document.getElementById('vdiv_{uid}');
  var viewer = $3Dmol.createViewer(container, {{backgroundColor: '#0d1117'}});

  // Draw unit-cell box once (shapes persist across removeAllModels)
  {lattice_js}

  // ── Atom helpers ──────────────────────────────────────────────────────────
  function makeAtoms(fi) {{
    var atoms = [];
    var pos = frames[fi];
    for (var i = 0; i < pos.length; i++) {{
      atoms.push({{
        elem: spList[i],
        x: pos[i][0], y: pos[i][1], z: pos[i][2],
        serial: i, bonds: [], bondOrder: []
      }});
    }}
    return atoms;
  }}

  function applyStyles(model) {{
    model.setStyle({{elem:'Sb'}}, {{sphere:{{radius:0.38, color:'#8B5CF6'}}}});
    model.setStyle({{elem:'Te'}}, {{sphere:{{radius:0.34, color:'#06B6D4'}}}});
    model.setStyle({{elem:'Cr'}}, {{sphere:{{radius:0.30, color:'#EF4444'}}}});
    model.setStyle({{elem:'S'}},  {{sphere:{{radius:0.25, color:'#FBBF24'}}}});
    model.setStyle({{elem:'O'}},  {{sphere:{{radius:0.22, color:'#F87171'}}}});
  }}

  // ── Initial frame ─────────────────────────────────────────────────────────
  curModel = viewer.addModel();
  curModel.addAtoms(makeAtoms(0));
  applyStyles(curModel);
  viewer.zoomTo();
  viewer.render();

  // ── Frame seek (core function) ─────────────────────────────────────────────
  function seekFrame(i) {{
    i = Math.max(0, Math.min(nFrames - 1, i | 0));
    curF = i;

    // Preserve camera before swapping model
    var view = viewer.getView();

    // Replace model (removeAllModels only clears molecular models, NOT shapes)
    viewer.removeAllModels();
    curModel = viewer.addModel();
    curModel.addAtoms(makeAtoms(i));
    applyStyles(curModel);

    // Restore camera so view doesn't jump
    viewer.setView(view);
    viewer.render();

    // Update controls
    document.getElementById('fslider_{uid}').value = i;
    document.getElementById('flbl_{uid}').textContent =
      'step ' + origIdx[i] + ' / ' + nTotal;
  }}

  // ── Slider ────────────────────────────────────────────────────────────────
  document.getElementById('fslider_{uid}').addEventListener('input', function() {{
    seekFrame(parseInt(this.value));
  }});

  // ── Play / Pause ──────────────────────────────────────────────────────────
  document.getElementById('pbtn_{uid}').addEventListener('click', function() {{
    if (playing) {{
      clearInterval(timer);
      playing = false;
      this.textContent = '▶ Play';
    }} else {{
      playing = true;
      this.textContent = '⏸ Pause';
      timer = setInterval(function() {{
        seekFrame((curF + 1) % nFrames);
      }}, interval);
    }}
  }});

  // ── Speed selector ────────────────────────────────────────────────────────
  document.getElementById('fspd_{uid}').addEventListener('change', function() {{
    interval = parseInt(this.value);
    if (playing) {{
      clearInterval(timer);
      timer = setInterval(function() {{
        seekFrame((curF + 1) % nFrames);
      }}, interval);
    }}
  }});

  // ── Reset view ────────────────────────────────────────────────────────────
  document.getElementById('rstbtn_{uid}').addEventListener('click', function() {{
    viewer.zoomTo();
    viewer.render();
  }});

}})();
</script>"""
    components.html(html, height=height, scrolling=False)


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
    species  = traj.get('species', [])
    energies = traj.get('energies')
    meta_raw = traj.get('metadata', {})

    # Parse lattice from first comment line for 3D viewer
    lattice_viewer = None
    try:
        with open(tmp_path, 'r') as _f:
            int(_f.readline().strip())
            lattice_viewer = _parse_lattice_from_comment(_f.readline())
    except Exception:
        pass

    X_upl, windows = extractor.extract_all_windows(coords, energies)
    X_upl = _impute(X_upl)

    framework = D['framework']
    results   = framework.predict(X_upl)

    # Sample coords for 3D viewer (75 frames max, store orig indices for step labels)
    n_total_fr = int(coords.shape[0])
    n_sample   = min(75, n_total_fr)
    idx_s      = np.linspace(0, n_total_fr - 1, n_sample, dtype=int)
    coords_s   = coords[idx_s]

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
        # 3D viewer data
        'coords_viewer':    coords_s,        # (n_sample, n_atoms, 3) float32
        'species_viewer':   species,
        'lattice_viewer':   lattice_viewer,
        'orig_idx_viewer':  idx_s,           # original step indices
        'n_total_frames':   n_total_fr,
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

_nav_pages = [
    "📊 Data Overview",
    "🔍 Feature Analysis",
    "⚠️ Anomaly Detection",
    "⚖️ AIMD vs Upload",
]
if OLLAMA_AVAILABLE:
    _nav_pages.append("🤖 AI Analysis")

page = st.sidebar.radio(
    "Navigate",
    _nav_pages,
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

    # ── 3D Trajectory Viewers ─────────────────────────────────────────────────
    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)
    st.subheader("3D Molecular Trajectories")
    st.caption(
        "WebGL interactive viewer · atoms rendered as spheres · "
        "Sb = purple, Te = teal, Cr = red · lattice box shown in gray"
    )

    v_col_a, v_col_b = st.columns(2)

    with v_col_a:
        # ── AIMD baseline file selector ───────────────────────────────────────
        raw_dir = ROOT / 'data' / 'raw'
        aimd_xyz_files = sorted(raw_dir.rglob('*.xyz'))
        aimd_labels    = [str(p.relative_to(raw_dir)) for p in aimd_xyz_files]

        if not aimd_xyz_files:
            st.info("No AIMD .xyz files found in data/raw/.")
        else:
            sel_label = st.selectbox(
                "AIMD file", aimd_labels, key="viewer_aimd_sel",
                label_visibility="collapsed",
                help="Select an AIMD trajectory file to visualize",
            )
            sel_path = str(raw_dir / sel_label)
            with st.spinner("Loading AIMD trajectory…"):
                try:
                    c_aimd, sp_aimd, lat_aimd, oidx_aimd, ntot_aimd = \
                        _sample_traj_for_viewer(sel_path, n_sample=75)
                    inject_3d_viewer(c_aimd, sp_aimd, lat_aimd, oidx_aimd, ntot_aimd,
                                     title=f"AIMD · {sel_label}",
                                     height=520)
                except Exception as _e:
                    st.error(f"Could not load viewer: {_e}")

    with v_col_b:
        # ── Active upload viewer ──────────────────────────────────────────────
        c_upl   = active_data.get('coords_viewer')
        sp_upl  = active_data.get('species_viewer')
        lat_upl = active_data.get('lattice_viewer')
        oidx_upl= active_data.get('orig_idx_viewer')
        ntot_upl= active_data.get('n_total_frames', 0)

        if c_upl is None or len(sp_upl or []) == 0:
            st.info(
                "Upload a trajectory via the sidebar to see the 3D viewer here.\n\n"
                "The default MLFF baseline does not carry raw coordinate data."
            )
        else:
            inject_3d_viewer(c_upl, sp_upl, lat_upl, oidx_upl, ntot_upl,
                             title=f"Upload · {upload_name}",
                             height=520)


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
        # ── Feature glossary / info button ───────────────────────────────────
        _FEAT_GLOSSARY = {
            "Displacement statistics": {
                "features": ["disp_mean","disp_std","disp_skew","disp_kurtosis","disp_max","disp_median","disp_p95"],
                "description": (
                    "Per-frame atomic displacement from the reference position, aggregated over all atoms in the window. "
                    "**mean/median/max/p95** measure how far atoms move on average, at typical values, at extremes. "
                    "**std** captures spread — high std means some atoms are stationary while others move a lot. "
                    "**skew/kurtosis** reveal distributional asymmetry (tail behavior): large positive kurtosis = rare extreme events."
                ),
            },
            "Dynamics": {
                "features": ["rms_velocity","crest_factor","impulse_factor","frame_variance","anisotropy"],
                "description": (
                    "**rms_velocity** = root-mean-square of per-atom velocity (energy proxy). "
                    "**crest_factor** = peak / RMS — detects abrupt velocity spikes (shocks). "
                    "**impulse_factor** = peak / mean — another spike metric, common in vibration analysis. "
                    "**frame_variance** = variance of mean displacement across consecutive frames (trajectory smoothness). "
                    "**anisotropy** = ratio of max-axis to min-axis displacement variance — detects directional bias."
                ),
            },
            "Frequency domain": {
                "features": ["dominant_freq","spectral_entropy","spectral_peak_ratio"],
                "description": (
                    "FFT applied to the displacement time-series within each window. "
                    "**dominant_freq** = frequency with most energy — captures characteristic oscillation rate. "
                    "**spectral_entropy** = Shannon entropy of the power spectrum — high entropy = broadband noise, "
                    "low entropy = clean oscillations. "
                    "**spectral_peak_ratio** = power at dominant_freq / total power — how 'peaked' the spectrum is."
                ),
            },
            "Mean Squared Displacement": {
                "features": ["msd_mean","msd_std","msd_final","msd_slope"],
                "description": (
                    "MSD(τ) = ⟨|r(t+τ) − r(t)|²⟩ — the standard metric for atomic diffusion. "
                    "**msd_final** = MSD at the end of the window (total diffusion). "
                    "**msd_slope** = linear fit slope ≈ 6D (diffusion coefficient × 6). "
                    "**msd_mean/std** characterize the MSD curve shape. "
                    "For normal DFT dynamics, MSD grows slowly; catastrophic MLFF drift shows explosive MSD growth."
                ),
            },
            "Energy": {
                "features": ["energy_mean","energy_std","energy_trend"],
                "description": (
                    "Total energy per frame from the .xyz file comment line. "
                    "**energy_mean** = average energy over the window (should be stable in equilibrium MD). "
                    "**energy_std** = energy fluctuations (thermodynamic: proportional to heat capacity). "
                    "**energy_trend** = linear slope of energy vs time — non-zero trend indicates drift / "
                    "non-equilibrium behavior or force-field instability."
                ),
            },
        }

        with st.expander("ℹ️ Feature Glossary — what do these metrics measure?", expanded=False):
            for cat, info in _FEAT_GLOSSARY.items():
                st.markdown(f"**{cat}**")
                st.markdown(info["description"])
                st.caption("Features: " + " · ".join(f"`{f}`" for f in info["features"]))
                st.markdown("---")

        st.markdown(
            "<div style='font-family:DM Mono,monospace;font-size:10px;letter-spacing:.12em;"
            f"color:{MUTED};text-transform:uppercase;margin-bottom:8px'>"
            f"Live comparison · AIMD baseline ({len(X_aimd)} windows) vs "
            f"{upload_name} ({len(X_mlff)} windows) · sorted by |z-score|</div>",
            unsafe_allow_html=True,
        )

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 3])
        with col_ctrl1:
            n_show = st.slider("Features to show", 5, 22, 12, key="dist_n_show")
        with col_ctrl2:
            chart_mode = st.radio("Chart type", ["📊 Mean ± std bars", "🎻 Violin distributions"],
                                  horizontal=False, key="dist_mode")
        with col_ctrl3:
            use_log = st.toggle("Log scale (x-axis)", value=False, key="dist_log")
            normalize = st.toggle("Z-score normalize (compare shapes)", value=False, key="dist_norm")

        # Select top-N features by |z-score|
        top_feats = feat_cmp.head(n_show)['feature'].tolist()
        top_feat_idx = [feature_names.index(fn) for fn in top_feats if fn in feature_names]

        if not top_feat_idx or X_aimd.size == 0 or X_mlff.size == 0:
            st.info("Not enough feature data to plot the live comparison yet.")
        else:
            # Per-feature category lookup for color accent
            _CAT_COLOR = {
                **{f: SAGE for f in ["disp_mean","disp_std","disp_skew","disp_kurtosis",
                                      "disp_max","disp_median","disp_p95"]},
                **{f: CORAL for f in ["rms_velocity","crest_factor","impulse_factor",
                                       "frame_variance","anisotropy"]},
                **{f: GOLD for f in ["dominant_freq","spectral_entropy","spectral_peak_ratio"]},
                **{f: '#7C5CBF' for f in ["msd_mean","msd_std","msd_final","msd_slope"]},
                **{f: '#C07050' for f in ["energy_mean","energy_std","energy_trend"]},
            }

            # Build per-feature description for hover
            _FEAT_DESC = {}
            for cat, info in _FEAT_GLOSSARY.items():
                for f in info["features"]:
                    _FEAT_DESC[f] = info["description"][:120] + "…"

            feat_labels = [feature_names[i] for i in top_feat_idx]
            aimd_mat = X_aimd[:, top_feat_idx].copy().astype(float)
            mlff_mat = X_mlff[:, top_feat_idx].copy().astype(float)

            if normalize:
                for j in range(len(top_feat_idx)):
                    mu = np.nanmean(aimd_mat[:, j])
                    sg = np.nanstd(aimd_mat[:, j]) + 1e-10
                    aimd_mat[:, j] = (aimd_mat[:, j] - mu) / sg
                    mlff_mat[:, j] = (mlff_mat[:, j] - mu) / sg

            if chart_mode == "📊 Mean ± std bars":
                aimd_m = np.nanmean(aimd_mat, axis=0)
                mlff_m = np.nanmean(mlff_mat, axis=0)
                aimd_s = np.nanstd(aimd_mat, axis=0)
                mlff_s = np.nanstd(mlff_mat, axis=0)

                # Get z-scores for each feature
                z_vals = []
                for fn in feat_labels:
                    row = feat_cmp[feat_cmp['feature'] == fn]
                    z_vals.append(float(row['z_score'].iloc[0]) if len(row) else 0.0)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='AIMD (baseline)',
                    y=feat_labels,
                    x=aimd_m,
                    error_x=dict(type='data', array=aimd_s, visible=True,
                                 color='rgba(156,141,127,0.55)', thickness=1.5, width=4),
                    orientation='h',
                    marker=dict(color='rgba(90,122,92,0.82)',
                                line=dict(color='rgba(90,122,92,0.9)', width=0)),
                    customdata=np.column_stack([aimd_m, aimd_s,
                                                [len(X_aimd)]*len(feat_labels),
                                                [_FEAT_DESC.get(f, '') for f in feat_labels],
                                                z_vals]),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "AIMD mean: <b>%{customdata[0]:.5g}</b><br>"
                        "AIMD std:  %{customdata[1]:.4g}<br>"
                        "Windows:   %{customdata[2]}<br>"
                        "Z-score:   %{customdata[4]:+.2f}<br>"
                        "<i style='color:#9C8D7F'>%{customdata[3]}</i>"
                        "<extra>AIMD baseline</extra>"
                    ),
                ))
                fig.add_trace(go.Bar(
                    name=upload_name[:28],
                    y=feat_labels,
                    x=mlff_m,
                    error_x=dict(type='data', array=mlff_s, visible=True,
                                 color='rgba(156,141,127,0.55)', thickness=1.5, width=4),
                    orientation='h',
                    marker=dict(color='rgba(224,92,58,0.82)',
                                line=dict(color='rgba(224,92,58,0.9)', width=0)),
                    customdata=np.column_stack([mlff_m, mlff_s,
                                                [len(X_mlff)]*len(feat_labels),
                                                [_FEAT_DESC.get(f, '') for f in feat_labels],
                                                z_vals]),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        f"{upload_name[:24]} mean: <b>%{{customdata[0]:.5g}}</b><br>"
                        "std:     %{customdata[1]:.4g}<br>"
                        "Windows: %{customdata[2]}<br>"
                        "Z-score vs AIMD: <b>%{customdata[4]:+.2f}</b><br>"
                        "<i style='color:#9C8D7F'>%{customdata[3]}</i>"
                        "<extra>Upload</extra>"
                    ),
                ))
                fig.update_layout(
                    barmode='group',
                    title=dict(
                        text=f"Feature means ± std — AIMD vs {upload_name[:30]}",
                        font=dict(family="DM Serif Display, serif", size=16, color=INK),
                        x=0.02,
                    ),
                    xaxis=dict(
                        title=dict(
                            text="Feature value" + (" (log)" if use_log else "") + (" [z-normalized]" if normalize else ""),
                            font=dict(family="DM Sans, sans-serif", size=11, color=SUB),
                        ),
                        type='log' if use_log else 'linear',
                        gridcolor=BORDER_C, gridwidth=0.8,
                        tickfont=dict(family="DM Mono, monospace", size=10, color=SUB),
                    ),
                    yaxis=dict(
                        tickfont=dict(family="DM Mono, monospace", size=10, color=INK),
                        gridcolor=BORDER_C, gridwidth=0.5,
                    ),
                    legend=dict(
                        font=dict(family="DM Sans, sans-serif", size=11, color=TEXT),
                        bgcolor=SURFACE, bordercolor=BORDER_C, borderwidth=1,
                        x=0.78, y=0.02,
                    ),
                    paper_bgcolor=BG,
                    plot_bgcolor=SURFACE2,
                    height=max(340, n_show * 42),
                    margin=dict(l=10, r=20, t=50, b=40),
                    hoverlabel=dict(
                        bgcolor=SURFACE,
                        bordercolor=BORDER_C,
                        font=dict(family="DM Sans, sans-serif", size=12, color=TEXT),
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

            else:  # Violin
                fig = go.Figure()
                for j, (fi, fn) in enumerate(zip(top_feat_idx, feat_labels)):
                    av = aimd_mat[:, j]; av = av[~np.isnan(av)]
                    mv = mlff_mat[:, j]; mv = mv[~np.isnan(mv)]
                    row = feat_cmp[feat_cmp['feature'] == fn]
                    z = float(row['z_score'].iloc[0]) if len(row) else 0.0
                    desc = _FEAT_DESC.get(fn, '')

                    fig.add_trace(go.Violin(
                        y=av, x=[fn]*len(av),
                        name='AIMD', legendgroup='AIMD', showlegend=(j == 0),
                        side='negative', line_color=SAGE,
                        fillcolor='rgba(90,122,92,0.27)', opacity=0.85,
                        box_visible=True, meanline_visible=True,
                        hovertemplate=(
                            f"<b>{fn}</b> — AIMD<br>"
                            f"n={len(av)} windows<br>"
                            f"mean={np.mean(av):.4g}  std={np.std(av):.4g}<br>"
                            f"Z-score vs upload: {z:+.2f}<br>"
                            f"<i style='color:#9C8D7F'>{desc[:100]}…</i>"
                            "<extra>AIMD</extra>"
                        ),
                    ))
                    fig.add_trace(go.Violin(
                        y=mv, x=[fn]*len(mv),
                        name=upload_name[:20], legendgroup='upload', showlegend=(j == 0),
                        side='positive', line_color=CORAL,
                        fillcolor='rgba(224,92,58,0.27)', opacity=0.85,
                        box_visible=True, meanline_visible=True,
                        hovertemplate=(
                            f"<b>{fn}</b> — {upload_name[:20]}<br>"
                            f"n={len(mv)} windows<br>"
                            f"mean={np.mean(mv):.4g}  std={np.std(mv):.4g}<br>"
                            f"Z-score vs AIMD: <b>{z:+.2f}</b><br>"
                            f"<i style='color:#9C8D7F'>{desc[:100]}…</i>"
                            "<extra>Upload</extra>"
                        ),
                    ))
                fig.update_layout(
                    violinmode='overlay',
                    title=dict(
                        text=f"Feature distributions (violin) — AIMD vs {upload_name[:30]}",
                        font=dict(family="DM Serif Display, serif", size=16, color=INK),
                        x=0.02,
                    ),
                    xaxis=dict(tickfont=dict(family="DM Mono, monospace", size=9, color=INK),
                               gridcolor=BORDER_C),
                    yaxis=dict(
                        title="Feature value" + (" [z-normalized]" if normalize else ""),
                        type='log' if use_log else 'linear',
                        gridcolor=BORDER_C,
                        tickfont=dict(family="DM Mono, monospace", size=10, color=SUB),
                    ),
                    legend=dict(font=dict(family="DM Sans, sans-serif", size=11, color=TEXT),
                                bgcolor=SURFACE, bordercolor=BORDER_C, borderwidth=1),
                    paper_bgcolor=BG, plot_bgcolor=SURFACE2,
                    height=max(420, n_show * 38),
                    margin=dict(l=10, r=20, t=50, b=60),
                    hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER_C,
                                    font=dict(family="DM Sans, sans-serif", size=12, color=TEXT)),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "💡 Hover over bars/violins for exact values, std, z-score, and a feature description. "
                "Toggle **Log scale** when MLFF values are orders-of-magnitude larger than AIMD. "
                "Toggle **Z-score normalize** to compare distribution shapes on the same scale."
            )

            st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

            # ── LLM Q&A on Feature Analysis ──────────────────────────────────
            with st.expander("🤖 Ask AI about these features", expanded=False):
                st.caption(
                    "Uses your local `glm-5:cloud` model via Ollama to answer questions about "
                    "the physics, statistics, or anomaly patterns you see above."
                )
                feat_q = st.text_area(
                    "Your question",
                    placeholder=(
                        "e.g. 'Why is rms_velocity so much higher in the MLFF trajectory?' "
                        "or 'What does a high spectral_entropy mean physically?'"
                    ),
                    height=80,
                    key="feat_llm_q",
                )
                if st.button("Ask AI", key="feat_llm_btn"):
                    if not feat_q.strip():
                        st.warning("Please enter a question first.")
                    else:
                        try:
                            analyst = OllamaAnalyst()
                            data_ctx = build_data_context(
                                X_aimd, X_mlff, feature_names, ra, rm, meta_aimd, feat_cmp
                            )
                            with st.spinner("Thinking…"):
                                answer = analyst.mechanism_analysis(feat_q, data_ctx)
                            lines = answer.split('\n')
                            out_parts = []
                            for ln in lines:
                                if ln.startswith('## '):
                                    out_parts.append(
                                        f"<div style='font-family:DM Serif Display,serif;"
                                        f"font-size:1rem;color:{INK};margin:12px 0 4px;"
                                        f"font-weight:600'>{ln[3:]}</div>"
                                    )
                                elif '► Claim:' in ln or '► Uncertain:' in ln:
                                    out_parts.append(
                                        f'<div class="claim-line">🔸 {ln.strip()}</div>'
                                    )
                                else:
                                    out_parts.append(
                                        f"<p style='color:{SUB};margin:3px 0'>{ln}</p>"
                                    )
                            st.markdown(
                                '<div class="evidence-block">'
                                + '\n'.join(out_parts)
                                + '</div>',
                                unsafe_allow_html=True,
                            )
                        except Exception as exc:
                            st.error(f"Ollama error: {exc}. Make sure `ollama serve` is running with `glm-5:cloud`.")

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
        inject_feature_table(feat_cmp, height=540)


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
    anom = res_sel['anomaly_label']
    
    # Safety check: ensure conf and anom match X length
    if len(conf) != n_win or len(anom) != n_win:
        st.warning(
            f"⚠️ Data shape mismatch: X has {n_win} windows, but results has "
            f"{len(conf)} confidence and {len(anom)} labels. Using first {n_win} elements."
        )
        conf = conf[:n_win]
        anom = anom[:n_win]
    
    ax_tl.fill_between(range(n_win), conf, alpha=0.25, color=color_sel)
    ax_tl.plot(conf, lw=0.8, color=color_sel)
    ax_tl.axhline(2, color=RED, ls='--', lw=1.5, label='Anomaly threshold (≥2/3 detectors)')
    ax_tl.set_ylim(-0.15, 3.5)
    ax_tl.legend(facecolor=SURFACE2, labelcolor=TEXT, fontsize=9)
    rate_pct = np.mean(anom)
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
        anom_in_range = float(np.mean(anom[w_start:w_end]))
        st.metric("Windows", w_end - w_start)
        st.metric("Rate in range", f"{anom_in_range:.1%}")

    # Zoomed timeline + detector heatmap
    fig_z, axes_z = plt.subplots(2, 1, figsize=(14, 6),
                                  sharex=True, facecolor=SURFACE,
                                  constrained_layout=True)
    idx    = np.arange(w_start, w_end)
    conf_z = conf[w_start:w_end]
    anom_z = anom[w_start:w_end]

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

    l1_flag = res_sel['l1_flag'][:n_win]
    l2_if_flag = res_sel['l2_if_flag'][:n_win]
    l2_svm_flag = res_sel['l2_svm_flag'][:n_win]
    flags = np.vstack([
        l2_svm_flag[w_start:w_end],
        l2_if_flag[w_start:w_end],
        l1_flag[w_start:w_end],
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
    st.caption(f"Live comparison: AIMD baseline vs {upload_name} · sorted by |z-score|, click headers to re-sort")
    inject_feature_table(feat_cmp, height=580)

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
        # Build per-window results DataFrame — use result length as reference
        _n = len(rm['anomaly_label'])
        win_df = pd.DataFrame({
            'window':        range(_n),
            'anomaly_label': np.asarray(rm['anomaly_label'])[:_n],
            'confidence':    np.asarray(rm['confidence'])[:_n],
            'l1_flag':       np.asarray(rm['l1_flag'])[:_n],
            'l2_if_flag':    np.asarray(rm['l2_if_flag'])[:_n],
            'l2_svm_flag':   np.asarray(rm['l2_svm_flag'])[:_n],
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
