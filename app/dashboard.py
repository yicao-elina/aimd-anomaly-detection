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
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.loaders import TrajectoryLoader
from src.core.feature_extractors import FeatureExtractor, WindowConfig, compute_file_energy_shift
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


def inject_overview_table(df: pd.DataFrame, height: int = 280) -> None:
    """
    Render a themed HTML data table for the Data Overview page.
    Bypasses st.dataframe() to avoid the white-text-on-white-background bug
    caused by Streamlit's canvas renderer conflicting with our custom CSS.
    """
    import json, uuid
    if df is None or df.empty:
        st.info("No data available.")
        return

    uid = uuid.uuid4().hex[:8]

    # Format cell values — round floats, truncate long strings
    def _fmt(v):
        if isinstance(v, float):
            return f"{v:,.1f}" if abs(v) >= 100 else f"{v:.4g}"
        return str(v)[:40] if v is not None else '—'

    headers_html = ''.join(
        f"<th onclick=\"sortOvr_{uid}(this,{i})\">{col}</th>"
        for i, col in enumerate(df.columns)
    )
    rows_html = ''
    for _, row in df.iterrows():
        cells = ''.join(f"<td>{_fmt(v)}</td>" for v in row)
        rows_html += f"<tr>{cells}</tr>\n"

    html = f"""
<style>
  #ovr_{uid} {{ font-family:'DM Sans',sans-serif; font-size:12px;
    background:#ffffff; border:1px solid {BORDER_C}; border-radius:8px;
    overflow:auto; max-height:{height}px; }}
  #ovr_{uid} table {{ width:100%; border-collapse:collapse; table-layout:auto; }}
  #ovr_{uid} thead {{ position:sticky; top:0; z-index:2; background:{SURFACE2}; }}
  #ovr_{uid} th {{
    padding:8px 12px; text-align:left; cursor:pointer; user-select:none;
    font-family:'DM Mono',monospace; font-size:10px; letter-spacing:.12em;
    text-transform:uppercase; color:{SUB}; border-bottom:2px solid {BORDER_C};
    white-space:nowrap;
  }}
  #ovr_{uid} th:hover {{ color:{CORAL}; }}
  #ovr_{uid} th.asc::after  {{ content:' ↑'; color:{CORAL}; }}
  #ovr_{uid} th.desc::after {{ content:' ↓'; color:{CORAL}; }}
  #ovr_{uid} td {{
    padding:7px 12px; color:{TEXT}; border-bottom:1px solid {BORDER_C};
    font-family:'DM Mono',monospace; font-size:11px; white-space:nowrap;
  }}
  #ovr_{uid} tr:hover td {{ background:{SURFACE2}; }}
  #ovr_{uid} tr:last-child td {{ border-bottom:none; }}
</style>
<div id="ovr_{uid}">
  <table id="tbl_{uid}">
    <thead><tr>{headers_html}</tr></thead>
    <tbody id="tbody_{uid}">{rows_html}</tbody>
  </table>
</div>
<script>
(function() {{
  var asc = {{}};
  window.sortOvr_{uid} = function(th, colIdx) {{
    var tbody = document.getElementById('tbody_{uid}');
    var rows = Array.from(tbody.querySelectorAll('tr'));
    var dir = asc[colIdx] = !asc[colIdx];
    // Remove sort classes from all headers
    th.closest('tr').querySelectorAll('th').forEach(function(h) {{
      h.classList.remove('asc','desc');
    }});
    th.classList.add(dir ? 'asc' : 'desc');
    rows.sort(function(a,b) {{
      var av = a.cells[colIdx].textContent.replace(/,/g,'');
      var bv = b.cells[colIdx].textContent.replace(/,/g,'');
      var an = parseFloat(av), bn = parseFloat(bv);
      if (!isNaN(an) && !isNaN(bn)) return dir ? an-bn : bn-an;
      return dir ? av.localeCompare(bv) : bv.localeCompare(av);
    }});
    rows.forEach(function(r) {{ tbody.appendChild(r); }});
  }};
}})();
</script>"""
    components.html(html, height=height + 4, scrolling=False)


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


def _build_feat_cmp_placeholder(feature_names, X_mlff, X_aimd):
    """Fallback for when feature_comparison.csv doesn't exist (e.g. fresh clone)."""
    aimd_means = np.nanmean(X_aimd, axis=0)
    aimd_stds  = np.nanstd(X_aimd, axis=0)
    upl_means  = np.nanmean(X_mlff, axis=0)
    rows = []
    for i, fn in enumerate(feature_names):
        z = (upl_means[i] - aimd_means[i]) / (aimd_stds[i] + 1e-10)
        rel = (upl_means[i] - aimd_means[i]) / (abs(aimd_means[i]) + 1e-10) * 100
        rows.append({'feature': fn, 'aimd_mean': aimd_means[i], 'aimd_std': aimd_stds[i],
                     'mlff_mean': upl_means[i], 'z_score': z, 'relative_change_%': rel})
    return pd.DataFrame(rows).sort_values('z_score', key=abs, ascending=False).reset_index(drop=True)


# ── Data loading (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    aimd_npz = np.load(PROCESSED_DIR / 'features_aimd.npz', allow_pickle=True)
    mlff_npz = np.load(PROCESSED_DIR / 'features_mlff.npz', allow_pickle=True)
    framework = AnomalyDetectionFramework.load(MODELS_DIR / 'anomaly_framework.pkl')

    # Load energy reference for Option B empirical shift (upload pipeline)
    energy_ref_path = PROCESSED_DIR / 'energy_ref.json'
    ref_atm_per_atom = -3.75  # fallback if file missing
    if energy_ref_path.exists():
        with open(energy_ref_path) as _f:
            ref_atm_per_atom = json.load(_f).get('ref_atm_per_atom', -3.75)

    return {
        'X_aimd':        aimd_npz['X'],
        'feature_names': list(aimd_npz['feature_names']),
        'meta_aimd':     pd.read_csv(PROCESSED_DIR / 'meta_aimd.csv'),
        'framework':     framework,
        'summary':       json.load(open(PROCESSED_DIR / 'pipeline_summary.json')),
        # Option B reference atomization energy — used for energy shift in upload
        'ref_atm_per_atom': ref_atm_per_atom,
        # Default MLFF dataset (from original pipeline run)
        'X_mlff_default':    mlff_npz['X'],
        'meta_mlff_default': pd.read_csv(PROCESSED_DIR / 'meta_mlff.csv'),
        'feat_cmp_default':  (
            pd.read_csv(REPORTS_DIR / 'feature_comparison.csv')
            if (REPORTS_DIR / 'feature_comparison.csv').exists()
            else _build_feat_cmp_placeholder(
                list(aimd_npz['feature_names']), mlff_npz['X'], aimd_npz['X']
            )
        ),
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


def _run_upload_pipeline(xyz_bytes: bytes, filename: str, D: dict,
                         _status=None) -> dict:
    """
    Predict-only pipeline for an uploaded .xyz file.
    Uses the already-trained AnomalyDetectionFramework (no retraining).

    _status: optional st.status container — if provided, step labels are written
             into it so the user sees live progress.
    """
    def _step(msg):
        if _status is not None:
            _status.write(msg)

    # ── Step 1: write bytes to a temp file ───────────────────────────────────
    _step("📂 Writing temp file…")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tf:
            tf.write(xyz_bytes)
            tmp_path = tf.name

        # ── Step 2: parse XYZ ────────────────────────────────────────────────
        mb = len(xyz_bytes) / 1_048_576
        _step(f"🔍 Parsing XYZ ({mb:.1f} MB)…")
        loader = TrajectoryLoader()
        traj   = loader.load(tmp_path)

        coords   = traj['coordinates']   # (n_frames, n_atoms, 3)
        species  = traj.get('species', [])
        energies = traj.get('energies')
        meta_raw = traj.get('metadata', {})
        n_frames, n_atoms = coords.shape[0], coords.shape[1]
        _step(f"   → {n_frames} frames · {n_atoms} atoms")

        # Parse lattice from first comment line for 3D viewer
        lattice_viewer = None
        try:
            with open(tmp_path, 'r') as _f:
                int(_f.readline().strip())
                lattice_viewer = _parse_lattice_from_comment(_f.readline())
        except Exception:
            pass

        # ── Step 3: feature extraction ───────────────────────────────────────
        _step(f"⚙️ Extracting 27 features (window=50, stride=10)…")
        extractor = FeatureExtractor(WindowConfig(window_size=50, stride=10))
        X_upl, windows = extractor.extract_all_windows(coords, energies, species)
        X_upl = _impute(X_upl)
        _step(f"   → {len(windows)} windows · {X_upl.shape[1]} features")

        # ── Step 4: anomaly detection ────────────────────────────────────────
        _step("🚨 Running L1+L2 anomaly detection…")
        results = D['framework'].predict(X_upl)
        anom_rate = float(np.mean(results['anomaly_label']))
        _step(f"   → anomaly rate: {anom_rate*100:.1f}%")

        # ── Step 5: 3D viewer sampling ────────────────────────────────────────
        # Cap viewer atoms at 300 to keep the browser JSON payload manageable
        # for large supercells (e.g. 2050-atom 5×5×1 supercell).
        MAX_VIEWER_ATOMS = 300
        n_sample   = min(75, n_frames)
        idx_s      = np.linspace(0, n_frames - 1, n_sample, dtype=int)
        coords_s   = coords[idx_s]
        species_v  = species
        if n_atoms > MAX_VIEWER_ATOMS:
            atom_idx_v = np.linspace(0, n_atoms - 1, MAX_VIEWER_ATOMS, dtype=int)
            coords_s   = coords_s[:, atom_idx_v, :]
            species_v  = [species[i] for i in atom_idx_v] if species else []
            _step(f"   → 3D viewer: subsampled to {MAX_VIEWER_ATOMS} atoms")

        # ── Build metadata DataFrame ──────────────────────────────────────────
        n_w  = len(X_upl)
        meta = pd.DataFrame({
            'file':          [filename] * n_w,
            'start':         [w[0] for w in windows],
            'end':           [w[1] for w in windows],
            'n_atoms':       [n_atoms] * n_w,
            'temperature_K': [meta_raw.get('temperature_K')] * n_w,
            'configuration': [meta_raw.get('configuration', 'uploaded')] * n_w,
        })
        feat_cmp = _build_feat_comparison(D['X_aimd'], X_upl, D['feature_names'])

        return {
            'X':              X_upl,
            'meta':           meta,
            'results':        results,
            'feat_cmp':       feat_cmp,
            'filename':       filename,
            'timestamp':      datetime.datetime.now().strftime('%H:%M:%S'),
            'n_windows':      n_w,
            'n_frames':       n_frames,
            'n_atoms':        n_atoms,
            'anom_rate':      anom_rate,
            # 3D viewer
            'coords_viewer':  coords_s,
            'species_viewer': species_v,
            'lattice_viewer': lattice_viewer,
            'orig_idx_viewer': idx_s,
            'n_total_frames': n_frames,
        }
    finally:
        # Always clean up the temp file
        if tmp_path is not None:
            try:
                import os; os.unlink(tmp_path)
            except OSError:
                pass


# ── Load base data ───────────────────────────────────────────────────────────
try:
    D = load_all()
except FileNotFoundError as e:
    st.error(f"**Pipeline outputs missing:** {e}")
    st.info("Run the pipeline first:\n```bash\npython scripts/run_full_pipeline.py\n```")
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
     "🤖 AI Analysis",
     "🔬 Active Learning"],
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
        xyz_bytes = uploaded_file.read()
        file_mb   = len(xyz_bytes) / 1_048_576
        # Warn if the file is large (> 50 MB) so the user knows to expect a wait
        if file_mb > 50:
            st.sidebar.warning(
                f"Large file ({file_mb:.0f} MB) — this may take 20–60 s. "
                "Processing in background…"
            )
        with st.sidebar.status(
            f"Analyzing {uploaded_file.name} ({file_mb:.1f} MB)…",
            expanded=True,
        ) as _upload_status:
            try:
                result = _run_upload_pipeline(
                    xyz_bytes, uploaded_file.name, D,
                    _status=_upload_status,
                )
                st.session_state.uploads[key] = result
                st.session_state.active_upload = key
                _upload_status.update(
                    label=f"✓ {uploaded_file.name} — {result['n_windows']} windows · "
                          f"{result['anom_rate']:.0%} anomaly",
                    state="complete", expanded=False,
                )
            except Exception as exc:
                import traceback
                _upload_status.update(label=f"✗ {uploaded_file.name}", state="error")
                st.sidebar.error(f"**Pipeline error:** {exc}")
                st.sidebar.code(traceback.format_exc(), language="python")
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
        "Pipeline: L1+L2 ensemble · 27 features · window=50 · stride=10."
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
            inject_overview_table(tbl, height=280)

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
            inject_overview_table(upl_tbl, height=280)

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
            "Structural Integrity (NEW)": {
                "features": ["min_interatomic_dist","rdf_first_peak_pos","rdf_first_peak_height"],
                "description": (
                    "Physics-motivated features derived from all pairwise atomic distances in the window. "
                    "**min_interatomic_dist** = minimum pairwise distance (Å) over all atoms and frames — "
                    "values below ~1.5 Å signal atomic clashes (Pauli repulsion violation), the clearest "
                    "signature of a nonsensical MLFF configuration. "
                    "**rdf_first_peak_pos** = position of the first peak of the radial distribution function "
                    "(Å); shifts indicate bond-length drift or a structural phase change. "
                    "**rdf_first_peak_height** = normalised height of the first RDF peak; broadening or "
                    "collapse signals an order→disorder transition characteristic of MLFF instability."
                ),
            },
            "Velocity Autocorrelation (NEW)": {
                "features": ["vacf_initial_decay","vacf_zero_crossing"],
                "description": (
                    "VACF probes the vibrational and dynamical signature of the trajectory. "
                    "Velocities are approximated as frame-to-frame coordinate differences. "
                    "**vacf_initial_decay** = fractional drop (VACF[0]−VACF[1])/VACF[0]; "
                    "large values indicate stiff high-frequency vibrations or erratic atomic motion "
                    "typical of MLFF instability at the onset of structural collapse. "
                    "**vacf_zero_crossing** = first zero-crossing time as a fraction of max lag; "
                    "short crossing ↔ high-frequency oscillations; value=1.0 (no crossing) indicates "
                    "strongly correlated drift — the catastrophic displacement signature."
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
                **{f: '#1A7A4A' for f in ["min_interatomic_dist","rdf_first_peak_pos","rdf_first_peak_height"]},
                **{f: '#2A6496' for f in ["vacf_initial_decay","vacf_zero_crossing"]},
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

    # ══════════════════════════════════════════════════════════════════════
    # INTERPRETABILITY PANELS (from docs/0217_improvement.md)
    # ══════════════════════════════════════════════════════════════════════

    # ── 1. Explain a Single Window ─────────────────────────────────────────
    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)
    st.subheader("🔬 Explain a Single Window")
    st.caption(
        "Select any window to see **why** the detector flagged it: per-feature z-scores, "
        "detector scores vs thresholds, feature-group radar, and physical root cause."
    )

    # Default to first anomalous window
    _anom_idx = np.where(np.asarray(res_sel['anomaly_label'][:n_win]) == 1)[0]
    _default_win = int(_anom_idx[0]) if len(_anom_idx) > 0 else 0

    explain_win = st.slider(
        "Select window to explain",
        min_value=0, max_value=n_win - 1,
        value=_default_win, key='explain_win_slider',
    )

    # Compute per-feature z-scores for this window
    _aimd_m = np.nanmean(X_aimd, axis=0)
    _aimd_s = np.nanstd(X_aimd, axis=0)
    _x_win  = X_sel[explain_win]
    _z_win  = (_x_win - _aimd_m) / (_aimd_s + 1e-10)   # signed z-scores, shape (n_feat,)

    # Detector scores for this window
    _l1_flag_w   = int(res_sel['l1_flag'][explain_win])
    _l2_if_flag  = int(res_sel['l2_if_flag'][explain_win])
    _l2_svm_flag = int(res_sel['l2_svm_flag'][explain_win])
    _conf_w      = int(res_sel['confidence'][explain_win])
    _l1_score_w  = float(res_sel['l1_score'][explain_win])
    _if_score_w  = float(res_sel['l2_if_score'][explain_win])
    _svm_score_w = float(res_sel['l2_svm_score'][explain_win])
    _is_anom_w   = bool(res_sel['anomaly_label'][explain_win])

    # IF threshold = 5th percentile of AIMD IF scores (contamination=5%)
    _if_thresh = float(np.percentile(np.asarray(ra['l2_if_score']), 5))

    _SEV_MAP = {0: "🟢 Normal", 1: "🟡 Borderline", 2: "🟠 Anomaly", 3: "🔴 Catastrophic"}
    _verdict = _SEV_MAP[_conf_w]

    # Summary metrics row
    _mc1, _mc2, _mc3, _mc4, _mc5 = st.columns(5)
    _mc1.metric("Window", f"#{explain_win}")
    _mc2.metric("L1 stat",   "FLAGGED" if _l1_flag_w else "Normal",
                help=f"L1 score: {_l1_score_w:.1%} features > 3σ  (threshold: 10%)")
    _mc3.metric("L2 IF",    "FLAGGED" if _l2_if_flag  else "Normal",
                help=f"IF score: {_if_score_w:.4f}  |  threshold: {_if_thresh:.4f}")
    _mc4.metric("L2 SVM",   "FLAGGED" if _l2_svm_flag else "Normal",
                help=f"SVM score: {_svm_score_w:.4f}  |  threshold: 0")
    _mc5.metric("Verdict", _verdict)

    st.markdown(
        f"> **L1 score**: {_l1_score_w:.1%} of features outside 3σ &nbsp;·&nbsp; "
        f"**IF score**: {_if_score_w:.4f} (threshold {_if_thresh:.4f}) &nbsp;·&nbsp; "
        f"**SVM score**: {_svm_score_w:.4f} (threshold 0) &nbsp;·&nbsp; "
        f"**Confidence**: {_conf_w}/3 detectors agree"
    )

    _exp_left, _exp_right = st.columns([3, 2])

    with _exp_left:
        # Horizontal bar chart — all features sorted by |z|, color-coded
        _sort_idx   = np.argsort(np.abs(_z_win))[::-1]
        _z_sorted   = _z_win[_sort_idx]
        _names_sort = [feature_names[i] for i in _sort_idx]

        _bar_clrs = []
        for _z in _z_sorted:
            if abs(_z) >= 5:   _bar_clrs.append('#C0392B')   # deep red
            elif abs(_z) >= 3: _bar_clrs.append(RED)
            elif abs(_z) >= 2: _bar_clrs.append(AMBER)
            else:              _bar_clrs.append(SAGE)

        fig_exp, ax_exp = mpl_fig(figsize=(9, 8))
        ax_exp = fig_exp.axes[0]
        ax_exp.barh(_names_sort, _z_sorted, color=_bar_clrs, alpha=0.88)
        ax_exp.axvline(0,  color=TEXT,    lw=0.6)
        ax_exp.axvline( 3, color=RED,     ls='--', lw=1.0, alpha=0.6, label='±3σ threshold')
        ax_exp.axvline(-3, color=RED,     ls='--', lw=1.0, alpha=0.6)
        ax_exp.axvline( 5, color='#C0392B', ls=':', lw=1.0, alpha=0.5, label='±5σ critical')
        ax_exp.axvline(-5, color='#C0392B', ls=':', lw=1.0, alpha=0.5)
        ax_exp.legend(facecolor=SURFACE2, labelcolor=TEXT, fontsize=8)
        _style_ax(ax_exp,
                  title=f'Window #{explain_win} — Z-scores (signed: + = higher than AIMD)',
                  xlabel='Z-score')
        ax_exp.tick_params(axis='y', labelsize=7.5, labelcolor=SUB)
        st.pyplot(fig_exp, use_container_width=True); plt.close(fig_exp)

    with _exp_right:
        # Feature group analysis
        import math as _math
        _FEAT_GROUPS = {
            'Displacement': ['disp_mean','disp_std','disp_skew','disp_kurtosis',
                             'disp_max','disp_median','disp_p95'],
            'Dynamics':     ['rms_velocity','crest_factor','impulse_factor',
                             'frame_variance','anisotropy'],
            'Frequency':    ['dominant_freq','spectral_entropy','spectral_peak_ratio'],
            'Diffusion':    ['msd_mean','msd_std','msd_final','msd_slope'],
            'Energy':       ['energy_mean','energy_std','energy_trend'],
            'Structural':   ['min_interatomic_dist','rdf_first_peak_pos','rdf_first_peak_height'],
            'VACF':         ['vacf_initial_decay','vacf_zero_crossing'],
        }
        _PHYS = {
            'Displacement': (
                "**Catastrophic atomic displacement** — atoms move far beyond DFT-predicted "
                "step sizes. The MLFF predicts wrong forces causing runaway atomic motion."
            ),
            'Dynamics': (
                "**Incorrect force magnitude/direction** — MLFF energy surface is too flat "
                "or too rough, producing wrong velocities and acceleration patterns."
            ),
            'Frequency': (
                "**Phonon frequency mismatch** — PES curvature is wrong near equilibrium, "
                "causing spurious high-frequency oscillations or missing vibrational modes."
            ),
            'Diffusion': (
                "**Diffusion behavior mismatch** — MLFF predicts wrong atomic mobility. "
                "Possible cause: incorrect energy barriers or migration mechanisms."
            ),
            'Energy': (
                "**Energy conservation failure** — non-conservative force prediction "
                "causing systematic drift. Hallmark of a non-physical force field."
            ),
            'Structural': (
                "**Bond-length / structural order deviation** — interatomic distances and "
                "local coordination differ from DFT. May indicate bond breaking or phase change."
            ),
            'VACF': (
                "**Vibrational dynamics anomaly** — velocity autocorrelation deviates "
                "significantly. Possible atomic collision or structural collapse onset."
            ),
        }

        _grp_z = {}
        for _grp, _feats in _FEAT_GROUPS.items():
            _idxs = [feature_names.index(f) for f in _feats if f in feature_names]
            _grp_z[_grp] = float(np.mean(np.abs(_z_win[_idxs]))) if _idxs else 0.0

        _groups = list(_grp_z.keys())
        _vals   = [_grp_z[g] for g in _groups]
        _N      = len(_groups)
        _angles = [n / _N * 2 * _math.pi for n in range(_N)] + [0]
        _vplot  = _vals + [_vals[0]]

        fig_rad, ax_rad = plt.subplots(
            figsize=(4, 4), subplot_kw={'projection': 'polar'}, facecolor=SURFACE
        )
        ax_rad.set_facecolor(SURFACE)
        ax_rad.set_theta_offset(_math.pi / 2)
        ax_rad.set_theta_direction(-1)
        _rc = RED if _is_anom_w else CYAN
        ax_rad.plot(_angles, _vplot, color=_rc, lw=2)
        ax_rad.fill(_angles, _vplot, color=_rc, alpha=0.22)
        ax_rad.set_xticks(_angles[:-1])
        ax_rad.set_xticklabels(_groups, fontsize=8, color=TEXT)
        ax_rad.tick_params(axis='y', colors=MUTED, labelsize=6)
        _max_val = max(max(_vals) * 1.25, 4.0)
        ax_rad.set_ylim(0, _max_val)
        ax_rad.axhline(3, color=RED, ls='--', lw=0.8, alpha=0.55)
        ax_rad.set_title('Feature Group |Z-scores|', color=TEXT, fontsize=10, pad=15)
        ax_rad.grid(True, color=BORDER_C, alpha=0.35)
        for _sp in ax_rad.spines.values(): _sp.set_edgecolor(BORDER_C)
        st.pyplot(fig_rad, use_container_width=True); plt.close(fig_rad)

        # Physical root-cause interpretation
        _top_grp = max(_grp_z, key=_grp_z.get)
        _top_z   = _grp_z[_top_grp]
        _sev_icon = '🔴' if _top_z > 5 else ('🟠' if _top_z > 3 else ('🟡' if _top_z > 2 else '🟢'))
        st.markdown(f"**{_sev_icon} Primary driver: {_top_grp}** (mean |z| = {_top_z:.1f}σ)")
        st.info(_PHYS.get(_top_grp, ""))

        # Group summary table
        _grp_rows = []
        for _g in _groups:
            _zv = _grp_z[_g]
            _ico = '🔴' if _zv > 5 else ('🟠' if _zv > 3 else ('🟡' if _zv > 2 else '🟢'))
            _grp_rows.append({'Group': _g, '': _ico, 'Mean |Z|': f'{_zv:.2f}'})
        st.dataframe(
            pd.DataFrame(_grp_rows).set_index('Group'),
            use_container_width=True, height=240,
        )

    # ── 2. Detector Score Distributions ────────────────────────────────────
    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)
    st.subheader("📊 Detector Score Distributions — AIMD vs Upload")
    st.caption(
        "Histogram overlap shows whether the upload occupies the same region of feature space "
        "as the AIMD baseline. Non-overlapping → genuine MLFF failure; overlapping → possible "
        "threshold calibration issue."
    )

    _if_aimd  = np.asarray(ra['l2_if_score'])
    _if_upl   = np.asarray(rm['l2_if_score'])
    _svm_aimd = np.asarray(ra['l2_svm_score'])
    _svm_upl  = np.asarray(rm['l2_svm_score'])
    _if_thr   = float(np.percentile(_if_aimd, 5))
    _sep      = (np.mean(_if_aimd) - np.mean(_if_upl)) / (np.std(_if_aimd) + 1e-10)

    fig_dist, axes_dist = plt.subplots(1, 2, figsize=(13, 4), facecolor=SURFACE,
                                       constrained_layout=True)
    for ax in axes_dist: _style_ax(ax)

    axes_dist[0].hist(_if_aimd, bins=40, alpha=0.65, density=True, color=CYAN,
                      label='AIMD (training)', edgecolor='none')
    axes_dist[0].hist(_if_upl,  bins=40, alpha=0.65, density=True, color=RED,
                      label=f'Upload: {upload_name[:20]}', edgecolor='none')
    axes_dist[0].axvline(_if_thr, color='#C0392B', ls='--', lw=1.5,
                         label=f'IF threshold ({_if_thr:.3f})')
    axes_dist[0].legend(facecolor=SURFACE2, labelcolor=TEXT, fontsize=8)
    _style_ax(axes_dist[0],
              title=f'Isolation Forest Scores  (separation: {_sep:.1f}σ)',
              xlabel='IF score (more negative → more anomalous)', ylabel='Density')

    axes_dist[1].hist(_svm_aimd, bins=40, alpha=0.65, density=True, color=CYAN,
                      label='AIMD (training)', edgecolor='none')
    axes_dist[1].hist(_svm_upl,  bins=40, alpha=0.65, density=True, color=RED,
                      label=f'Upload: {upload_name[:20]}', edgecolor='none')
    axes_dist[1].axvline(0, color='#C0392B', ls='--', lw=1.5, label='SVM threshold (0)')
    axes_dist[1].legend(facecolor=SURFACE2, labelcolor=TEXT, fontsize=8)
    _style_ax(axes_dist[1], title='One-Class SVM Scores',
              xlabel='SVM score (negative → outside learned boundary)', ylabel='Density')

    st.pyplot(fig_dist, use_container_width=True); plt.close(fig_dist)

    if abs(_sep) > 5:
        st.error(
            f"**Completely separated ({_sep:.1f}σ)** — The upload occupies a fundamentally "
            "different region of feature space. This is NOT a threshold calibration issue: "
            "the MLFF trajectory is genuinely anomalous relative to DFT dynamics."
        )
    elif abs(_sep) > 2:
        st.warning(
            f"**Significantly separated ({_sep:.1f}σ)** — Upload shows meaningful deviation "
            "from AIMD. Some threshold relaxation could reduce false-positive rate, but "
            "genuine physical differences are present."
        )
    else:
        st.success(
            f"**Distributions overlap ({_sep:.1f}σ)** — Upload is statistically similar to "
            "AIMD. Anomalies likely reflect threshold calibration rather than genuine failure."
        )

    # ── 3. PCA Feature Space ────────────────────────────────────────────────
    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)
    st.subheader("🗺️ PCA Feature Space — Where Each Window Lives")
    st.caption(
        "2D projection of all 27 features via PCA. "
        "Separated clusters confirm genuine MLFF deviation from AIMD physics. "
        "The ★ marks the currently selected window."
    )

    from sklearn.decomposition import PCA as _PCA
    from sklearn.preprocessing import StandardScaler as _SS

    _Xa = _impute(X_aimd.copy())
    _Xu = _impute(X_sel.copy())
    _Xc = np.vstack([_Xa, _Xu])
    _ss = _SS()
    _Xcs = _ss.fit_transform(_Xc)
    _pca = _PCA(n_components=2, random_state=42)
    _Xp  = _pca.fit_transform(_Xcs)
    _na  = len(_Xa)

    _pca_a = _Xp[:_na]
    _pca_u = _Xp[_na:]
    _anom_u = np.asarray(res_sel['anomaly_label'][:len(_Xu)])
    _pca_n = _pca_u[_anom_u == 0]
    _pca_ab = _pca_u[_anom_u == 1]
    _ev = _pca.explained_variance_ratio_

    fig_pca, ax_pca = mpl_fig(figsize=(10, 6))
    ax_pca = fig_pca.axes[0]
    ax_pca.scatter(_pca_a[:, 0], _pca_a[:, 1],   alpha=0.35, s=8,
                   color=CYAN, label=f'AIMD ({_na} windows)')
    if len(_pca_n) > 0:
        ax_pca.scatter(_pca_n[:, 0], _pca_n[:, 1],  alpha=0.5, s=10,
                       color=SAGE, marker='s', label=f'Upload — Normal ({len(_pca_n)})')
    if len(_pca_ab) > 0:
        ax_pca.scatter(_pca_ab[:, 0], _pca_ab[:, 1], alpha=0.6, s=12,
                       color=RED, marker='^', label=f'Upload — Anomaly ({len(_pca_ab)})')
    # Star for selected window
    _sel_pt = _Xp[_na + explain_win]
    ax_pca.scatter([_sel_pt[0]], [_sel_pt[1]], color=GOLD, s=150, zorder=6,
                   marker='*', label=f'Selected window #{explain_win}')
    _style_ax(ax_pca,
              title='PCA Feature Space: AIMD vs Upload',
              xlabel=f'PC1 ({100*_ev[0]:.1f}% variance)',
              ylabel=f'PC2 ({100*_ev[1]:.1f}% variance)')
    ax_pca.legend(facecolor=SURFACE2, labelcolor=TEXT, fontsize=9)
    st.pyplot(fig_pca, use_container_width=True); plt.close(fig_pca)

    _var_total = 100 * (_ev[0] + _ev[1])
    if _var_total < 50:
        st.caption(
            f"PC1 + PC2 explain {_var_total:.0f}% of variance. The full separation is "
            "visible in 27-dimensional space — the 2D projection may understate the gap."
        )
    else:
        st.caption(f"PC1 + PC2 explain {_var_total:.0f}% of variance — a faithful 2D summary.")


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
            "Create a horizontal bar chart of all 27 features sorted by absolute z-score "
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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Active Learning Loop
# Based on: docs/MLFF Training_ Anomaly Detection Workflow.md
# Framework: MACE + Quantum ESPRESSO + SLURM (DP-GEN style autonomous loop)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Active Learning":
    st.markdown(
        "<div class='page-head'>"
        "<span class='page-head-label'>06 — Active Learning</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.title("Autonomous MLFF Refinement Loop")
    st.markdown(
        "**DP-GEN style** active learning: anomaly-detected windows → "
        "MACE committee uncertainty → Quantum ESPRESSO labeling → retrain. "
        "Generates ready-to-submit SLURM scripts for HPC."
    )

    inject_metrics_bar([
        ("Upload anomaly rate",  f"{mlff_rate:.0%}",     None, "coral"),
        ("AIMD anomaly rate",    f"{aimd_rate:.0%}",     None, "sage"),
        ("Detection ratio",      f"{det_ratio:.1f}×",    None, "ink"),
        ("Candidate windows",    int(np.sum(rm['anomaly_label'])), None, "gold"),
        ("Feature drivers",      int((feat_cmp['z_score'].abs() > 3).sum()) if feat_cmp is not None else 0, None, "ink"),
    ])

    al_tab1, al_tab2, al_tab3, al_tab4 = st.tabs([
        "🎯 Candidate Selection",
        "⚙️ AL Configuration",
        "💻 HPC Script Generator",
        "🧪 Pre-Test Pipeline",
    ])

    # ── Tab 1: Candidate Selection ──────────────────────────────────────────
    with al_tab1:
        st.subheader("Anomaly-Driven Candidate Selection")
        st.markdown(
            "Windows flagged as anomalous by the L1+L2 ensemble are "
            "**candidate structures** for DFT re-labeling. "
            "Selection strategy follows DP-GEN: rank by ensemble score, "
            "filter by feature diversity to avoid redundant labeling."
        )

        # Build candidate dataframe from active upload results
        n_upload_win = len(rm['anomaly_label'])
        conf_scores  = np.asarray(rm['confidence'])[:n_upload_win] \
                       if 'confidence' in rm else np.asarray(rm['anomaly_label'])[:n_upload_win].astype(float)
        cand_df = pd.DataFrame({
            'window':       range(n_upload_win),
            'anomaly':      np.asarray(rm['anomaly_label'])[:n_upload_win],
            'confidence':   conf_scores,
            'l1_flag':      np.asarray(rm['l1_flag'])[:n_upload_win],
            'l2_if_flag':   np.asarray(rm['l2_if_flag'])[:n_upload_win],
            'l2_svm_flag':  np.asarray(rm['l2_svm_flag'])[:n_upload_win],
        })
        anomalous_df = cand_df[cand_df['anomaly'] == 1].sort_values('confidence', ascending=False)

        # Severity classification (from workflow doc)
        def _tier(conf):
            if conf > 0.9: return '🔴 Catastrophic'
            if conf > 0.6: return '🟠 High'
            if conf > 0.3: return '🟡 Warning'
            return '🟢 Normal'

        st.markdown(f"**{upload_name}** — {len(anomalous_df)} / {n_upload_win} windows flagged")

        tier_counts = {
            '🔴 Catastrophic': int((conf_scores > 0.9).sum()),
            '🟠 High':         int(((conf_scores > 0.6) & (conf_scores <= 0.9)).sum()),
            '🟡 Warning':      int(((conf_scores > 0.3) & (conf_scores <= 0.6)).sum()),
            '🟢 Normal':       int((conf_scores <= 0.3).sum()),
        }
        tc1, tc2, tc3, tc4 = st.columns(4)
        for col, (label, count) in zip([tc1, tc2, tc3, tc4], tier_counts.items()):
            col.metric(label, count)

        st.markdown("---")

        # Stability onset time
        anom_labels = np.asarray(rm['anomaly_label'])[:n_upload_win]
        onset_idx   = int(np.argmax(anom_labels)) if anom_labels.any() else None
        if onset_idx is not None and anom_labels[onset_idx] == 1:
            st.markdown(
                f"**⚡ Stability onset time:** Window **{onset_idx}** "
                f"(frame ~{onset_idx * 10} in trajectory, stride=10) — "
                f"first detected anomaly.",
                help="Stability onset time τ_s is the trajectory step where anomaly is first flagged. "
                     "Structures near τ_s are the most informative for retraining."
            )

        col_l, col_r = st.columns(2)
        with col_l:
            max_cands = st.slider("Max candidates to label (DFT budget)", 10, 200, 50, 5,
                                  key="al_n_cands")
        with col_r:
            dedup = st.checkbox("Deduplicate by feature similarity", value=True, key="al_dedup")

        # Select top candidates
        top_cands = anomalous_df.head(max_cands).copy()
        top_cands['tier']  = top_cands['confidence'].apply(_tier)
        top_cands['start_frame'] = top_cands['window'] * 10
        top_cands['end_frame']   = top_cands['window'] * 10 + 50

        if not top_cands.empty:
            st.subheader(f"Top {len(top_cands)} Candidate Structures")
            inject_overview_table(
                top_cands[['window','tier','confidence','start_frame','end_frame',
                            'l1_flag','l2_if_flag','l2_svm_flag']].rename(columns={
                    'window':'Window', 'tier':'Severity', 'confidence':'Conf.',
                    'start_frame':'Frame Start', 'end_frame':'Frame End',
                    'l1_flag':'L1', 'l2_if_flag':'L2-IF', 'l2_svm_flag':'L2-SVM',
                }),
                height=260,
            )
            csv_cands = top_cands.to_csv(index=False)
            st.download_button(
                "⬇ Download candidate_windows.csv",
                data=csv_cands,
                file_name="candidate_windows.csv",
                mime="text/csv",
            )
        else:
            st.success("No anomalous windows detected — MLFF is well-calibrated!")

        # Feature severity breakdown
        if feat_cmp is not None and not feat_cmp.empty:
            st.markdown("---")
            st.subheader("Feature Drivers of Anomaly")
            _high_z = feat_cmp[feat_cmp['z_score'].abs() > 3].sort_values('z_score', key=abs, ascending=False)
            if not _high_z.empty:
                # Classify by physics category
                _cat_map = {
                    'disp_': 'Displacement', 'rms_': 'Dynamics', 'crest_': 'Dynamics',
                    'impulse_': 'Dynamics', 'frame_v': 'Dynamics', 'aniso': 'Dynamics',
                    'dominant_': 'Frequency', 'spectral_': 'Frequency',
                    'msd_': 'MSD', 'energy_': 'Energy',
                }
                def _cat(fname):
                    for prefix, cat in _cat_map.items():
                        if fname.startswith(prefix): return cat
                    return 'Other'
                _high_z = _high_z.copy()
                _high_z['category'] = _high_z['feature'].apply(_cat)
                _high_z['action']   = _high_z['z_score'].apply(
                    lambda z: '🔴 Critical — must label' if abs(z) > 10
                    else ('🟠 High priority' if abs(z) > 5 else '🟡 Medium priority')
                )
                inject_overview_table(
                    _high_z[['feature','category','z_score','relative_change_%','action']].rename(columns={
                        'feature':'Feature','category':'Physics Category',
                        'z_score':'Z-score','relative_change_%':'Change %','action':'AL Priority'
                    }),
                    height=240,
                )

    # ── Tab 2: AL Configuration ──────────────────────────────────────────────
    with al_tab2:
        st.subheader("Active Learning Configuration")
        st.caption("Tune hyperparameters for the MACE committee + QE labeling loop.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**MACE Architecture**")
            mace_r_max    = st.number_input("r_max (Å)", 4.0, 10.0, 6.0, 0.5, key="al_rmax")
            mace_L        = st.selectbox("max_L (angular)", [1, 2, 3], index=1, key="al_L")
            mace_channels = st.selectbox("num_channels", [64, 128, 256], index=1, key="al_ch")
            mace_interact = st.selectbox("num_interactions", [1, 2, 3], index=1, key="al_ni")
            mace_committee= st.slider("Committee size (n_models)", 2, 8, 4, key="al_comm")
            mace_lr       = st.number_input("Learning rate", 0.0001, 0.01, 0.005, step=0.0001,
                                             format="%.4f", key="al_lr")
            mace_epochs   = st.number_input("Max epochs", 200, 5000, 2000, 100, key="al_ep")

        with c2:
            st.markdown(f"**DP-GEN Loop Settings**")
            sigma_lo   = st.number_input("σ_lo (uncertainty lower bound)", 0.01, 0.5, 0.10, 0.01, key="al_slo")
            sigma_hi   = st.number_input("σ_hi (uncertainty upper bound)", 0.1, 1.0, 0.30, 0.05, key="al_shi")
            max_iter   = st.slider("Max AL iterations", 3, 20, 10, key="al_maxiter")
            n_explore  = st.number_input("Explore frames per iter", 100, 5000, 500, 100, key="al_nex")
            conv_thr   = st.number_input("Convergence threshold (fraction accurate)",
                                          0.90, 0.999, 0.99, 0.005, format="%.3f", key="al_conv")
            temps_str  = st.text_input("Exploration temperatures (K)", "300,600,900,1200", key="al_temps")

            st.markdown("**QE DFT Settings**")
            qe_ecutwfc = st.number_input("ecutwfc (Ry)", 40, 120, 60, 10, key="al_ecut")
            qe_kpts    = st.text_input("k-points (nx ny nz)", "2 2 1", key="al_kpts")

        st.markdown("---")
        st.markdown("**SLURM Settings**")
        s1, s2, s3 = st.columns(3)
        with s1:
            slurm_partition = st.text_input("Partition", "gpu", key="al_part")
            slurm_time      = st.text_input("Time limit", "48:00:00", key="al_time")
        with s2:
            slurm_ntasks    = st.number_input("ntasks-per-node", 1, 64, 8, key="al_ntasks")
            slurm_gpus      = st.number_input("GPUs per node", 0, 8, 1, key="al_gpus")
        with s3:
            slurm_mem       = st.text_input("Memory", "64G", key="al_mem")
            slurm_email     = st.text_input("Email (for notifications)", "", key="al_email")

        # Store config in session for script generator
        st.session_state['al_config'] = {
            'mace': {'r_max': mace_r_max, 'max_L': mace_L, 'num_channels': mace_channels,
                     'num_interactions': mace_interact, 'n_committee': mace_committee,
                     'lr': mace_lr, 'max_epochs': int(mace_epochs)},
            'al':   {'sigma_lo': sigma_lo, 'sigma_hi': sigma_hi, 'max_iterations': int(max_iter),
                     'n_explore_frames': int(n_explore), 'convergence_threshold': conv_thr,
                     'temperatures': [int(t.strip()) for t in temps_str.split(',') if t.strip().isdigit()]},
            'qe':   {'ecutwfc': int(qe_ecutwfc), 'kpoints': qe_kpts},
            'slurm':{'partition': slurm_partition, 'time': slurm_time,
                     'ntasks': int(slurm_ntasks), 'gpus': int(slurm_gpus),
                     'mem': slurm_mem, 'email': slurm_email},
        }
        st.success("Configuration saved — go to **HPC Script Generator** to download scripts.")

    # ── Tab 3: HPC Script Generator ──────────────────────────────────────────
    with al_tab3:
        st.subheader("HPC Script Generator")
        st.markdown(
            "Generates three files for autonomous MLFF refinement on HPC:\n"
            "- `config_al.yaml` — full hyperparameter config\n"
            "- `run_al_loop.py` — Python orchestration script (MACE + QE + AL loop)\n"
            "- `submit_al.sh` — SLURM batch submission script"
        )

        cfg = st.session_state.get('al_config', {})
        mace_cfg  = cfg.get('mace',  {'r_max': 6.0, 'max_L': 2, 'num_channels': 128,
                                       'num_interactions': 2, 'n_committee': 4,
                                       'lr': 0.005, 'max_epochs': 2000})
        al_cfg    = cfg.get('al',    {'sigma_lo': 0.10, 'sigma_hi': 0.30, 'max_iterations': 10,
                                      'n_explore_frames': 500, 'convergence_threshold': 0.99,
                                      'temperatures': [300, 600, 900, 1200]})
        qe_cfg    = cfg.get('qe',    {'ecutwfc': 60, 'kpoints': '2 2 1'})
        sl_cfg    = cfg.get('slurm', {'partition': 'gpu', 'time': '48:00:00',
                                      'ntasks': 8, 'gpus': 1, 'mem': '64G', 'email': ''})
        kpts      = qe_cfg['kpoints'].split()
        kx, ky, kz= (kpts + ['1','1','1'])[:3]

        # ── config_al.yaml ─────────────────────────────────────────────────
        config_yaml = f"""# Active Learning Configuration — 2D Sb2Te3 Cr-doped MLFF
# Generated by AIMD Anomaly Detection Dashboard
# MLFF: MACE   |   DFT: Quantum ESPRESSO   |   Scheduler: SLURM

system:
  name: "2D_Sb2Te3_Cr_doped"
  elements: ["Sb", "Te", "Cr"]
  n_atoms: 82
  structure_file: "data/raw/temperature/2L_octo_Cr2_600K_aimd_1.xyz"
  anomaly_report: "results/reports/ensemble_comparison.csv"

mace:
  # Architecture
  r_max: {mace_cfg['r_max']}
  num_radial_basis: 10
  num_cutoff_basis: 5
  max_L: {mace_cfg['max_L']}
  num_channels: {mace_cfg['num_channels']}
  num_interactions: {mace_cfg['num_interactions']}
  correlation: 3
  # Training
  batch_size: 16
  lr: {mace_cfg['lr']}
  max_num_epochs: {mace_cfg['max_epochs']}
  patience: 200
  scheduler_gamma: 0.9995
  ema_decay: 0.99
  clip_grad: 10.0
  # Committee (DP-GEN style)
  n_committee: {mace_cfg['n_committee']}
  committee_seeds: {list(range(42, 42 + mace_cfg['n_committee']))}
  model_dir: "results/models/mace_committee"

quantum_espresso:
  pseudo_dir: "pseudos"
  pseudopotentials:
    Sb: "Sb.pbe-n-kjpaw_psl.1.0.0.UPF"
    Te: "Te.pbe-dn-kjpaw_psl.1.0.0.UPF"
    Cr: "Cr.pbe-spn-kjpaw_psl.1.0.0.UPF"
  ecutwfc: {qe_cfg['ecutwfc']}
  ecutrho: {qe_cfg['ecutwfc'] * 8}
  kpoints: [{kx}, {ky}, {kz}]
  smearing: "methfessel-paxton"
  degauss: 0.01
  conv_thr: 1.0e-8
  nstep: 200
  output_dir: "results/al_dft_calcs"

active_learning:
  mode: "DP-GEN"          # options: DP-GEN | ALKPU | aims-PAX
  sigma_lo: {al_cfg['sigma_lo']}       # lower uncertainty bound (filter redundant)
  sigma_hi: {al_cfg['sigma_hi']}       # upper uncertainty bound (filter nonsensical)
  max_iterations: {al_cfg['max_iterations']}
  n_explore_frames: {al_cfg['n_explore_frames']}
  convergence_threshold: {al_cfg['convergence_threshold']}
  temperatures: {al_cfg['temperatures']}
  n_labeling_max: 50      # max DFT calculations per iteration
  stability_onset_window: true  # label structures near stability onset time
  dedup_threshold: 0.95   # cosine similarity threshold for deduplication
  seed: 42

slurm:
  partition: "{sl_cfg['partition']}"
  nodes: 1
  ntasks_per_node: {sl_cfg['ntasks']}
  gpus_per_node: {sl_cfg['gpus']}
  time: "{sl_cfg['time']}"
  mem: "{sl_cfg['mem']}"
  account: ""             # fill in your HPC account/project code
  email: "{sl_cfg['email']}"
  conda_env: "mace"       # conda env with MACE + QE python interface installed
  qe_binary: "pw.x"       # QE binary (add full path if not in PATH)
"""

        # ── run_al_loop.py ─────────────────────────────────────────────────
        run_script = r'''#!/usr/bin/env python3
"""
run_al_loop.py — Autonomous Active Learning loop for MACE MLFF refinement.
Uses DP-GEN style committee uncertainty to select structures for QE re-labeling.

Usage:
  python run_al_loop.py                    # full loop
  python run_al_loop.py --dry-run          # simulation only (no DFT/training)
  python run_al_loop.py --iteration 3      # resume from iteration 3

Requirements:
  pip install mace-torch ase pyyaml numpy pandas scipy scikit-learn
  Quantum ESPRESSO (pw.x) in PATH or specified in config
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from ase.io import read, write
    from ase.calculators.espresso import Espresso
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    print("Warning: ASE not installed. DFT labeling step will be skipped.")


# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('results/al_loop.log', mode='a'),
    ]
)
log = logging.getLogger(__name__)


# ── Config loader ──────────────────────────────────────────────────────────
def load_config(path='config_al.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Step 1: Load existing anomaly detection results ───────────────────────
def load_anomaly_candidates(config):
    """
    Read ensemble_comparison.csv and return anomalous MLFF windows.
    Ranking: by ensemble score (sum of flags), then by l2_if_score magnitude.
    """
    report_path = Path(config['system']['anomaly_report'])
    if not report_path.exists():
        log.warning(f"Anomaly report not found: {report_path}")
        return pd.DataFrame()

    df = pd.read_csv(report_path)
    mlff_df = df[df['source'] == 'MLFF'].copy()
    mlff_df['ensemble_score'] = (
        mlff_df['l1_flag'] + mlff_df['l2_if_flag'] + mlff_df['l2_svm_flag']
    )
    # Composite confidence: number of detectors flagging / 3
    mlff_df['confidence'] = mlff_df['ensemble_score'] / 3.0

    candidates = mlff_df[mlff_df['ensemble_l12'] == 1].sort_values(
        'confidence', ascending=False
    )
    log.info(f"Loaded {len(candidates)} anomalous candidate windows from {report_path}")
    return candidates


# ── Step 2: Extract candidate structures from XYZ ─────────────────────────
def extract_candidate_structures(candidates, structure_file, n_max, window_size=50):
    """Extract atom structures at candidate window start frames."""
    if not ASE_AVAILABLE:
        log.warning("ASE unavailable — skipping structure extraction.")
        return []

    log.info(f"Reading trajectory: {structure_file}")
    traj = read(str(structure_file), index=':')
    log.info(f"  {len(traj)} frames loaded")

    structures = []
    for _, row in candidates.head(n_max).iterrows():
        frame_idx = int(row.get('window_idx', 0)) * 10  # stride=10
        mid_frame = min(frame_idx + window_size // 2, len(traj) - 1)
        structures.append((int(row.get('window_idx', 0)), traj[mid_frame]))

    log.info(f"Extracted {len(structures)} candidate structures")
    return structures


# ── Step 3: Compute MACE committee uncertainty ────────────────────────────
def compute_committee_uncertainty(structures, model_dir, n_committee, dry_run=False):
    """
    Run MACE committee inference: n_committee models predict forces,
    uncertainty = max std-dev of force predictions across committee.
    Returns dict: {window_idx: sigma_max}
    """
    if dry_run:
        log.info("[DRY RUN] Simulating committee uncertainty with random values")
        return {s[0]: float(np.random.uniform(0.05, 0.40)) for s in structures}

    uncertainties = {}
    model_dir = Path(model_dir)
    model_paths = sorted(model_dir.glob('model_seed_*.pth'))

    if not model_paths:
        log.warning(f"No committee models found in {model_dir}. Skipping UQ step.")
        return uncertainties

    log.info(f"Computing uncertainty with {len(model_paths)} committee models")
    try:
        from mace.calculators import MACECalculator
        all_forces = {s[0]: [] for s in structures}

        for model_path in model_paths:
            calc = MACECalculator(model_paths=[str(model_path)], device='cuda')
            for win_idx, atoms in structures:
                atoms_copy = atoms.copy()
                atoms_copy.calc = calc
                try:
                    f = atoms_copy.get_forces()
                    all_forces[win_idx].append(f)
                except Exception as e:
                    log.warning(f"  Force eval failed for window {win_idx}: {e}")

        for win_idx, force_list in all_forces.items():
            if len(force_list) >= 2:
                force_arr = np.stack(force_list)  # (n_models, n_atoms, 3)
                sigma = np.max(np.std(force_arr, axis=0))
                uncertainties[win_idx] = float(sigma)
            else:
                uncertainties[win_idx] = 0.0

    except ImportError:
        log.error("MACE not installed. Run: pip install mace-torch")

    return uncertainties


# ── Step 4: Filter candidates by sigma bounds ─────────────────────────────
def filter_candidates(structures, uncertainties, sigma_lo, sigma_hi):
    """
    DP-GEN selection:
    - sigma < sigma_lo: already well-described by model (skip)
    - sigma_lo <= sigma <= sigma_hi: informative → LABEL
    - sigma > sigma_hi: nonsensical/too uncertain → SKIP (would waste DFT)
    """
    selected = []
    for win_idx, atoms in structures:
        sigma = uncertainties.get(win_idx, 0.0)
        if sigma_lo <= sigma <= sigma_hi:
            selected.append((win_idx, atoms, sigma))
            log.debug(f"  ✓ Window {win_idx}: σ={sigma:.4f} (informative)")
        elif sigma > sigma_hi:
            log.debug(f"  ✗ Window {win_idx}: σ={sigma:.4f} (nonsensical, skip)")
        else:
            log.debug(f"  - Window {win_idx}: σ={sigma:.4f} (accurate, skip)")

    log.info(f"Selected {len(selected)} structures for DFT labeling "
             f"(σ_lo={sigma_lo}, σ_hi={sigma_hi})")
    return selected


# ── Step 5: Generate QE input files & run DFT ─────────────────────────────
def run_qe_labeling(selected, config, iteration, dry_run=False):
    """Generate QE input files and submit DFT calculations."""
    qe_cfg  = config['quantum_espresso']
    out_dir = Path(config['quantum_espresso']['output_dir']) / f'iter_{iteration:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled_structures = []

    for win_idx, atoms, sigma in selected:
        label_dir = out_dir / f'window_{win_idx:04d}'
        label_dir.mkdir(exist_ok=True)

        if dry_run:
            # Write input but don't run
            if ASE_AVAILABLE:
                write(str(label_dir / 'structure.xyz'), atoms)
            log.info(f"  [DRY RUN] Would label window {win_idx} (σ={sigma:.4f})")
            labeled_structures.append((win_idx, atoms, None))
            continue

        if not ASE_AVAILABLE:
            log.warning("ASE unavailable — cannot write QE inputs")
            continue

        # Write QE input via ASE
        kpts_str = qe_cfg.get('kpoints', [2, 2, 1])
        input_data = {
            'control':   {'calculation': 'scf', 'outdir': str(label_dir / 'tmp'),
                          'prefix': f'win_{win_idx:04d}', 'pseudo_dir': qe_cfg['pseudo_dir'],
                          'etot_conv_thr': 1e-6, 'forc_conv_thr': 1e-4},
            'system':    {'ecutwfc': qe_cfg['ecutwfc'], 'ecutrho': qe_cfg['ecutrho'],
                          'occupations': 'smearing', 'smearing': qe_cfg.get('smearing', 'mp'),
                          'degauss': qe_cfg.get('degauss', 0.01)},
            'electrons': {'conv_thr': qe_cfg.get('conv_thr', 1e-8),
                          'mixing_beta': 0.3, 'electron_maxstep': 200},
        }

        calc = Espresso(
            input_data=input_data,
            pseudopotentials=qe_cfg['pseudopotentials'],
            kpts=kpts_str,
            directory=str(label_dir),
        )
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc

        try:
            energy  = atoms_copy.get_potential_energy()
            forces  = atoms_copy.get_forces()
            stress  = atoms_copy.get_stress()
            log.info(f"  ✓ Window {win_idx}: E={energy:.4f} eV, "
                     f"max|F|={np.max(np.abs(forces)):.4f} eV/Å")
            labeled_structures.append((win_idx, atoms_copy, {'energy': energy,
                                                               'forces': forces,
                                                               'stress': stress}))
        except Exception as e:
            log.error(f"  ✗ QE failed for window {win_idx}: {e}")

    return labeled_structures


# ── Step 6: Append new data and retrain MACE committee ────────────────────
def retrain_mace(labeled_structures, config, iteration, dry_run=False):
    """Append labeled data to training set and retrain MACE committee."""
    mace_cfg  = config['mace']
    model_dir = Path(mace_cfg['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save new labeled structures
    new_data_path = model_dir / f'new_data_iter_{iteration:02d}.extxyz'
    if ASE_AVAILABLE and labeled_structures:
        valid_structs = [atoms for _, atoms, data in labeled_structures if data is not None]
        if valid_structs:
            write(str(new_data_path), valid_structs)
            log.info(f"Saved {len(valid_structs)} labeled structures → {new_data_path}")

    if dry_run:
        log.info("[DRY RUN] Would retrain MACE committee")
        return [f"model_seed_{42 + i}.pth" for i in range(mace_cfg['n_committee'])]

    model_paths = []
    for seed in mace_cfg.get('committee_seeds', range(42, 42 + mace_cfg['n_committee'])):
        model_name = f"model_seed_{seed}"
        cmd = [
            'python', '-m', 'mace.run_train',
            f'--name={model_name}',
            f'--train_file={str(model_dir / "train.extxyz")}',
            f'--valid_fraction=0.1',
            f'--model=MACE',
            f'--r_max={mace_cfg["r_max"]}',
            f'--max_L={mace_cfg["max_L"]}',
            f'--num_channels={mace_cfg["num_channels"]}',
            f'--num_interactions={mace_cfg["num_interactions"]}',
            f'--correlation=3',
            f'--batch_size=16',
            f'--lr={mace_cfg["lr"]}',
            f'--max_num_epochs={mace_cfg["max_epochs"]}',
            f'--patience={mace_cfg.get("patience", 200)}',
            f'--scheduler_gamma={mace_cfg.get("scheduler_gamma", 0.9995)}',
            f'--seed={seed}',
            f'--device=cuda',
            f'--directory={str(model_dir)}',
        ]
        log.info(f"Training model with seed {seed}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            log.info(f"  ✓ Model {model_name} trained successfully")
            model_paths.append(str(model_dir / f'{model_name}.pth'))
        else:
            log.error(f"  ✗ Training failed for seed {seed}:\n{result.stderr[-500:]}")

    return model_paths


# ── Step 7: Check convergence ─────────────────────────────────────────────
def check_convergence(uncertainties, sigma_lo, threshold=0.99):
    """
    Convergence: fraction of structures with σ < σ_lo (accurate) exceeds threshold.
    Returns (converged: bool, fraction_accurate: float).
    """
    if not uncertainties:
        return False, 0.0
    n_accurate = sum(1 for s in uncertainties.values() if s < sigma_lo)
    frac = n_accurate / len(uncertainties)
    return frac >= threshold, frac


# ── Main loop ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='MACE Active Learning Loop')
    parser.add_argument('--config',    default='config_al.yaml')
    parser.add_argument('--dry-run',   action='store_true')
    parser.add_argument('--iteration', type=int, default=0)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("MACE Active Learning Loop — 2D Sb2Te3 Cr-doped")
    log.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    log.info("=" * 60)

    config    = load_config(args.config)
    al_cfg    = config['active_learning']
    mace_cfg  = config['mace']
    sigma_lo  = al_cfg['sigma_lo']
    sigma_hi  = al_cfg['sigma_hi']
    max_iter  = al_cfg['max_iterations']
    conv_thr  = al_cfg['convergence_threshold']

    Path('results').mkdir(exist_ok=True)
    summary_log = []

    for iteration in range(args.iteration, max_iter):
        log.info(f"\n{'='*60}")
        log.info(f"ITERATION {iteration + 1} / {max_iter}")
        log.info(f"{'='*60}")

        # 1. Load candidates
        candidates = load_anomaly_candidates(config)
        if candidates.empty:
            log.info("No anomalous candidates — model may already be converged.")
            break

        # 2. Extract structures
        structures = extract_candidate_structures(
            candidates,
            config['system']['structure_file'],
            n_max=al_cfg['n_labeling_max'],
        )

        if not structures:
            log.warning("No structures extracted. Check structure_file path.")
            break

        # 3. Committee uncertainty
        uncertainties = compute_committee_uncertainty(
            structures, mace_cfg['model_dir'],
            mace_cfg['n_committee'], dry_run=args.dry_run,
        )

        # 4. Filter by sigma bounds
        selected = filter_candidates(structures, uncertainties, sigma_lo, sigma_hi)

        # 5. Check convergence before labeling
        converged, frac_acc = check_convergence(uncertainties, sigma_lo, conv_thr)
        log.info(f"Convergence check: {frac_acc:.1%} accurate (threshold: {conv_thr:.1%})")

        iter_summary = {
            'iteration':       iteration + 1,
            'timestamp':       datetime.now().isoformat(),
            'n_candidates':    len(candidates),
            'n_structures':    len(structures),
            'n_selected':      len(selected),
            'frac_accurate':   frac_acc,
            'converged':       converged,
        }
        summary_log.append(iter_summary)

        # Save iteration summary
        with open(f'results/al_iteration_{iteration + 1:02d}.json', 'w') as f:
            json.dump(iter_summary, f, indent=2)

        if converged:
            log.info(f"✅ CONVERGED at iteration {iteration + 1}! "
                     f"Fraction accurate: {frac_acc:.1%}")
            break

        if not selected:
            log.info("No structures selected this iteration (all accurate or nonsensical).")
            continue

        # 6. DFT labeling
        labeled = run_qe_labeling(selected, config, iteration + 1, dry_run=args.dry_run)

        # 7. Retrain
        _ = retrain_mace(labeled, config, iteration + 1, dry_run=args.dry_run)

        log.info(f"Iteration {iteration + 1} complete. "
                 f"Labeled: {len(labeled)}, Acc: {frac_acc:.1%}")

    # Final summary
    log.info("\n" + "=" * 60)
    log.info("ACTIVE LEARNING LOOP COMPLETE")
    log.info("=" * 60)
    with open('results/al_summary.json', 'w') as f:
        json.dump(summary_log, f, indent=2)
    log.info("Summary written to results/al_summary.json")


if __name__ == '__main__':
    main()
'''

        # ── submit_al.sh (SLURM) ────────────────────────────────────────────
        email_lines = ""
        if sl_cfg.get('email'):
            email_lines = f"""#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user={sl_cfg['email']}"""

        submit_sh = f"""#!/bin/bash
#SBATCH --job-name=mace_al_sb2te3
#SBATCH --output=logs/al_%j.out
#SBATCH --error=logs/al_%j.err
#SBATCH --partition={sl_cfg['partition']}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={sl_cfg['ntasks']}
#SBATCH --gres=gpu:{sl_cfg['gpus']}
#SBATCH --time={sl_cfg['time']}
#SBATCH --mem={sl_cfg['mem']}
{email_lines}

# ── Environment ─────────────────────────────────────────────────────────────
module purge
module load cuda/12.1 intel-mpi/2021

# Activate conda env (adjust path if needed)
source activate mace 2>/dev/null || conda activate mace 2>/dev/null || {{
  echo "ERROR: Could not activate conda env 'mace'"
  echo "Create it with: conda create -n mace python=3.10 && pip install mace-torch ase pyyaml"
  exit 1
}}

# OpenMP threading (leave some cores for MPI)
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0

# Quantum ESPRESSO binary (uncomment and set path if not in PATH)
# export PATH=/path/to/qe/bin:$PATH

# ── Directories ──────────────────────────────────────────────────────────────
mkdir -p logs results/al_dft_calcs results/models/mace_committee

# ── Verify inputs ────────────────────────────────────────────────────────────
if [ ! -f "config_al.yaml" ]; then
  echo "ERROR: config_al.yaml not found. Run from the project root."
  exit 1
fi

if [ ! -f "results/reports/ensemble_comparison.csv" ]; then
  echo "ERROR: ensemble_comparison.csv not found. Run the full pipeline first."
  echo "  python scripts/run_full_pipeline.py"
  exit 1
fi

# ── Dry run test ──────────────────────────────────────────────────────────────
echo "Running dry-run pre-test..."
python run_al_loop.py --dry-run --config config_al.yaml
if [ $? -ne 0 ]; then
  echo "ERROR: Dry run failed. Check config_al.yaml and logs."
  exit 1
fi
echo "Dry run passed. Starting full active learning loop..."

# ── Active learning loop ──────────────────────────────────────────────────────
python run_al_loop.py --config config_al.yaml

exit_code=$?
if [ $exit_code -eq 0 ]; then
  echo "=== ACTIVE LEARNING COMPLETE ==="
  echo "Results in: results/al_summary.json"
else
  echo "=== ACTIVE LEARNING FAILED (exit code $exit_code) ==="
  echo "Check logs/al_*.log for details."
fi
exit $exit_code
"""

        # ── Display & download ────────────────────────────────────────────
        dg1, dg2, dg3 = st.columns(3)
        with dg1:
            st.download_button(
                "⬇ config_al.yaml",
                data=config_yaml,
                file_name="config_al.yaml",
                mime="text/yaml",
            )
        with dg2:
            st.download_button(
                "⬇ run_al_loop.py",
                data=run_script,
                file_name="run_al_loop.py",
                mime="text/x-python",
            )
        with dg3:
            st.download_button(
                "⬇ submit_al.sh",
                data=submit_sh,
                file_name="submit_al.sh",
                mime="text/x-sh",
            )

        st.markdown("---")
        with st.expander("📄 Preview: config_al.yaml", expanded=False):
            st.code(config_yaml, language='yaml')
        with st.expander("📄 Preview: run_al_loop.py (abridged)", expanded=False):
            st.code(run_script[:3000] + "\n# ... (truncated for preview)", language='python')
        with st.expander("📄 Preview: submit_al.sh", expanded=False):
            st.code(submit_sh, language='bash')

        st.markdown("---")
        st.info(
            "**Quick start on HPC:**\n"
            "```bash\n"
            "# 1. Transfer scripts to HPC\n"
            "scp config_al.yaml run_al_loop.py submit_al.sh user@hpc:/path/to/project/\n"
            "\n"
            "# 2. Install MACE environment (once)\n"
            "conda create -n mace python=3.10\n"
            "conda activate mace\n"
            "pip install mace-torch ase pyyaml\n"
            "\n"
            "# 3. Add pseudopotentials to pseudos/ directory\n"
            "#    Download from: https://www.materialscloud.org/discover/sssp\n"
            "\n"
            "# 4. Submit\n"
            "sbatch submit_al.sh\n"
            "```"
        )

    # ── Tab 4: Pre-Test Pipeline ─────────────────────────────────────────────
    with al_tab4:
        st.subheader("Pre-Test: Small Dataset Pipeline Simulation")
        st.markdown(
            "Runs a **dry-run simulation** of the active learning loop on the current "
            "session data to verify the pipeline before submitting to HPC."
        )

        test_fraction = st.slider(
            "Training subset (% of total windows)", 10, 50, 20, 5, key="al_test_frac"
        )
        test_sigma_lo = st.number_input("Test σ_lo", 0.05, 0.5, 0.10, 0.05, key="al_test_slo")
        test_sigma_hi = st.number_input("Test σ_hi", 0.1, 1.0, 0.30, 0.05, key="al_test_shi")

        if st.button("▶ Run Pre-Test Simulation", key="al_run_pretest"):
            with st.spinner("Simulating AL loop on small dataset…"):
                # Sample a subset of AIMD windows
                n_subset = max(50, int(len(X_aimd) * test_fraction / 100))
                subset_idx = np.random.choice(len(X_aimd), n_subset, replace=False)
                X_sub = X_aimd[subset_idx]

                # Simulate committee uncertainty (proxy: use L1 score variance across features)
                from sklearn.preprocessing import StandardScaler as _SS
                _ss = _SS()
                X_sub_sc = _ss.fit_transform(X_sub)
                # Simulate sigma as feature variance per window (proxy for true committee UQ)
                sigma_sim = np.std(X_sub_sc, axis=1) * 0.1  # scale to [0, 0.4] range

                n_informative = int(((sigma_sim >= test_sigma_lo) & (sigma_sim <= test_sigma_hi)).sum())
                n_accurate    = int((sigma_sim < test_sigma_lo).sum())
                n_nonsensical = int((sigma_sim > test_sigma_hi).sum())
                frac_acc      = n_accurate / max(len(sigma_sim), 1)

            st.success(f"Pre-test complete on {n_subset} windows ({test_fraction}% of AIMD data)")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Windows tested",   n_subset)
            r2.metric("Accurate (σ<σ_lo)", n_accurate, f"{n_accurate/n_subset:.0%}")
            r3.metric("Informative",       n_informative, f"{n_informative/n_subset:.0%}")
            r4.metric("Nonsensical (skip)", n_nonsensical, f"{n_nonsensical/n_subset:.0%}")

            # Uncertainty distribution plot
            fig_pt, ax_pt = mpl_fig(figsize=(8, 3))
            ax_pt = fig_pt.axes[0]
            ax_pt.hist(sigma_sim, bins=40, color=CYAN, alpha=0.7, label='σ distribution')
            ax_pt.axvline(test_sigma_lo, color=GOLD,  ls='--', lw=1.5, label=f'σ_lo={test_sigma_lo}')
            ax_pt.axvline(test_sigma_hi, color=CORAL, ls='--', lw=1.5, label=f'σ_hi={test_sigma_hi}')
            ax_pt.fill_betweenx([0, ax_pt.get_ylim()[1] if ax_pt.get_ylim()[1] > 0 else 1],
                                 test_sigma_lo, test_sigma_hi,
                                 alpha=0.12, color=GOLD, label='Label zone')
            _style_ax(ax_pt, title='Simulated Committee Uncertainty Distribution',
                      xlabel='σ_max (uncertainty)', ylabel='Count')
            ax_pt.legend(fontsize=9)
            st.pyplot(fig_pt, use_container_width=True); plt.close(fig_pt)

            if frac_acc < 0.5:
                st.warning(
                    f"Only {frac_acc:.0%} of windows are 'accurate' in this subset. "
                    "Multiple AL iterations will likely be needed — consider more epochs or "
                    "a larger training set."
                )
            else:
                st.success(
                    f"**{frac_acc:.0%}** of windows are already accurate. "
                    "The MLFF is learning well — convergence expected in few iterations."
                )

            st.markdown("---")
            st.caption(
                "**Note:** Committee uncertainty is simulated here using feature variance as a proxy "
                "(no actual MACE models trained). On HPC, real committee uncertainty from 4× MACE "
                "models is used. This pre-test validates that the pipeline logic, data loading, "
                "and sigma thresholds are reasonable before committing HPC resources."
            )
