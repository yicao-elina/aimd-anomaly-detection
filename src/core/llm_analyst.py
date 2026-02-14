"""
LLM-powered analysis for AIMD anomaly detection.

Uses Ollama (glm-5:cloud) to:
  1. Generate matplotlib/pandas code from natural language queries
  2. Execute that code safely against real data
  3. Produce evidence-grounded mechanism analysis (no hallucination)

Anti-hallucination contract:
  - The LLM receives only actual computed statistics in its prompt.
  - Every mechanism claim must cite a specific number from those statistics.
  - Figures are produced by executing code against real arrays — not described.
"""

import io
import re
import traceback
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ollama


# ---------------------------------------------------------------------------
# Data context builder
# ---------------------------------------------------------------------------

def build_data_context(
    X_aimd: np.ndarray,
    X_mlff: np.ndarray,
    feature_names: List[str],
    results_aimd: Dict,
    results_mlff: Dict,
    meta_aimd: pd.DataFrame,
    feat_comparison: pd.DataFrame,
) -> str:
    """
    Build a compact, factual data context string to pass to the LLM.
    Every number here comes from actual computation — no fabrication.
    """
    n_aimd = len(X_aimd)
    n_mlff = len(X_mlff)
    aimd_rate = float(np.mean(results_aimd['anomaly_label']))
    mlff_rate = float(np.mean(results_mlff['anomaly_label']))

    # Per-feature stats
    aimd_means = np.nanmean(X_aimd, axis=0)
    mlff_means = np.nanmean(X_mlff, axis=0)
    aimd_stds  = np.nanstd(X_aimd, axis=0)

    feat_lines = []
    for i, name in enumerate(feature_names):
        z = (mlff_means[i] - aimd_means[i]) / (aimd_stds[i] + 1e-10)
        feat_lines.append(
            f"  {name}: AIMD={aimd_means[i]:.4f}±{aimd_stds[i]:.4f}, "
            f"MLFF={mlff_means[i]:.4f}, z={z:+.2f}"
        )

    files = meta_aimd['file'].unique().tolist()
    temps = meta_aimd['temperature_K'].dropna().unique().tolist()

    ctx = f"""=== AIMD ANOMALY DETECTION — DATA CONTEXT ===

DATASET OVERVIEW
  AIMD windows (normal training): {n_aimd}
  MLFF windows (test/anomalous):  {n_mlff}
  Features per window: {len(feature_names)}
  Window size: 50 frames, stride: 10 frames

ANOMALY RATES
  AIMD  L1+L2 ensemble: {aimd_rate:.1%}
  MLFF  L1+L2 ensemble: {mlff_rate:.1%}
  Detection ratio: {mlff_rate/max(aimd_rate,1e-4):.1f}×
  AIMD  L1  (3-sigma):  {np.mean(results_aimd['l1_flag']):.1%}
  AIMD  L2  IF:         {np.mean(results_aimd['l2_if_flag']):.1%}
  AIMD  L2  SVM:        {np.mean(results_aimd['l2_svm_flag']):.1%}
  MLFF  L1  (3-sigma):  {np.mean(results_mlff['l1_flag']):.1%}
  MLFF  L2  IF:         {np.mean(results_mlff['l2_if_flag']):.1%}
  MLFF  L2  SVM:        {np.mean(results_mlff['l2_svm_flag']):.1%}

PER-FEATURE STATISTICS (AIMD mean ± std  |  MLFF mean  |  z-score)
{chr(10).join(feat_lines)}

TOP 5 MOST DEVIATING FEATURES (by |z-score|):
{feat_comparison.head(5).to_string(index=False)}

TRAJECTORIES
  Files: {', '.join(files[:6])}{'...' if len(files)>6 else ''}
  Temperatures: {sorted([t for t in temps if t])}K

PYTHON VARIABLES AVAILABLE IN EXECUTION NAMESPACE
  X_aimd        — np.ndarray shape ({n_aimd}, {len(feature_names)})
  X_mlff        — np.ndarray shape ({n_mlff}, {len(feature_names)})
  feature_names — list of {len(feature_names)} str
  meta_aimd     — pd.DataFrame cols: file, start, end, n_atoms, temperature_K, configuration
  meta_mlff     — pd.DataFrame cols: file, start, end, n_atoms, temperature_K, configuration
  results_aimd  — dict: anomaly_label, confidence, l1_flag, l2_if_flag, l2_svm_flag,
                        l1_score, l2_if_score, l2_svm_score  (each np.ndarray len {n_aimd})
  results_mlff  — same structure, len {n_mlff}
  feat_comparison — pd.DataFrame cols: feature, aimd_mean, mlff_mean, aimd_std,
                                        z_score, relative_change_%
  np, pd, plt, sns — already imported
"""
    return ctx


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_CODE_SYSTEM = """\
You are a scientific data analysis assistant for AIMD molecular dynamics anomaly detection.
Your job: generate executable Python/matplotlib code to answer the user's question.

STRICT RULES:
1. Use ONLY the variables listed in the DATA CONTEXT (X_aimd, X_mlff, feature_names,
   meta_aimd, meta_mlff, results_aimd, results_mlff, feat_comparison, np, pd, plt, sns).
2. Always create exactly one figure. Use fig, ax = plt.subplots(...) or
   fig, axes = plt.subplots(rows, cols, ...).
3. Add a descriptive title, axis labels, and a legend where applicable.
4. For every visual claim (e.g. a bar taller than another), add a comment:
   # EVIDENCE: <computed value> confirms <claim>
5. Do NOT import anything extra — all necessary modules are pre-imported.
6. Do NOT use plt.show(). The caller captures the figure.
7. Output ONLY a single fenced Python code block. No prose before or after.
8. If the question cannot be answered with the available data, output:
   ```python
   # INSUFFICIENT_DATA: <reason>
   fig, ax = plt.subplots()
   ax.text(0.5, 0.5, "Insufficient data:\\n<reason>", ha='center', va='center',
           transform=ax.transAxes, fontsize=12)
   ax.set_title("Cannot answer: <reason>")
   ```
"""

_MECHANISM_SYSTEM = """\
You are a scientific analyst explaining anomalies in AIMD vs MLFF molecular dynamics trajectories.
The system studies 2D Sb2Te3 with Cr dopants (topological insulator, spintronics research).

YOUR TASK: Given the data summary and the figure description, explain the likely physical
mechanism behind the observed anomaly pattern.

STRICT ANTI-HALLUCINATION RULES:
1. Every factual claim MUST cite a specific number from the DATA CONTEXT below.
   Format each claim as: "► Claim: [statement] | Evidence: [exact value from data]"
2. If you cannot find supporting evidence in the data for a claim, write:
   "► Uncertain: [statement] | Evidence: insufficient data to confirm"
3. Do NOT invent numbers, physical constants, or mechanisms not derivable from the data.
4. Structure your response as:
   ## Observed Pattern
   ## Physical Mechanism (with evidence)
   ## Key Features Driving Detection
   ## Limitations & Caveats
5. Keep the response concise and specific — under 400 words.
"""


# ---------------------------------------------------------------------------
# OllamaAnalyst
# ---------------------------------------------------------------------------

class OllamaAnalyst:
    """
    LLM-powered analyst using Ollama (glm-5:cloud).
    All outputs are grounded in actual computed data passed via context.
    """

    def __init__(self, model: str = 'glm-5:cloud', max_retries: int = 2):
        self.model = model
        self.max_retries = max_retries

    # ------------------------------------------------------------------ #
    # Figure generation
    # ------------------------------------------------------------------ #

    def generate_and_execute(
        self,
        query: str,
        data_context: str,
        namespace: Dict[str, Any],
    ) -> Tuple[Optional[plt.Figure], str, str]:
        """
        Generate Python code for `query`, execute against `namespace`.

        Returns:
            (fig, code, error_or_empty)
            fig   — matplotlib Figure if successful, else None
            code  — generated code string
            error — error message if execution failed, else ''
        """
        code = self._generate_code(query, data_context)
        fig, error = self._execute_code(code, namespace)

        # Retry once with error feedback
        if error and self.max_retries > 0:
            code = self._generate_code(
                query, data_context,
                previous_error=f"Previous attempt failed:\n{error}\nFix the code."
            )
            fig, error = self._execute_code(code, namespace)

        return fig, code, error

    def _generate_code(
        self,
        query: str,
        data_context: str,
        previous_error: str = '',
    ) -> str:
        user_content = (
            f"DATA CONTEXT:\n{data_context}\n\n"
            f"USER QUERY: {query}"
        )
        if previous_error:
            user_content += f"\n\n{previous_error}"

        resp = ollama.chat(
            model=self.model,
            messages=[
                {'role': 'system', 'content': _CODE_SYSTEM},
                {'role': 'user',   'content': user_content},
            ],
        )
        raw = resp['message']['content']
        return self._extract_code(raw)

    @staticmethod
    def _extract_code(text: str) -> str:
        """Pull the first ```python ... ``` block from LLM output."""
        match = re.search(r'```python\s*\n(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: strip backtick fences if present
        return re.sub(r'```\w*', '', text).strip()

    @staticmethod
    def _execute_code(
        code: str,
        namespace: Dict[str, Any],
    ) -> Tuple[Optional[plt.Figure], str]:
        """
        Execute code in a copy of namespace.
        Returns the first matplotlib Figure created, or None + error string.
        """
        plt.close('all')
        local_ns = dict(namespace)  # shallow copy so we don't pollute caller

        try:
            exec(compile(code, '<llm_generated>', 'exec'), local_ns)  # noqa: S102
        except Exception:
            return None, traceback.format_exc()

        # Collect any figure created during execution
        fig_nums = plt.get_fignums()
        if fig_nums:
            fig = plt.figure(fig_nums[-1])
            return fig, ''

        return None, 'Code ran without error but produced no matplotlib figure.'

    # ------------------------------------------------------------------ #
    # Mechanism analysis
    # ------------------------------------------------------------------ #

    def mechanism_analysis(
        self,
        query: str,
        data_context: str,
        figure_description: str = '',
    ) -> str:
        """
        Generate evidence-grounded mechanism analysis.
        Every claim must reference a number from data_context.
        """
        user_content = (
            f"DATA CONTEXT (all numbers are computed from real data):\n{data_context}\n\n"
        )
        if figure_description:
            user_content += f"FIGURE SHOWN TO USER:\n{figure_description}\n\n"
        user_content += f"USER QUESTION: {query}"

        resp = ollama.chat(
            model=self.model,
            messages=[
                {'role': 'system', 'content': _MECHANISM_SYSTEM},
                {'role': 'user',   'content': user_content},
            ],
        )
        return resp['message']['content']

    # ------------------------------------------------------------------ #
    # Window-level analysis
    # ------------------------------------------------------------------ #

    def analyze_window_region(
        self,
        start_win: int,
        end_win: int,
        source: str,   # 'AIMD' or 'MLFF'
        X: np.ndarray,
        feature_names: List[str],
        results: Dict,
        data_context: str,
    ) -> str:
        """Explain what is unusual about a specific window range."""
        X_region   = X[start_win:end_win]
        conf_region = results['confidence'][start_win:end_win]
        anom_region = results['anomaly_label'][start_win:end_win]

        means = np.nanmean(X_region, axis=0)
        # Which features are most anomalous (outside 2-sigma of their training dist)?
        aimd_means_global = np.array([
            float(re.search(r'AIMD=([\d.eE+\-]+)', line).group(1))
            for line in data_context.split('\n')
            if 'AIMD=' in line and '±' in line
        ] or [0.0] * len(feature_names))

        aimd_stds_global = np.array([
            float(re.search(r'±([\d.eE+\-]+)', line).group(1))
            for line in data_context.split('\n')
            if 'AIMD=' in line and '±' in line
        ] or [1.0] * len(feature_names))

        n = min(len(aimd_means_global), len(feature_names), len(means))
        z_scores = (means[:n] - aimd_means_global[:n]) / (aimd_stds_global[:n] + 1e-10)
        top_feat_idx = np.argsort(np.abs(z_scores))[::-1][:5]
        top_feats = [
            f"  {feature_names[i]}: region_mean={means[i]:.4f}, z={z_scores[i]:+.2f}"
            for i in top_feat_idx if i < len(feature_names)
        ]

        region_summary = (
            f"SELECTED REGION: {source} windows {start_win}–{end_win} "
            f"({end_win - start_win} windows)\n"
            f"  Anomaly rate in region: {np.mean(anom_region):.1%}\n"
            f"  Mean confidence score:  {np.mean(conf_region):.2f}/3\n"
            f"  Max confidence:         {np.max(conf_region)}/3\n"
            f"  Top deviating features in this region:\n"
            + '\n'.join(top_feats)
        )

        user_content = (
            f"GLOBAL DATA CONTEXT:\n{data_context}\n\n"
            f"{region_summary}\n\n"
            "Explain what physical or methodological reason could cause this specific "
            "window region to show these anomaly patterns. "
            "Cite the exact values above to support every claim."
        )

        resp = ollama.chat(
            model=self.model,
            messages=[
                {'role': 'system', 'content': _MECHANISM_SYSTEM},
                {'role': 'user',   'content': user_content},
            ],
        )
        return resp['message']['content']
