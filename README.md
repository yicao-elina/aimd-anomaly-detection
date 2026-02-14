# AIMD Anomaly Detection Framework

A **multi-level anomaly detection system** for *ab initio* molecular dynamics (AIMD) trajectories of 2D Sb₂Te₃ with Cr dopants. Built for two purposes: (1) a scientific result for an *npj 2D Materials* paper, and (2) a portfolio project demonstrating ML engineering for real-world time-series anomaly detection.

| Metric | Value |
|--------|-------|
| AIMD training windows | 1,648 |
| MLFF test windows | 799 |
| AIMD anomaly rate (L1+L2) | **3.6%** ✅ |
| MLFF anomaly rate (L1+L2) | **100%** ✅ |
| Detection ratio | **27.9×** |
| Features per window | 22 |
| LSTM val loss | 0.137 (converged epoch 46) |

> **Key Finding:** The MLFF trajectory shows `disp_median = 17.05 Å` vs AIMD `0.003 Å` (z = +7213), indicating catastrophic atomic displacements — atoms move thousands of times further per timestep than DFT predicts.

---

## Quick Start

> **Prerequisite:** Use the full Python path — `conda run -n agentic` has a known `libarchive.19.dylib` conflict on macOS.

```bash
# 1. Run full pipeline  (~2 min, trains all models)
/Users/alina/anaconda3/envs/agentic/bin/python scripts/run_full_pipeline.py

# 2. Launch dashboard  (Ollama must be running for Page 5 / AI Analysis)
/Users/alina/anaconda3/envs/agentic/bin/streamlit run app/dashboard.py
```

For other machines:

```bash
# Install dependencies
pip install -r requirements.txt

# Run
python scripts/run_full_pipeline.py
streamlit run app/dashboard.py
```

---

## Project Structure

```
aimd/
├── app/
│   ├── dashboard.py            # Streamlit 5-page dashboard (warm cream theme)
│   ├── project_showcase.html   # Static HTML showcase (dark editorial theme)
│   └── skills_playground.html  # Frontend component reference
├── src/core/
│   ├── loaders.py              # TrajectoryLoader: parse .xyz → (n_frames, n_atoms, 3)
│   ├── feature_extractors.py   # 22 aggregate features, sliding window
│   ├── detectors.py            # L1 (3σ) + L2 (IF + SVM) ensemble, serialisable
│   ├── models.py               # LSTM Autoencoder (PyTorch)
│   └── llm_analyst.py          # OllamaAnalyst: code gen + mechanism analysis
├── scripts/
│   └── run_full_pipeline.py    # End-to-end: extract → train → evaluate → save
├── data/
│   ├── raw/
│   │   ├── temperature/        # AIMD at 300K, 600K, 1200K
│   │   ├── concentration/      # 7 Cr configurations at 600K
│   │   └── mlff/               # MLFF trajectory (test / anomalous)
│   └── processed/              # Feature matrices (.npz), metadata (.csv)
├── results/
│   ├── models/                 # anomaly_framework.pkl, lstm_autoencoder.pt
│   ├── figures/                # 6 publication-quality plots
│   └── reports/                # ensemble_comparison.csv, feature_comparison.csv
├── docs/
│   └── PROJECT_ACHIEVEMENT_SUMMARY.md
├── requirements.txt
├── CLAUDE.md                   # AI assistant context (design decisions, env info)
└── streamlit_app.py -> app/dashboard.py  (symlink for Streamlit Cloud deploy)
```

---

## Detection Architecture

```
Raw .xyz files (25,054 frames)
    ↓
TrajectoryLoader  →  FeatureExtractor (22 features, window=50, stride=10)
    ↓                        ↓                 ↓               ↓
Level 1 (3σ)    Level 2a (Isolation Forest)  Level 2b (OC-SVM)  Level 3 (LSTM)
    └──────────────────── Ensemble Vote (≥2/3 = anomaly) ──────────────────┘
                                   ↓
                       Streamlit Dashboard (5 pages)
                                   ↓
                       glm-5:cloud via Ollama  (NL queries, code gen)
```

### Level 1 — Statistical (3σ)
Per-feature bounds from AIMD training set. Zero model loading, fully interpretable.

### Level 2 — Machine Learning ensemble
- **Isolation Forest** — 100 trees, contamination=5%
- **One-Class SVM** — RBF kernel, ν=0.05
- Both trained exclusively on AIMD (normal) windows.

### Level 3 — LSTM Autoencoder
- Encoder: LSTM(64) → LSTM(32) bottleneck → Decoder: LSTM(32) → LSTM(64) → Dense
- Anomaly threshold: 95th percentile of validation reconstruction error (0.202)
- Trains in ~2 min on CPU.

### Ensemble vote
`confidence ∈ {0, 1, 2, 3}` — number of detectors flagging the window. Anomaly if ≥ 2.

---

## 22 Features (all atom-count-agnostic)

| Category | Features |
|----------|----------|
| **Displacement stats** (7) | `disp_mean`, `disp_std`, `disp_skew`, `disp_kurtosis`, `disp_max`, `disp_median`, `disp_p95` |
| **Dynamics** (5) | `rms_velocity`, `crest_factor`, `impulse_factor`, `frame_variance`, `anisotropy` |
| **Frequency domain** (3) | `dominant_freq`, `spectral_entropy`, `spectral_peak_ratio` |
| **MSD** (4) | `msd_mean`, `msd_std`, `msd_final`, `msd_slope` |
| **Energy** (3) | `energy_mean`, `energy_std`, `energy_trend` |

All features aggregate over atoms, so the same pipeline handles 80–86 atom systems without modification.

---

## Dashboard (5 Pages)

### Page 1 — Data Overview
- Metrics bar: file count, window count, temperatures, features
- AIMD trajectory inventory table with atom counts and configurations
- Bar charts: windows per file, windows by temperature

### Page 2 — Feature Analysis
Four tabs:
- **Distributions** — violin plots AIMD vs upload per feature
- **Importance & Correlation** — RF importance bar chart + 22×22 heatmap
- **Interactive Explorer** — feature selector with histogram, time-series, z-score card
- **Statistics Table** — full numerical summary

### Page 3 — Anomaly Detection *(key page)*
- **SVG Status Monitor** — three animated detector rings (L1, L2a, L2b) with live anomaly rates, colored by severity (sage = normal, gold = warning, coral = catastrophic); ensemble verdict with detection ratio
- Full confidence timeline (AIMD or upload selectable)
- **Dynamic Window Zoom** — range slider → zoomed confidence + per-detector flag heatmap (3×N) + top-8 feature deviation bar chart
- **Ask AI** expander — LLM explains the selected window region
- Threshold slider, confidence distribution, LSTM training curve

### Page 4 — AIMD vs Upload Comparison
- Z-score deviation bar chart (top features ranked)
- Side-by-side distributions for selected feature
- Full 22-feature comparison table with color gradient on z-score
- CSV download buttons

### Page 5 — AI Analysis Assistant
- **Natural language query** → `glm-5:cloud` generates matplotlib code → executes against real data → displays figure
- 3 modes: Generate Figure / Mechanism Analysis / Both
- 5 quick one-click analyses
- Full data context viewer (shows exact values sent to LLM)

---

## Upload New Trajectories

The dashboard accepts new `.xyz` trajectory files at runtime — no pipeline retraining needed.

1. Drop a `.xyz` file in the sidebar uploader
2. Predict-only pipeline runs in ~5 s (loads saved detectors, extracts features, predicts)
3. All 5 dashboard pages update with the new upload's results
4. Switch freely between multiple uploads; delete any (except the baseline MLFF)

This means you can compare any new MLFF or test trajectory against the AIMD baseline interactively.

---

## LLM Analysis (Page 5)

Uses **`glm-5:cloud`** via [Ollama](https://ollama.com) running locally.

**Anti-hallucination design:**
- Every mechanism claim must be formatted as `► Claim: ... | Evidence: [value]`
- Figures are produced by executing LLM-generated code against real numpy arrays — they cannot be fabricated
- The LLM receives only actual computed statistics; no free-form speculation is rewarded

**To enable:** start Ollama with `glm-5:cloud` pulled before launching the dashboard.
```bash
ollama pull glm-5:cloud
ollama serve
```

---

## Dependencies

```
Python ≥ 3.9
numpy scipy pandas scikit-learn
torch
streamlit
matplotlib seaborn
ollama          # Python client — only needed for Page 5
```

Install:
```bash
pip install -r requirements.txt
pip install ollama   # optional, for AI Analysis page
```

---

## Data

| Source | Files | Frames | Role |
|--------|-------|--------|------|
| AIMD — temperature variants (300K, 600K, 1200K) | 3 | 3,494 | Training (normal) |
| AIMD — Cr concentration variants (7 configs @ 600K) | 9 | 14,482 | Training (normal) |
| MLFF trajectory | 1 | 8,038 | Test (anomalous) |
| **Total** | **13** | **25,054** | |

Raw `.xyz` files are **not** included in this repository (> 1 GB). Contact the author or see the associated paper for data access.

---

## Outputs

After running `run_full_pipeline.py`:

```
data/processed/
  features_aimd.npz       1,648 × 22 feature matrix
  features_mlff.npz         799 × 22 feature matrix
  pipeline_summary.json    key metrics + config

results/models/
  anomaly_framework.pkl   trained L1+L2 ensemble
  lstm_autoencoder.pt     trained LSTM (epoch 46)
  lstm_scaler.pkl         StandardScaler for LSTM input

results/figures/
  feature_distributions.png
  feature_importance.png
  feature_correlation.png
  anomaly_timeline.png
  detector_agreement.png
  lstm_training.png

results/reports/
  ensemble_comparison.csv   per-window results (2,447 rows)
  feature_comparison.csv    22-feature AIMD vs MLFF statistics
  aimd_vs_mlff_analysis.md  narrative findings
```

---

## Known Issues

- `conda run -n agentic` fails with `libarchive.19.dylib` error on macOS. Always use the **full Python path** `/Users/alina/anaconda3/envs/agentic/bin/python` on this machine.
- Pylance import warnings in VS Code are expected — the `agentic` conda env is not the VS Code interpreter. The code runs correctly with the full path.

---

## Citation

If you use this code or the anomaly detection methodology in your work, please cite the associated npj 2D Materials paper (in preparation).

---

*Built with PyTorch · scikit-learn · Streamlit · Ollama · matplotlib*
