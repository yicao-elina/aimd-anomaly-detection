# Steps 2–8 Completion Summary

## Status: COMPLETE

**Date:** 2026-02-13
**Environment:** conda `agentic` (Python 3.11)

---

## What Was Built

### Step 2+3: Feature Extraction (`src/core/feature_extractors.py`)

- **`FeatureExtractor`** class with sliding window support
- **22 features** extracted per window (50 frames, stride 10):
  - Displacement statistics (7): mean, std, skew, kurtosis, max, median, p95
  - Dynamics (5): rms_velocity, crest_factor, impulse_factor, frame_variance, anisotropy
  - Frequency domain (3): dominant_freq, spectral_entropy, spectral_peak_ratio
  - MSD (4): msd_mean, msd_std, msd_final, msd_slope
  - Energy (3): energy_mean, energy_std, energy_trend
- All features are atom-count-agnostic (aggregated over atoms) → handles 80–86 atom files

**Results:**
- AIMD: **1,648 windows** from 12 trajectories
- MLFF: **799 windows** from 1 trajectory

### Step 4: Level 1 & 2 Detectors (`src/core/detectors.py`)

- **`StatisticalDetector`**: Per-feature 3σ thresholds on AIMD training set
- **`AnomalyDetectionFramework`**: Ensemble of Level 1 + Isolation Forest + One-Class SVM
- Trained on 70% of AIMD windows (1,318 samples)
- Validation anomaly rate: **2.7%** (sanity check: expected ~5%)

**Saved:** `results/models/anomaly_framework.pkl`

### Step 5: LSTM Autoencoder (`src/core/models.py`)

- Architecture: LSTM(64)→LSTM(32) Encoder | LSTM(32)→LSTM(64)→Dense Decoder
- Input: (batch, seq_len=10, n_features=22)
- Training: 131 sequences, early stopping at epoch 46
- Val loss converged: 0.137 → Anomaly threshold: **0.202**

**Saved:** `results/models/lstm_autoencoder.pt`

### Step 6: Ensemble Analysis (`scripts/run_full_pipeline.py`)

| Metric | AIMD | MLFF |
|--------|------|------|
| L1+L2 anomaly rate | **3.6%** | **100.0%** |
| L1 (statistical) | ~5% | 100% |
| L2 Isolation Forest | ~5% | 100% |
| L2 One-Class SVM | ~5% | 100% |
| Detection ratio | — | **27.9×** |

The MLFF trajectory is completely outside the AIMD distribution.

**Top deviating features (MLFF vs AIMD):**
- Energy features (energy_mean, energy_std, energy_trend) — largest z-scores
- MSD features (msd_slope, msd_final) — MLFF shows different diffusion
- Velocity features (rms_velocity, crest_factor) — different force magnitudes

### Step 7: Streamlit Dashboard (`app/dashboard.py`)

4 interactive pages:
1. **Data Overview** — file inventory, temperature breakdown, frame counts
2. **Feature Analysis** — violin plots, correlation heatmap, importance, interactive explorer
3. **Anomaly Detection** — timeline, detector agreement, threshold slider, LSTM curves
4. **AIMD vs MLFF** — Z-score bar chart, full comparison table, CSV download

**Run:** `streamlit run app/dashboard.py`

### Step 8: Documentation

- `README.md` — comprehensive project documentation
- `results/reports/aimd_vs_mlff_analysis.md` — analysis findings
- `results/reports/ensemble_comparison.csv` — per-window detection results (2,447 rows)
- `results/reports/feature_comparison.csv` — 22-feature AIMD vs MLFF statistics

---

## All Deliverables Generated

| File | Status |
|------|--------|
| `data/processed/features_aimd.npz` | ✅ |
| `data/processed/features_mlff.npz` | ✅ |
| `data/processed/pipeline_summary.json` | ✅ |
| `results/models/anomaly_framework.pkl` | ✅ |
| `results/models/lstm_autoencoder.pt` | ✅ |
| `results/figures/feature_distributions.png` | ✅ |
| `results/figures/anomaly_timeline.png` | ✅ |
| `results/figures/detector_agreement.png` | ✅ |
| `results/figures/feature_importance.png` | ✅ |
| `results/figures/feature_correlation.png` | ✅ |
| `results/figures/lstm_training.png` | ✅ |
| `results/reports/ensemble_comparison.csv` | ✅ |
| `results/reports/feature_comparison.csv` | ✅ |
| `results/reports/aimd_vs_mlff_analysis.md` | ✅ |
| `app/dashboard.py` | ✅ |
| `README.md` | ✅ |

---

## One-Command Reproduction

```bash
conda activate agentic
cd /Users/alina/.../aimd
python scripts/run_full_pipeline.py    # ~2 min total
streamlit run app/dashboard.py          # launch dashboard
```
