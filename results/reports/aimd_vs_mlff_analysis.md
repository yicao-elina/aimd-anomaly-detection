# AIMD vs MLFF Anomaly Analysis

**Generated:** 2026-02-13
**Pipeline:** Multi-level anomaly detection (Statistical + ML + LSTM)

---

## Summary Statistics

| Metric | AIMD (normal) | MLFF (test) |
|--------|--------------|-------------|
| Total windows | 1648 | 835 |
| L1+L2 anomaly rate | 3.9% | 100.0% |
| All-3 anomaly rate | 35.4% | 100.0% |
| L1 (statistical) | 6.5% | 100.0% |
| L2 Isolation Forest | 5.2% | 100.0% |
| L2 One-Class SVM | 4.7% | 100.0% |

---

## Top 10 Most Deviating Features (MLFF vs AIMD)

| Feature | AIMD mean | MLFF mean | Z-score | Change % |
|---------|-----------|-----------|---------|----------|
| frame_variance | 0.0000 | 1.3146 | +2517692.00 | +520618752.0% |
| disp_p95 | 0.0055 | 39.8208 | +8108.34 | +721560.0% |
| disp_median | 0.0029 | 16.3176 | +6902.15 | +562595.1% |
| msd_mean | 0.0133 | 449.9925 | +6397.46 | +3394179.8% |
| msd_std | 0.0104 | 271.7797 | +6013.30 | +2621664.8% |
| disp_mean | 0.0032 | 18.2810 | +4799.67 | +576124.7% |
| msd_final | 0.0326 | 585.2461 | +4639.82 | +1796056.6% |
| rms_velocity | 0.0041 | 21.5418 | +1769.60 | +520993.4% |
| disp_std | 0.0022 | 11.2431 | +964.85 | +506701.0% |
| energy_std | 0.0029 | 1.8069 | +479.58 | +62688.2% |

---

## Physical Interpretation

The MLFF trajectory shows a **25.4x higher anomaly rate**
than AIMD at the L1+L2 level. The top deviating features suggest:

- **Dynamics features** (RMS velocity, crest factor) deviating → MLFF force field may
  overestimate atomic velocities or produce unrealistic acceleration events.
- **Spectral features** (dominant frequency, spectral entropy) deviating → different
  vibrational mode structure in MLFF vs DFT.
- **MSD features** deviating → different diffusion behavior in MLFF vs AIMD.

This quantifies the physical regime where the machine-learned force field diverges from
first-principles DFT dynamics.

---

## Files Generated

- `results/figures/feature_distributions.png` — Violin plots AIMD vs MLFF
- `results/figures/anomaly_timeline.png` — Detection confidence over time
- `results/figures/detector_agreement.png` — Per-detector anomaly rates
- `results/figures/feature_importance.png` — RF-based feature importance
- `results/figures/feature_correlation.png` — Feature correlation heatmap
- `results/figures/lstm_training.png` — LSTM training curves
- `results/reports/ensemble_comparison.csv` — Full per-window results
- `results/reports/feature_comparison.csv` — AIMD vs MLFF feature statistics
