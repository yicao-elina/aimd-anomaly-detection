# AIMD vs MLFF Anomaly Analysis

**Generated:** 2026-02-13
**Pipeline:** Multi-level anomaly detection (Statistical + ML + LSTM)

---

## Summary Statistics

| Metric | AIMD (normal) | MLFF (test) |
|--------|--------------|-------------|
| Total windows | 1648 | 799 |
| L1+L2 anomaly rate | 3.6% | 100.0% |
| All-3 anomaly rate | 31.1% | 100.0% |
| L1 (statistical) | 6.2% | 100.0% |
| L2 Isolation Forest | 5.1% | 100.0% |
| L2 One-Class SVM | 4.7% | 100.0% |

---

## Top 10 Most Deviating Features (MLFF vs AIMD)

| Feature | AIMD mean | MLFF mean | Z-score | Change % |
|---------|-----------|-----------|---------|----------|
| frame_variance | 0.0000 | 1.3738 | +2631129.75 | +544075904.0% |
| disp_p95 | 0.0055 | 41.6138 | +8473.49 | +754054.6% |
| disp_median | 0.0029 | 17.0522 | +7212.93 | +587926.5% |
| msd_mean | 0.0133 | 470.2637 | +6685.67 | +3547084.8% |
| msd_std | 0.0104 | 284.0227 | +6284.19 | +2739768.2% |
| disp_mean | 0.0032 | 19.1040 | +5015.79 | +602066.8% |
| msd_final | 0.0326 | 611.6080 | +4848.83 | +1876962.9% |
| rms_velocity | 0.0041 | 22.5117 | +1849.29 | +544454.9% |
| disp_std | 0.0022 | 11.7494 | +1008.31 | +529523.0% |
| energy_std | 0.2182 | 291.9116 | +914.28 | +133694.7% |

---

## Physical Interpretation

The MLFF trajectory shows a **27.9x higher anomaly rate**
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
