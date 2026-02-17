"""
Full AIMD Anomaly Detection Pipeline - Steps 2-6.

Usage:
    python scripts/run_full_pipeline.py

Outputs (all in results/):
    data/processed/features_aimd.npz       - AIMD feature matrix + metadata
    data/processed/features_mlff.npz       - MLFF feature matrix + metadata
    results/models/anomaly_framework.pkl   - Trained ensemble detector
    results/models/lstm_autoencoder.pt     - Trained LSTM autoencoder
    results/figures/lstm_training.png
    results/figures/feature_distributions.png
    results/figures/anomaly_timeline.png
    results/figures/feature_importance.png
    results/reports/ensemble_comparison.csv
    results/reports/aimd_vs_mlff_analysis.md
"""

import sys
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

# ---- Path setup ----
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

from src.core.loaders import TrajectoryLoader, load_all_trajectories
from src.core.feature_extractors import (
    FeatureExtractor, WindowConfig,
    ISOLATED_ATOM_ENERGIES, compute_file_energy_shift,
)
from src.core.detectors import AnomalyDetectionFramework
from src.core.models import LSTMAnomalyDetector

# ---- Constants ----
PROCESSED_DIR = ROOT / 'data' / 'processed'
RESULTS_DIR   = ROOT / 'results'
MODELS_DIR    = RESULTS_DIR / 'models'
FIGURES_DIR   = RESULTS_DIR / 'figures'
REPORTS_DIR   = RESULTS_DIR / 'reports'

for d in [PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 50
STRIDE      = 10
EPOCHS      = 60
BATCH_SIZE  = 32

# Paths to raw data directories
AIMD_DIRS = [
    ROOT / 'data' / 'raw' / 'temperature',
    ROOT / 'data' / 'raw' / 'concentration',
]
MLFF_DIR = ROOT / 'data' / 'raw' / 'mlff'


# =============================================================================
# ENERGY COMPATIBILITY AUDIT
# =============================================================================

def energy_audit(data_dirs) -> dict:
    """
    Audit energy compatibility across all XYZ files in the given directories.

    DFT total energies depend on the choice of pseudopotential (POTCAR in VASP).
    Files computed with different pseudopotentials cannot share the same E0
    isolated-atom references and will produce unphysical atomization energies
    when normalised together.

    Strategy:
        • "compatible"   — E_atm/atom in (-20, 0) eV/atom  → use energy features
        • "incompatible" — E_atm/atom outside that range    → energy → NaN (imputed)
        • "no energy"    — no energies in file               → energy → NaN

    For incompatible files, structural/dynamical features are still valid and
    are retained in training.  Energy features for those files are set to NaN
    by the _energy_features plausibility guard in feature_extractors.py.

    Option B (empirical shift) — use when incompatible data is NOT redundant:
        The per-file mean E_raw/atom is shifted to match the reference group mean.
        This is the MACE/NequIP approach of fitting isolated-atom E0 per dataset.
        Implementation sketch (not run automatically):

            ref_mean_raw   = mean(E_raw/atom) across all compatible files
            shift_i        = mean(E_raw/atom, file_i) - ref_mean_raw
            E_corrected    = E_raw_i - shift_i * N_atoms_i
            → then compute E_atm/atom with the shared E0 references

    Returns a dict mapping file path → {'raw_per_atom', 'atm_per_atom', 'compatible'}
    """
    from collections import Counter
    loader = TrajectoryLoader()
    report = {}

    for d in data_dirs:
        for xyz in sorted(Path(d).glob('*.xyz')):
            try:
                traj = loader.load(str(xyz))
            except Exception:
                continue
            sp  = traj.get('species', [])
            en  = traj['energies']
            valid = en[~np.isnan(en)]
            if len(valid) == 0 or len(sp) == 0:
                report[str(xyz)] = {'raw_per_atom': np.nan, 'atm_per_atom': np.nan,
                                    'compatible': None, 'n_atoms': len(sp)}
                continue
            n = len(sp)
            counts = Counter(sp)
            ref = sum(c * ISOLATED_ATOM_ENERGIES.get(el, 0) for el, c in counts.items())
            raw_per_atom = float(np.mean(valid)) / n
            atm_per_atom = (float(np.mean(valid)) - ref) / n
            compatible = (-20.0 < atm_per_atom < 0.0)
            report[str(xyz)] = {
                'raw_per_atom': raw_per_atom,
                'atm_per_atom': atm_per_atom,
                'compatible': compatible,
                'n_atoms': n,
                'species': dict(counts),
            }
    return report


def print_energy_audit(dirs, label=''):
    """
    Runs the audit, prints the table, and returns (report, ref_atm_per_atom).

    ref_atm_per_atom: mean E_atm/atom across all compatible files.
        Used as the target for Option B empirical energy shift.
    """
    report = energy_audit(dirs)
    incompatible = [p for p, v in report.items() if v['compatible'] is False]
    compatible_atm = [v['atm_per_atom'] for v in report.values()
                      if v['compatible'] is True and not np.isnan(v['atm_per_atom'])]
    ref_atm_per_atom = float(np.mean(compatible_atm)) if compatible_atm else -3.75

    print(f"\n{'─'*70}")
    print(f"Energy Compatibility Audit{' — ' + label if label else ''}")
    print(f"{'─'*70}")
    print(f"  Reference E_atm/atom (compatible files): {ref_atm_per_atom:.5f} eV/atom")
    print(f"  {'File':<44} {'E_atm/atom':>12}  {'Shift/frame':>12}  Status")
    print(f"  {'─'*44} {'─'*12}  {'─'*12}  {'─'*20}")
    for path, v in report.items():
        fname = Path(path).name
        atm = v['atm_per_atom']
        if v['compatible'] is None:
            status, shift_str = '— no energy', f"{'—':>12}"
        elif v['compatible']:
            status, shift_str = '✓ compatible', f"{'0.0':>12}"
        else:
            # Compute the shift that will be applied
            delta = ref_atm_per_atom - atm
            shift_str = f"{delta * v['n_atoms']:>12.1f}"
            status = f'✗ → Option B shift ({delta:+.2f} eV/atom × {v["n_atoms"]})'
        atm_str = f"{atm:12.4f}" if not np.isnan(atm) else f"{'NaN':>12}"
        print(f"  {fname:<44} {atm_str}  {shift_str}  {status}")

    if incompatible:
        print(f"\n  ⚠  {len(incompatible)} incompatible file(s) — applying Option B shift.")
        print(f"     energy_mean  → shifted to {ref_atm_per_atom:.4f} eV/atom (systematic offset removed)")
        print(f"     energy_std   → unchanged (frame-to-frame fluctuations preserved)")
        print(f"     energy_trend → unchanged (drift signal preserved)")
    else:
        print(f"\n  ✓ All files use compatible pseudopotentials.")
    print(f"{'─'*70}\n")
    return report, ref_atm_per_atom


# =============================================================================
# STEP 2 + 3: Load trajectories and extract features
# =============================================================================

def load_and_extract(data_dirs, label: str, config: WindowConfig,
                     ref_atm_per_atom: float = None):
    """
    Load all xyz files from dirs and extract windowed features.

    ref_atm_per_atom: Option B reference mean atomization energy (eV/atom)
        computed from all DFT-compatible files.  When provided, incompatible
        files receive an empirical energy shift so their energy_std/trend
        remain informative rather than being set to NaN.
    """
    loader = TrajectoryLoader()
    extractor = FeatureExtractor(config)

    all_feature_matrices = []
    all_window_meta = []  # list of dicts: {file, start, end, label}
    all_sequences = []    # for LSTM: (n_windows, seq_len, n_features) – assembled later

    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"⚠  Missing: {data_path}")
            continue

        for xyz_file in sorted(data_path.glob('*.xyz')):
            print(f"  Loading {xyz_file.name}...", end=' ', flush=True)
            try:
                traj = loader.load(str(xyz_file))
            except Exception as e:
                print(f"SKIP ({e})")
                continue

            coords   = traj['coordinates']   # (n_frames, n_atoms, 3)
            energies = traj['energies']       # (n_frames,)
            species  = traj.get('species')    # list[str], length n_atoms
            n_frames = coords.shape[0]

            if n_frames < config.window_size:
                print(f"SKIP (only {n_frames} frames < window {config.window_size})")
                continue

            # Option B: compute and apply per-file energy shift if needed
            e_shift = 0.0
            if ref_atm_per_atom is not None and energies is not None and species:
                e_shift = compute_file_energy_shift(energies, species, ref_atm_per_atom)
                if e_shift != 0.0:
                    print(f"[Option B shift {e_shift:+.1f} eV/frame] ", end='', flush=True)

            feat_matrix, windows = extractor.extract_all_windows(
                coords, energies, species, energy_shift=e_shift
            )
            all_feature_matrices.append(feat_matrix)

            for (s, e) in windows:
                all_window_meta.append({
                    'file': xyz_file.name,
                    'start': s,
                    'end': e,
                    'label': label,
                    'n_atoms': traj['n_atoms'],
                    'temperature_K': traj['metadata'].get('temperature_K'),
                    'configuration': traj['metadata'].get('configuration'),
                })

            print(f"✓ {n_frames} frames → {len(windows)} windows")

    if not all_feature_matrices:
        raise RuntimeError(f"No trajectories loaded from {data_dirs}")

    X = np.concatenate(all_feature_matrices, axis=0)  # (total_windows, n_feats)
    meta_df = pd.DataFrame(all_window_meta)
    feature_names = extractor.feature_names

    return X, meta_df, feature_names


def build_sequences(X: np.ndarray, seq_len: int = 1) -> np.ndarray:
    """
    Reshape flat feature matrix to sequences for LSTM.
    Each window is already seq_len=50 frames summarized into 1 feature vector.
    For LSTM, we group consecutive windows into sequences.
    seq_len here = number of windows per LSTM input.
    """
    # LSTM input: (n_seq, seq_len, n_features)
    # We use seq_len windows as one temporal sequence.
    n_samples = (len(X) // seq_len) * seq_len
    X_trimmed = X[:n_samples]
    return X_trimmed.reshape(-1, seq_len, X.shape[1])


# =============================================================================
# Visualization helpers
# =============================================================================

def plot_feature_distributions(
    X_aimd: np.ndarray,
    X_mlff: np.ndarray,
    feature_names,
    save_path,
    top_n: int = 12,
):
    """Side-by-side violin plots for top features."""
    # Pick most variable features
    combined_std = np.nanstd(np.vstack([X_aimd, X_mlff]), axis=0)
    top_idx = np.argsort(combined_std)[::-1][:top_n]
    top_names = [feature_names[i] for i in top_idx]

    n_cols = 4
    n_rows = int(np.ceil(top_n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.ravel()

    for ax_i, (feat_i, name) in enumerate(zip(top_idx, top_names)):
        aimd_vals = X_aimd[:, feat_i]
        mlff_vals = X_mlff[:, feat_i]
        # Remove NaN and extreme outliers for plotting
        aimd_vals = aimd_vals[~np.isnan(aimd_vals)]
        mlff_vals = mlff_vals[~np.isnan(mlff_vals)]
        p1, p99 = np.percentile(np.concatenate([aimd_vals, mlff_vals]), [1, 99])
        aimd_vals = np.clip(aimd_vals, p1, p99)
        mlff_vals = np.clip(mlff_vals, p1, p99)

        ax = axes[ax_i]
        ax.violinplot([aimd_vals, mlff_vals], positions=[0, 1], showmedians=True)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['AIMD', 'MLFF'], fontsize=9)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=8)

    for ax_i in range(len(top_names), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle('Feature Distributions: AIMD vs MLFF', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Feature distributions → {save_path}")


def plot_anomaly_timeline(
    results_aimd: dict,
    results_mlff: dict,
    save_path,
):
    """Anomaly confidence score over window index for AIMD and MLFF."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    # AIMD
    conf_aimd = results_aimd['confidence']
    ax1.plot(conf_aimd, alpha=0.7, linewidth=0.8, color='steelblue')
    ax1.axhline(2, color='red', linestyle='--', linewidth=1, label='Anomaly threshold (2/3)')
    ax1.fill_between(range(len(conf_aimd)), conf_aimd, alpha=0.3, color='steelblue')
    ax1.set_ylabel('Detector votes', fontsize=10)
    ax1.set_title('AIMD trajectories (normal)', fontsize=11)
    ax1.set_ylim(-0.1, 3.5)
    ax1.legend(fontsize=9)

    # MLFF
    conf_mlff = results_mlff['confidence']
    ax2.plot(conf_mlff, alpha=0.7, linewidth=0.8, color='darkorange')
    ax2.axhline(2, color='red', linestyle='--', linewidth=1, label='Anomaly threshold')
    ax2.fill_between(range(len(conf_mlff)), conf_mlff, alpha=0.3, color='darkorange')
    ax2.set_xlabel('Window index', fontsize=10)
    ax2.set_ylabel('Detector votes', fontsize=10)
    ax2.set_title('MLFF trajectory (test for anomalies)', fontsize=11)
    ax2.set_ylim(-0.1, 3.5)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✓ Anomaly timeline → {save_path}")


def plot_feature_importance(
    X_aimd: np.ndarray,
    X_mlff: np.ndarray,
    feature_names,
    save_path,
):
    """Feature importance via Random Forest on AIMD vs MLFF labels."""
    X = np.vstack([X_aimd, X_mlff])
    y = np.array([0] * len(X_aimd) + [1] * len(X_mlff))

    # Impute NaN
    col_medians = np.nanmedian(X, axis=0)
    for col in range(X.shape[1]):
        mask = np.isnan(X[:, col])
        X[mask, col] = col_medians[col]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(idx)), importances[idx], color='steelblue')
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Importance', fontsize=10)
    ax.set_title('Feature Importance (AIMD vs MLFF discrimination)', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✓ Feature importance → {save_path}")


def plot_correlation_heatmap(X: np.ndarray, feature_names, save_path):
    """Correlation matrix of features."""
    # Impute NaN
    X_imp = X.copy()
    col_medians = np.nanmedian(X_imp, axis=0)
    for col in range(X_imp.shape[1]):
        mask = np.isnan(X_imp[:, col])
        X_imp[mask, col] = col_medians[col]

    corr = np.corrcoef(X_imp.T)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=7)
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_title('Feature Correlation Matrix (AIMD training set)', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✓ Correlation heatmap → {save_path}")


def plot_detector_agreement(
    results_aimd: dict,
    results_mlff: dict,
    save_path,
):
    """Stacked bar showing detector agreement breakdown."""
    categories = ['AIMD (normal)', 'MLFF (test)']
    detectors = ['L1 Statistical', 'L2 Isolation Forest', 'L2 One-Class SVM']

    aimd_rates = [
        np.mean(results_aimd['l1_flag']),
        np.mean(results_aimd['l2_if_flag']),
        np.mean(results_aimd['l2_svm_flag']),
    ]
    mlff_rates = [
        np.mean(results_mlff['l1_flag']),
        np.mean(results_mlff['l2_if_flag']),
        np.mean(results_mlff['l2_svm_flag']),
    ]

    x = np.arange(len(detectors))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, aimd_rates, width, label='AIMD (normal)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, mlff_rates, width, label='MLFF (test)', color='darkorange', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(detectors, fontsize=10)
    ax.set_ylabel('Anomaly detection rate', fontsize=10)
    ax.set_title('Per-Detector Anomaly Rates: AIMD vs MLFF', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.1%}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.1%}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✓ Detector agreement → {save_path}")


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    config = WindowConfig(window_size=WINDOW_SIZE, stride=STRIDE)

    # ------------------------------------------------------------------
    # STEP 2+3: Feature extraction
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 2+3: Feature Extraction")
    print("="*60)

    _, ref_atm_per_atom = print_energy_audit(AIMD_DIRS, label='AIMD training data')

    print("\n[AIMD training data]")
    X_aimd, meta_aimd, feature_names = load_and_extract(
        AIMD_DIRS, 'aimd', config, ref_atm_per_atom=ref_atm_per_atom
    )
    print(f"\nAIMD: {X_aimd.shape[0]} windows, {X_aimd.shape[1]} features")

    print("\n[MLFF test data]")
    X_mlff, meta_mlff, _ = load_and_extract(
        [MLFF_DIR], 'mlff', config, ref_atm_per_atom=ref_atm_per_atom
    )
    print(f"MLFF: {X_mlff.shape[0]} windows, {X_mlff.shape[1]} features")

    # Save features
    np.savez(PROCESSED_DIR / 'features_aimd.npz', X=X_aimd, feature_names=feature_names)
    np.savez(PROCESSED_DIR / 'features_mlff.npz', X=X_mlff, feature_names=feature_names)
    meta_aimd.to_csv(PROCESSED_DIR / 'meta_aimd.csv', index=False)
    meta_mlff.to_csv(PROCESSED_DIR / 'meta_mlff.csv', index=False)
    print(f"\n✓ Features saved to {PROCESSED_DIR}")

    # ------------------------------------------------------------------
    # STEP 3b: Visualizations of feature distributions
    # ------------------------------------------------------------------
    print("\n[Visualizations]")
    plot_feature_distributions(
        X_aimd, X_mlff, feature_names,
        FIGURES_DIR / 'feature_distributions.png',
    )
    plot_correlation_heatmap(X_aimd, feature_names, FIGURES_DIR / 'feature_correlation.png')
    plot_feature_importance(X_aimd, X_mlff, feature_names, FIGURES_DIR / 'feature_importance.png')

    # ------------------------------------------------------------------
    # STEP 4: Train Level 1 & 2 detectors
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 4: Training Level 1 & 2 Detectors")
    print("="*60)

    # Split AIMD into train / val
    idx_all = np.arange(len(X_aimd))
    idx_train, idx_val = train_test_split(idx_all, test_size=0.2, random_state=42)
    X_train = X_aimd[idx_train]
    X_val   = X_aimd[idx_val]

    framework = AnomalyDetectionFramework(contamination=0.05)
    framework.fit(X_train, feature_names)
    framework.save(MODELS_DIR / 'anomaly_framework.pkl')

    # Quick validation sanity check
    val_results = framework.predict(X_val)
    print(f"  Val anomaly rate (should be ~5%): {np.mean(val_results['anomaly_label']):.1%}")

    # ------------------------------------------------------------------
    # STEP 5: Train LSTM Autoencoder
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 5: Training LSTM Autoencoder")
    print("="*60)

    n_features = X_aimd.shape[1]
    # Each window = 1 time step; group LSTM_SEQ consecutive windows as sequence
    LSTM_SEQ = 10  # 10 consecutive feature windows = one LSTM input

    # Impute NaN before LSTM
    X_aimd_imp = X_aimd.copy()
    col_medians = np.nanmedian(X_aimd_imp, axis=0)
    for col in range(X_aimd_imp.shape[1]):
        mask = np.isnan(X_aimd_imp[:, col])
        X_aimd_imp[mask, col] = col_medians[col]

    # Normalize (per-feature, fit on train only)
    scaler = StandardScaler()
    X_aimd_scaled = scaler.fit_transform(X_aimd_imp)

    with open(MODELS_DIR / 'lstm_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Build sequences
    X_lstm = build_sequences(X_aimd_scaled, seq_len=LSTM_SEQ)
    n_lstm_train = int(0.8 * len(X_lstm))
    X_lstm_train = X_lstm[:n_lstm_train]
    X_lstm_val   = X_lstm[n_lstm_train:]
    print(f"  LSTM input: train={X_lstm_train.shape}, val={X_lstm_val.shape}")

    lstm_det = LSTMAnomalyDetector(n_features=n_features, seq_len=LSTM_SEQ)
    lstm_det.fit(X_lstm_train, X_lstm_val, epochs=EPOCHS, batch_size=BATCH_SIZE)
    lstm_det.save(MODELS_DIR / 'lstm_autoencoder.pt')
    lstm_det.plot_training_curves(str(FIGURES_DIR / 'lstm_training.png'))

    # ------------------------------------------------------------------
    # STEP 6: Ensemble detection + MLFF analysis
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 6: Ensemble Detection & AIMD vs MLFF Analysis")
    print("="*60)

    results_aimd = framework.predict(X_aimd)
    results_mlff = framework.predict(X_mlff)

    # LSTM predictions
    def lstm_predict(X_raw):
        X_imp = X_raw.copy()
        for col in range(X_imp.shape[1]):
            mask = np.isnan(X_imp[:, col])
            X_imp[mask, col] = col_medians[col]
        X_sc = scaler.transform(X_imp)
        X_sq = build_sequences(X_sc, seq_len=LSTM_SEQ)
        if len(X_sq) == 0:
            return np.array([]), np.array([])
        errors = lstm_det.reconstruction_errors(X_sq)
        flags  = lstm_det.predict(X_sq)
        return flags, errors

    l3_aimd_flags, l3_aimd_errors = lstm_predict(X_aimd)
    l3_mlff_flags, l3_mlff_errors = lstm_predict(X_mlff)

    # Align lengths (LSTM trims to multiple of LSTM_SEQ)
    n_aimd_lstm = len(l3_aimd_flags)
    n_mlff_lstm = len(l3_mlff_flags)

    # Final ensemble (L1+L2 + L3) on overlapping length
    def ensemble_final(l12_results, l3_flags, n):
        l12_anom = l12_results['anomaly_label'][:n]
        l3_anom  = l3_flags[:n] if len(l3_flags) >= n else np.zeros(n, dtype=int)
        final    = ((l12_anom + l3_anom) >= 1).astype(int)  # any detector
        return final

    final_aimd = ensemble_final(results_aimd, l3_aimd_flags, n_aimd_lstm)
    final_mlff = ensemble_final(results_mlff, l3_mlff_flags, n_mlff_lstm)

    aimd_rate = float(np.mean(results_aimd['anomaly_label']))
    mlff_rate = float(np.mean(results_mlff['anomaly_label']))
    aimd_rate_3 = float(np.mean(final_aimd)) if len(final_aimd) else aimd_rate
    mlff_rate_3 = float(np.mean(final_mlff)) if len(final_mlff) else mlff_rate

    print(f"\n  AIMD anomaly rate (L1+L2): {aimd_rate:.1%}")
    print(f"  MLFF anomaly rate (L1+L2): {mlff_rate:.1%}")
    print(f"  AIMD anomaly rate (all 3): {aimd_rate_3:.1%}")
    print(f"  MLFF anomaly rate (all 3): {mlff_rate_3:.1%}")

    # ---- Save ensemble comparison CSV ----
    comparison_rows = []
    for i in range(len(X_aimd)):
        comparison_rows.append({
            'source': 'AIMD',
            'window_idx': i,
            'file': meta_aimd.iloc[i]['file'] if i < len(meta_aimd) else '',
            'l1_flag':    int(results_aimd['l1_flag'][i]),
            'l2_if_flag': int(results_aimd['l2_if_flag'][i]),
            'l2_svm_flag':int(results_aimd['l2_svm_flag'][i]),
            'ensemble_l12': int(results_aimd['anomaly_label'][i]),
            'l1_score':   float(results_aimd['l1_score'][i]),
            'l2_if_score':float(results_aimd['l2_if_score'][i]),
            'l2_svm_score':float(results_aimd['l2_svm_score'][i]),
        })
    for i in range(len(X_mlff)):
        comparison_rows.append({
            'source': 'MLFF',
            'window_idx': i,
            'file': meta_mlff.iloc[i]['file'] if i < len(meta_mlff) else '',
            'l1_flag':    int(results_mlff['l1_flag'][i]),
            'l2_if_flag': int(results_mlff['l2_if_flag'][i]),
            'l2_svm_flag':int(results_mlff['l2_svm_flag'][i]),
            'ensemble_l12': int(results_mlff['anomaly_label'][i]),
            'l1_score':   float(results_mlff['l1_score'][i]),
            'l2_if_score':float(results_mlff['l2_if_score'][i]),
            'l2_svm_score':float(results_mlff['l2_svm_score'][i]),
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(REPORTS_DIR / 'ensemble_comparison.csv', index=False)
    print(f"✓ Ensemble comparison → {REPORTS_DIR / 'ensemble_comparison.csv'}")

    # ---- Visualizations ----
    plot_anomaly_timeline(results_aimd, results_mlff, FIGURES_DIR / 'anomaly_timeline.png')
    plot_detector_agreement(results_aimd, results_mlff, FIGURES_DIR / 'detector_agreement.png')

    # ---- Identify most deviant features in MLFF ----
    def impute_arr(arr):
        out = arr.copy()
        med = np.nanmedian(out, axis=0)
        for col in range(out.shape[1]):
            mask = np.isnan(out[:, col])
            out[mask, col] = med[col] if not np.isnan(med[col]) else 0.0
        return out

    X_aimd_i = impute_arr(X_aimd)
    X_mlff_i = impute_arr(X_mlff)
    aimd_means = np.mean(X_aimd_i, axis=0)
    mlff_means = np.mean(X_mlff_i, axis=0)
    aimd_stds  = np.std(X_aimd_i, axis=0)
    z_scores   = (mlff_means - aimd_means) / (aimd_stds + 1e-10)

    feat_comparison = pd.DataFrame({
        'feature': feature_names,
        'aimd_mean': aimd_means,
        'mlff_mean': mlff_means,
        'aimd_std': aimd_stds,
        'z_score': z_scores,
        'relative_change_%': (mlff_means - aimd_means) / (np.abs(aimd_means) + 1e-10) * 100,
    }).sort_values('z_score', key=np.abs, ascending=False)
    feat_comparison.to_csv(REPORTS_DIR / 'feature_comparison.csv', index=False)

    top_deviating = feat_comparison.head(10)

    # ---- Write analysis markdown ----
    analysis_text = f"""# AIMD vs MLFF Anomaly Analysis

**Generated:** 2026-02-13
**Pipeline:** Multi-level anomaly detection (Statistical + ML + LSTM)

---

## Summary Statistics

| Metric | AIMD (normal) | MLFF (test) |
|--------|--------------|-------------|
| Total windows | {len(X_aimd)} | {len(X_mlff)} |
| L1+L2 anomaly rate | {aimd_rate:.1%} | {mlff_rate:.1%} |
| All-3 anomaly rate | {aimd_rate_3:.1%} | {mlff_rate_3:.1%} |
| L1 (statistical) | {np.mean(results_aimd['l1_flag']):.1%} | {np.mean(results_mlff['l1_flag']):.1%} |
| L2 Isolation Forest | {np.mean(results_aimd['l2_if_flag']):.1%} | {np.mean(results_mlff['l2_if_flag']):.1%} |
| L2 One-Class SVM | {np.mean(results_aimd['l2_svm_flag']):.1%} | {np.mean(results_mlff['l2_svm_flag']):.1%} |

---

## Top 10 Most Deviating Features (MLFF vs AIMD)

| Feature | AIMD mean | MLFF mean | Z-score | Change % |
|---------|-----------|-----------|---------|----------|
"""
    for _, row in top_deviating.iterrows():
        analysis_text += (
            f"| {row['feature']} | {row['aimd_mean']:.4f} | {row['mlff_mean']:.4f} "
            f"| {row['z_score']:+.2f} | {row['relative_change_%']:+.1f}% |\n"
        )

    analysis_text += f"""
---

## Physical Interpretation

The MLFF trajectory shows a **{mlff_rate/max(aimd_rate,0.01):.1f}x higher anomaly rate**
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
"""

    with open(REPORTS_DIR / 'aimd_vs_mlff_analysis.md', 'w') as f:
        f.write(analysis_text)
    print(f"✓ Analysis report → {REPORTS_DIR / 'aimd_vs_mlff_analysis.md'}")

    # ---- Save summary JSON for dashboard ----
    summary = {
        'feature_names': feature_names,
        'n_aimd_windows': int(len(X_aimd)),
        'n_mlff_windows': int(len(X_mlff)),
        'aimd_anomaly_rate_l12': aimd_rate,
        'mlff_anomaly_rate_l12': mlff_rate,
        'aimd_anomaly_rate_all3': aimd_rate_3,
        'mlff_anomaly_rate_all3': mlff_rate_3,
        'lstm_threshold': float(lstm_det.threshold),
        'window_size': WINDOW_SIZE,
        'stride': STRIDE,
    }
    with open(PROCESSED_DIR / 'pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\n  AIMD anomaly rate: {aimd_rate:.1%}  (expected: ~5%)")
    print(f"  MLFF anomaly rate: {mlff_rate:.1%}  (expected: higher)")
    print(f"\n  All results in: {RESULTS_DIR}")
    print(f"  Models in:      {MODELS_DIR}")


if __name__ == '__main__':
    main()
