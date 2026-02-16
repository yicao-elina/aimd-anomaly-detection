"""
Active Learning Loop File Generator for MACE MLFF — 2D Sb2Te3 / Cr dopants.

This module exposes three string variables containing the full content of:
  - config_al.yaml       : MACE + QE + SLURM + AL-loop configuration
  - run_al_loop.py       : Python orchestration script (~150 lines)
  - submit_al.sh         : SLURM batch submission script

Usage in a Streamlit dashboard download widget:
    from scripts.al_loop_files import config_yaml, run_script, submit_sh

    st.download_button("Download config_al.yaml",  data=config_yaml, file_name="config_al.yaml")
    st.download_button("Download run_al_loop.py",  data=run_script,  file_name="run_al_loop.py")
    st.download_button("Download submit_al.sh",    data=submit_sh,   file_name="submit_al.sh")

Or access as a dict:
    files = al_loop_files()
    files["config_yaml"]
    files["run_script"]
    files["submit_sh"]
"""

# =============================================================================
# FILE 1 — config_al.yaml
# =============================================================================

config_yaml = """\
# ============================================================
# config_al.yaml — Active Learning Configuration
# MACE MLFF for 2D Sb2Te3 with Cr dopants (82-atom bilayer)
# ============================================================

system:
  name: "2D-Sb2Te3-Cr"
  description: "2-layer bilayer Sb2Te3 with Cr dopants, topological insulator"
  elements: ["Sb", "Te", "Cr"]
  n_atoms: 82
  structure: "hexagonal_bilayer"
  composition: "Sb16Te24Cr2"          # representative 2-Cr supercell

# ------------------------------------------------------------
# MACE training hyperparameters
# ------------------------------------------------------------
mace:
  r_max: 6.0                          # cutoff radius in Angstrom
  num_radial_basis: 10                # number of radial basis functions
  max_L: 2                            # maximum angular momentum (l=0,1,2)
  num_channels: 128                   # embedding / message-passing channels
  num_interactions: 2                 # number of interaction blocks
  batch_size: 16
  lr: 0.005
  max_num_epochs: 2000
  patience: 200                       # early-stopping patience (epochs)
  scheduler_gamma: 0.9995             # exponential LR decay per epoch
  energy_weight: 1.0
  forces_weight: 100.0
  stress_weight: 1.0
  seed: 42                            # base seed; committee models offset this
  device: "cuda"                      # "cuda" or "cpu"
  default_dtype: "float64"
  model_dir: "results/models/mace"
  train_file: "data/al/train.xyz"
  valid_file: "data/al/valid.xyz"
  test_file:  "data/al/test.xyz"

# ------------------------------------------------------------
# Committee / query-by-committee settings
# ------------------------------------------------------------
committee:
  n_committee: 4
  seeds: [42, 137, 271, 999]          # one unique seed per committee member
  model_prefix: "mace_committee"
  uncertainty_metric: "forces_std"    # std of force vectors across committee
  agreement_threshold: 0.95           # fraction of committee agreeing = "certain"

# ------------------------------------------------------------
# Quantum ESPRESSO DFT labeling settings
# ------------------------------------------------------------
qe:
  calculation: "scf"
  prefix: "Sb2Te3_Cr"
  pseudo_dir: "data/pseudopotentials"
  outdir: "data/al/qe_scratch"
  pseudopotentials:
    Sb: "Sb.pbe-n-kjpaw_psl.1.0.0.UPF"
    Te: "Te.pbe-n-kjpaw_psl.1.0.0.UPF"
    Cr: "Cr.pbe-spn-kjpaw_psl.1.0.0.UPF"
  ecutwfc: 60                         # kinetic energy cutoff (Ry)
  ecutrho: 480                        # charge density cutoff (Ry); 8x ecutwfc
  k_points: [2, 2, 1]                 # Monkhorst-Pack grid (2D: kz=1)
  k_shift: [0, 0, 0]
  occupations: "smearing"
  smearing: "cold"
  degauss: 0.01                       # smearing width (Ry)
  conv_thr: 1.0e-8                    # SCF convergence threshold (Ry)
  mixing_beta: 0.3
  mixing_mode: "local-TF"
  electron_maxstep: 200
  lspinorb: false
  noncolin: false
  nspin: 2                            # spin-polarised for Cr dopants
  starting_magnetization:
    Cr: 3.0
    Sb: 0.0
    Te: 0.0
  vdw_corr: "grimme-d3"              # van der Waals correction (2D layers)
  # parallelism — match SLURM ntasks
  npool: 4
  ndiag: 1

# ------------------------------------------------------------
# DP-GEN style active learning loop
# ------------------------------------------------------------
active_learning:
  sigma_lo: 0.10                      # eV/Ang; below = already accurate, skip
  sigma_hi: 0.30                      # eV/Ang; above = too uncertain, skip
  max_iterations: 10
  n_explore_frames: 500               # MD frames to generate per iteration
  temperatures: [300, 600, 900, 1200] # K; explore at each temperature
  candidate_selection:
    strategy: "uncertainty_sampling"
    max_candidates_per_iter: 50       # max DFT calls per iteration
    diversity_threshold: 0.05         # min pairwise feature distance (cosine)
    prioritise_high_T: true           # weight high-T frames more
  convergence:
    min_accurate_fraction: 0.99       # stop when >99% frames have sigma < sigma_lo
    min_iterations: 3                 # always run at least 3 iterations
  exploration:
    md_steps: 1000
    md_timestep_fs: 1.0
    md_ensemble: "nvt"
    thermostat: "langevin"
    langevin_gamma: 0.01              # ps^-1
  data_dir: "data/al"
  results_dir: "results/al"
  initial_train_xyz: "data/processed/initial_train.xyz"   # seed structures
  anomaly_csv: "results/reports/ensemble_comparison.csv"  # anomaly scores used
                                                           # for warm-start selection

# ------------------------------------------------------------
# SLURM scheduler settings
# ------------------------------------------------------------
slurm:
  partition: "gpu"
  nodes: 1
  ntasks: 8
  ntasks_per_node: 8
  gpus_per_node: 1
  gres: "gpu:1"
  time: "48:00:00"
  mem: "64G"
  job_name: "mace_al_Sb2Te3"
  mail_type: "END,FAIL"
  mail_user: ""                       # fill in your email address
  output_log: "logs/al_loop_%j.out"
  error_log:  "logs/al_loop_%j.err"
  conda_env: "mace"
  qe_module: "quantum-espresso/7.2"
  mpi_launcher: "srun"
"""

# =============================================================================
# FILE 2 — run_al_loop.py
# =============================================================================

run_script = """\
#!/usr/bin/env python3
\"\"\"
run_al_loop.py — Active Learning Orchestration for MACE MLFF
System: 2D Sb2Te3 with Cr dopants (82 atoms, bilayer)

Usage:
    python run_al_loop.py [--config config_al.yaml] [--iter N] [--dry-run]

Workflow per iteration:
  1. Load anomaly-detected candidates (ensemble_comparison.csv)
  2. Select uncertain frames: sigma_lo < force_std < sigma_hi
  3. Generate Quantum ESPRESSO SCF input files
  4. Submit QE DFT jobs and wait for completion
  5. Retrain MACE committee (4 models, different seeds)
  6. Evaluate uncertainty on exploration set
  7. Check convergence; write summary JSON
\"\"\"

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    log.info("Config loaded from %s", path)
    return cfg


# ---------------------------------------------------------------------------
# Candidate selection from anomaly detection results
# ---------------------------------------------------------------------------

def load_anomaly_candidates(csv_path: str, cfg: dict) -> pd.DataFrame:
    \"\"\"Load ensemble_comparison.csv and derive anomaly confidence scores.\"\"\"
    df = pd.read_csv(csv_path)
    # Derive a scalar confidence = number of detectors voting 'anomalous'
    vote_cols = [c for c in df.columns if c.endswith("_flag") and "ensemble" not in c]
    df["confidence"] = df[vote_cols].sum(axis=1)
    df["anomaly_label"] = df["ensemble_l12"].astype(int)
    log.info(
        "Loaded %d windows from %s (anomalous: %d)",
        len(df), csv_path, df["anomaly_label"].sum(),
    )
    return df


def select_candidates(df: pd.DataFrame, cfg: dict, iteration: int) -> pd.DataFrame:
    \"\"\"
    Select top candidates by anomaly confidence, enforcing diversity.
    Focuses on MLFF frames (anomalous) as exploration seeds.
    \"\"\"
    al_cfg = cfg["active_learning"]
    max_cand = al_cfg["candidate_selection"]["max_candidates_per_iter"]

    # Prefer MLFF-sourced anomalous frames as they represent force-field
    # uncertainty regions most relevant for retraining
    anomalous = df[df["anomaly_label"] == 1].copy()
    if anomalous.empty:
        log.warning("No anomalous frames found; using all frames by confidence rank")
        anomalous = df.copy()

    # Sort by composite score: confidence desc, then l2_if_score asc (more negative = more anomalous)
    anomalous = anomalous.sort_values(
        ["confidence", "l2_if_score"], ascending=[False, True]
    ).reset_index(drop=True)

    # Enforce per-file diversity: at most ceil(max_cand / n_files) per source file
    if "file" in anomalous.columns:
        n_files = anomalous["file"].nunique()
        per_file = max(1, max_cand // max(n_files, 1))
        diverse = (
            anomalous.groupby("file", group_keys=False)
            .apply(lambda g: g.head(per_file))
            .reset_index(drop=True)
        )
    else:
        diverse = anomalous

    candidates = diverse.head(max_cand).copy()
    candidates["al_iteration"] = iteration
    log.info("Selected %d candidates for DFT labeling (iteration %d)", len(candidates), iteration)
    return candidates


# ---------------------------------------------------------------------------
# QE input file generation
# ---------------------------------------------------------------------------

QE_INPUT_TEMPLATE = \"\"\"&CONTROL
  calculation  = 'scf'
  prefix       = '{prefix}'
  pseudo_dir   = '{pseudo_dir}'
  outdir       = '{outdir}'
  tprnfor      = .true.
  tstress      = .true.
  verbosity    = 'low'
/
&SYSTEM
  ibrav        = 0
  nat          = {nat}
  ntyp         = {ntyp}
  ecutwfc      = {ecutwfc}
  ecutrho      = {ecutrho}
  occupations  = 'smearing'
  smearing     = 'cold'
  degauss      = {degauss}
  nspin        = {nspin}
  {magnetization_lines}
  vdw_corr     = 'grimme-d3'
/
&ELECTRONS
  conv_thr     = {conv_thr}
  mixing_beta  = {mixing_beta}
  mixing_mode  = 'local-TF'
  electron_maxstep = 200
/
ATOMIC_SPECIES
{atomic_species}
ATOMIC_POSITIONS angstrom
{atomic_positions}
K_POINTS automatic
{kx} {ky} {kz}  0 0 0
CELL_PARAMETERS angstrom
{cell_vectors}
\"\"\"


def write_qe_input(candidate_row: pd.Series, structure_xyz: str,
                   cfg: dict, out_dir: Path) -> Path:
    \"\"\"
    Write a QE SCF input file for a single candidate structure.
    `structure_xyz` is the atomic positions block in XYZ format (raw string).
    \"\"\"
    qe = cfg["qe"]
    sys_cfg = cfg["system"]
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"scf_{candidate_row.get('window_idx', 'X'):05d}.in"
    out_path = out_dir / fname

    # Parse XYZ block: skip first two header lines
    lines = structure_xyz.strip().splitlines()
    coord_lines = lines[2:] if len(lines) > 2 else lines
    nat = len(coord_lines)
    elements = sorted(set(l.split()[0] for l in coord_lines if l.strip()))
    ntyp = len(elements)

    atomic_species = "\\n".join(
        f"  {el}  {_get_mass(el)}  {qe['pseudopotentials'].get(el, el + '.UPF')}"
        for el in elements
    )
    atomic_positions = "\\n".join(f"  {l.strip()}" for l in coord_lines)

    mag_lines = "\\n  ".join(
        f"starting_magnetization({i+1}) = {qe['starting_magnetization'].get(el, 0.0)}"
        for i, el in enumerate(elements)
    )

    kpts = qe["k_points"]
    content = QE_INPUT_TEMPLATE.format(
        prefix=qe["prefix"],
        pseudo_dir=qe["pseudo_dir"],
        outdir=qe["outdir"],
        nat=nat,
        ntyp=ntyp,
        ecutwfc=qe["ecutwfc"],
        ecutrho=qe["ecutrho"],
        degauss=qe["degauss"],
        nspin=qe["nspin"],
        magnetization_lines=mag_lines,
        conv_thr=f"{qe['conv_thr']:.1e}",
        mixing_beta=qe["mixing_beta"],
        atomic_species=atomic_species,
        atomic_positions=atomic_positions,
        kx=kpts[0], ky=kpts[1], kz=kpts[2],
        cell_vectors="  # insert 3x3 lattice vectors here",
    )
    out_path.write_text(content)
    return out_path


def _get_mass(element: str) -> float:
    masses = {"Sb": 121.760, "Te": 127.600, "Cr": 51.996,
               "Se": 78.971, "Bi": 208.980}
    return masses.get(element, 1.0)


# ---------------------------------------------------------------------------
# MACE committee training
# ---------------------------------------------------------------------------

def train_mace_committee(cfg: dict, dry_run: bool = False) -> list[Path]:
    \"\"\"
    Launch 4 MACE training runs (one per committee seed) via subprocess.
    Returns paths to the saved model files.
    \"\"\"
    mace_cfg = cfg["mace"]
    committee_cfg = cfg["committee"]
    model_dir = Path(mace_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    model_paths = []
    for idx, seed in enumerate(committee_cfg["seeds"]):
        model_name = f"{committee_cfg['model_prefix']}_{idx}_seed{seed}"
        model_path = model_dir / f"{model_name}.model"
        model_paths.append(model_path)

        cmd = [
            "mace_run_train",
            f"--name={model_name}",
            f"--train_file={mace_cfg['train_file']}",
            f"--valid_file={mace_cfg['valid_file']}",
            f"--model_dir={model_dir}",
            f"--r_max={mace_cfg['r_max']}",
            f"--num_radial_basis={mace_cfg['num_radial_basis']}",
            f"--max_L={mace_cfg['max_L']}",
            f"--num_channels={mace_cfg['num_channels']}",
            f"--num_interactions={mace_cfg['num_interactions']}",
            f"--batch_size={mace_cfg['batch_size']}",
            f"--lr={mace_cfg['lr']}",
            f"--max_num_epochs={mace_cfg['max_num_epochs']}",
            f"--patience={mace_cfg['patience']}",
            f"--scheduler_gamma={mace_cfg['scheduler_gamma']}",
            f"--energy_weight={mace_cfg['energy_weight']}",
            f"--forces_weight={mace_cfg['forces_weight']}",
            f"--stress_weight={mace_cfg['stress_weight']}",
            f"--seed={seed}",
            f"--device={mace_cfg['device']}",
            f"--default_dtype={mace_cfg['default_dtype']}",
        ]

        if dry_run:
            log.info("[DRY-RUN] Would run: %s", " ".join(cmd))
        else:
            log.info("Training committee model %d/%d (seed=%d) ...",
                     idx + 1, len(committee_cfg["seeds"]), seed)
            result = subprocess.run(cmd, capture_output=False, text=True)
            if result.returncode != 0:
                log.error("MACE training failed for seed=%d (exit %d)", seed, result.returncode)
                raise RuntimeError(f"MACE training failed for seed={seed}")
            log.info("Committee model %d trained -> %s", idx, model_path)

    return model_paths


# ---------------------------------------------------------------------------
# Committee uncertainty computation
# ---------------------------------------------------------------------------

def compute_committee_uncertainty(model_paths: list[Path], explore_xyz: str,
                                  cfg: dict, dry_run: bool = False) -> np.ndarray:
    \"\"\"
    Run each committee model on the exploration set and compute force std.
    Returns array of shape (n_frames,) with per-frame force uncertainty (eV/Ang).
    In dry-run mode returns random synthetic uncertainties for testing.
    \"\"\"
    if dry_run:
        n_frames = cfg["active_learning"]["n_explore_frames"]
        rng = np.random.default_rng(42)
        sigma = rng.uniform(0.0, 0.5, size=n_frames)
        log.info("[DRY-RUN] Synthetic uncertainties: mean=%.4f, max=%.4f",
                 sigma.mean(), sigma.max())
        return sigma

    all_forces = []
    for mp in model_paths:
        if not mp.exists():
            raise FileNotFoundError(f"Committee model not found: {mp}")
        cmd = ["mace_eval", f"--model={mp}", f"--xyz={explore_xyz}", "--output_forces"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"mace_eval failed: {result.stderr[:200]}")
        # Parse forces from stdout (assumed n_frames x n_atoms x 3 array)
        forces = _parse_forces_stdout(result.stdout)
        # Per-frame RMS force magnitude: shape (n_frames,)
        all_forces.append(np.sqrt((forces ** 2).sum(axis=-1).mean(axis=-1)))

    stacked = np.stack(all_forces, axis=0)          # (n_committee, n_frames)
    uncertainty = stacked.std(axis=0)               # (n_frames,)
    log.info("Force uncertainty: mean=%.4f, max=%.4f eV/Ang",
             uncertainty.mean(), uncertainty.max())
    return uncertainty


def _parse_forces_stdout(stdout: str) -> np.ndarray:
    \"\"\"Minimal parser: expects whitespace-separated float lines from mace_eval.\"\"\"
    rows = [list(map(float, l.split())) for l in stdout.splitlines() if l.strip()]
    return np.array(rows).reshape(-1, 82, 3)


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------

def check_convergence(uncertainty: np.ndarray, cfg: dict) -> dict:
    al = cfg["active_learning"]
    sigma_lo = al["sigma_lo"]
    sigma_hi = al["sigma_hi"]
    conv_threshold = al["convergence"]["min_accurate_fraction"]

    n_accurate = int((uncertainty <= sigma_lo).sum())
    n_candidate = int(((uncertainty > sigma_lo) & (uncertainty <= sigma_hi)).sum())
    n_failed = int((uncertainty > sigma_hi).sum())
    n_total = len(uncertainty)
    frac_accurate = n_accurate / n_total if n_total > 0 else 0.0

    converged = frac_accurate >= conv_threshold
    log.info(
        "Convergence: accurate=%.1f%% (%d/%d), candidates=%d, failed=%d | %s",
        frac_accurate * 100, n_accurate, n_total, n_candidate, n_failed,
        "CONVERGED" if converged else "not yet",
    )
    return {
        "n_total": n_total,
        "n_accurate": n_accurate,
        "n_candidate": n_candidate,
        "n_failed": n_failed,
        "frac_accurate": frac_accurate,
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# Main AL loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MACE Active Learning Loop — 2D Sb2Te3/Cr")
    parser.add_argument("--config", default="config_al.yaml")
    parser.add_argument("--iter", type=int, default=0, dest="start_iter",
                        help="Starting iteration index (for resuming)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate workflow without running DFT/training")
    args = parser.parse_args()

    cfg = load_config(args.config)
    al_cfg = cfg["active_learning"]
    results_dir = Path(al_cfg["results_dir"])
    data_dir = Path(al_cfg["data_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== MACE Active Learning Loop: %s ===", cfg["system"]["name"])
    log.info("Elements: %s | n_atoms: %d | max_iter: %d",
             cfg["system"]["elements"], cfg["system"]["n_atoms"], al_cfg["max_iterations"])
    if args.dry_run:
        log.info("*** DRY-RUN MODE — no DFT or MACE training will be executed ***")

    # Load anomaly detection results for warm-start candidate selection
    anomaly_csv = al_cfg.get("anomaly_csv", "results/reports/ensemble_comparison.csv")
    anomaly_df = load_anomaly_candidates(anomaly_csv, cfg)

    converged = False
    for iteration in range(args.start_iter, al_cfg["max_iterations"]):
        iter_start = time.time()
        log.info("--- Iteration %d / %d ---", iteration + 1, al_cfg["max_iterations"])

        # Step 1: Select candidates
        candidates = select_candidates(anomaly_df, cfg, iteration)
        qe_input_dir = data_dir / f"iter_{iteration:02d}" / "qe_inputs"

        # Step 2: Generate QE inputs (placeholder structures — real use reads XYZ)
        log.info("Generating %d QE input files in %s", len(candidates), qe_input_dir)
        if not args.dry_run:
            placeholder_xyz = (
                f"{cfg['system']['n_atoms']}\\nframe from anomaly candidate\\n"
                + "\\n".join(f"Sb  0.0  0.0  {i*3.0:.2f}" for i in range(cfg["system"]["n_atoms"]))
            )
            for _, row in candidates.iterrows():
                write_qe_input(row, placeholder_xyz, cfg, qe_input_dir)
            log.info("QE inputs written to %s", qe_input_dir)
        else:
            log.info("[DRY-RUN] Would write %d QE input files", len(candidates))

        # Step 3: (In production) submit QE jobs and collect outputs
        # This section submits via sbatch and polls; skipped in dry-run
        if not args.dry_run:
            log.info("Submitting QE DFT jobs (not implemented in this template).")
            log.info("In production: call sbatch for each .in file and poll via sacct.")
        else:
            log.info("[DRY-RUN] Would submit %d QE SCF calculations", len(candidates))

        # Step 4: Retrain MACE committee
        model_paths = train_mace_committee(cfg, dry_run=args.dry_run)

        # Step 5: Evaluate committee uncertainty on exploration set
        explore_xyz = str(data_dir / f"iter_{iteration:02d}" / "explore.xyz")
        uncertainty = compute_committee_uncertainty(model_paths, explore_xyz, cfg, dry_run=args.dry_run)

        # Step 6: Apply sigma filter for next iteration's candidates
        sigma_lo = al_cfg["sigma_lo"]
        sigma_hi = al_cfg["sigma_hi"]
        mask_lo = uncertainty <= sigma_lo
        mask_candidate = (uncertainty > sigma_lo) & (uncertainty <= sigma_hi)
        mask_hi = uncertainty > sigma_hi

        # Step 7: Convergence check (only after min_iterations)
        conv_stats = check_convergence(uncertainty, cfg)
        min_iters = al_cfg["convergence"]["min_iterations"]

        # Write iteration summary
        summary = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "n_candidates_selected": len(candidates),
            "n_qe_jobs": len(candidates),
            "committee_seeds": cfg["committee"]["seeds"],
            "sigma_lo": sigma_lo,
            "sigma_hi": sigma_hi,
            "uncertainty_mean": float(uncertainty.mean()),
            "uncertainty_max": float(uncertainty.max()),
            "frames_accurate": int(mask_lo.sum()),
            "frames_candidate": int(mask_candidate.sum()),
            "frames_failed": int(mask_hi.sum()),
            "frac_accurate": conv_stats["frac_accurate"],
            "converged": conv_stats["converged"],
            "elapsed_s": round(time.time() - iter_start, 2),
            "dry_run": args.dry_run,
        }
        summary_path = results_dir / f"al_iteration_{iteration:02d}.json"
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        log.info("Iteration summary -> %s", summary_path)

        if conv_stats["converged"] and iteration >= min_iters - 1:
            log.info("Convergence criterion met at iteration %d. Stopping.", iteration)
            converged = True
            break

    status = "CONVERGED" if converged else "MAX_ITERATIONS_REACHED"
    log.info("=== AL loop finished: %s ===", status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""

# =============================================================================
# FILE 3 — submit_al.sh
# =============================================================================

submit_sh = """\
#!/usr/bin/env bash
# ============================================================
# submit_al.sh — SLURM batch script for MACE Active Learning
# System: 2D Sb2Te3 with Cr dopants (82 atoms, bilayer)
# ============================================================

#SBATCH --job-name=mace_al_Sb2Te3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/al_loop_%j.out
#SBATCH --error=logs/al_loop_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@jhu.edu   # <-- replace with your address

# ------------------------------------------------------------
# 0. Strict error handling
# ------------------------------------------------------------
set -euo pipefail
trap 'echo "[ERROR] Script failed at line $LINENO — exit code $?" >&2; exit 1' ERR

# ------------------------------------------------------------
# 1. Create log directory if needed
# ------------------------------------------------------------
mkdir -p logs

echo "============================================================"
echo "  MACE AL Loop — 2D Sb2Te3/Cr"
echo "  SLURM Job : $SLURM_JOB_ID"
echo "  Node      : $SLURMD_NODENAME"
echo "  Start     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ------------------------------------------------------------
# 2. Activate Python environment
#    Try conda env 'mace' first; fall back to module system.
# ------------------------------------------------------------
ENV_ACTIVATED=0

if command -v conda &>/dev/null; then
    # Conda is available — initialise and activate
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    if conda activate mace 2>/dev/null; then
        echo "[env] Conda environment 'mace' activated"
        ENV_ACTIVATED=1
    else
        echo "[env] WARNING: conda env 'mace' not found, trying module system..."
    fi
fi

if [[ $ENV_ACTIVATED -eq 0 ]]; then
    if module avail quantum-espresso 2>&1 | grep -q quantum-espresso; then
        module load quantum-espresso/7.2
        echo "[env] Loaded module: quantum-espresso/7.2"
    fi
    if module avail python 2>&1 | grep -q python; then
        module load python/3.11
        echo "[env] Loaded module: python/3.11"
    fi
    # Install mace-torch if not present (HPC without conda env)
    python -c "import mace" 2>/dev/null || {
        echo "[env] Installing mace-torch in user site ..."
        pip install --user mace-torch
    }
    ENV_ACTIVATED=1
fi

# Verify MACE is importable
python -c "import mace; print('[env] MACE version:', mace.__version__)" || {
    echo "[ERROR] MACE not importable. Check your environment setup." >&2
    exit 1
}

# Verify PyYAML
python -c "import yaml" || {
    echo "[env] Installing PyYAML ..."
    pip install --user pyyaml
}

# ------------------------------------------------------------
# 3. Environment variables
# ------------------------------------------------------------
export OMP_NUM_THREADS=1              # avoid over-subscription with MPI
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# CUDA device visibility: expose GPU 0 (first GPU on the node)
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}

# Optional: pin NCCL / cuDNN to deterministic mode
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONFAULTHANDLER=1          # better tracebacks on segfault

echo "[env] OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[env] Python: $(which python)"
echo "[env] GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    2>/dev/null || echo "  (nvidia-smi not available)"

# ------------------------------------------------------------
# 4. Validate required files before starting
# ------------------------------------------------------------
CONFIG_FILE="config_al.yaml"
SCRIPT_FILE="run_al_loop.py"

for f in "$CONFIG_FILE" "$SCRIPT_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "[ERROR] Required file not found: $f" >&2
        exit 1
    fi
done
echo "[check] Required files present: $CONFIG_FILE, $SCRIPT_FILE"

# Check pseudopotential directory from config
PSEUDO_DIR=$(python -c "
import yaml
with open('config_al.yaml') as f:
    c = yaml.safe_load(f)
print(c['qe']['pseudo_dir'])
")
if [[ ! -d "$PSEUDO_DIR" ]]; then
    echo "[WARNING] Pseudopotential directory not found: $PSEUDO_DIR" >&2
    echo "          Download from https://pseudopotentials.quantum-espresso.org" >&2
fi

# ------------------------------------------------------------
# 5. Parse optional --iter argument (for resuming from checkpoint)
# ------------------------------------------------------------
START_ITER=${AL_START_ITER:-0}
echo "[run] Starting from AL iteration $START_ITER"

# ------------------------------------------------------------
# 6. Run the active learning loop
# ------------------------------------------------------------
echo ""
echo "[run] Launching run_al_loop.py at $(date '+%H:%M:%S') ..."
echo ""

python run_al_loop.py \\
    --config "$CONFIG_FILE" \\
    --iter "$START_ITER"

EXIT_CODE=$?

# ------------------------------------------------------------
# 7. Report completion
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "  AL Loop finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Exit code : $EXIT_CODE"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "  Status    : SUCCESS"
    # Summarise last iteration JSON
    LAST_JSON=$(ls results/al/al_iteration_*.json 2>/dev/null | sort | tail -1)
    if [[ -n "$LAST_JSON" ]]; then
        echo "  Last summary: $LAST_JSON"
        python -c "
import json, sys
with open('$LAST_JSON') as f:
    d = json.load(f)
print(f'  Iteration      : {d[\"iteration\"]}')
print(f'  Frac accurate  : {d[\"frac_accurate\"]:.3f}')
print(f'  Converged      : {d[\"converged\"]}')
print(f'  Elapsed (s)    : {d[\"elapsed_s\"]}')
" 2>/dev/null || true
    fi
else
    echo "  Status    : FAILED (check logs/al_loop_${SLURM_JOB_ID}.err)"
fi

echo "============================================================"
exit $EXIT_CODE
"""


# =============================================================================
# Convenience accessor
# =============================================================================

def al_loop_files() -> dict:
    """Return all three AL loop file contents as a dict.

    Keys:
        config_yaml  — content of config_al.yaml
        run_script   — content of run_al_loop.py
        submit_sh    — content of submit_al.sh
    """
    return {
        "config_yaml": config_yaml,
        "run_script": run_script,
        "submit_sh": submit_sh,
    }


if __name__ == "__main__":
    # Quick smoke-test: print first line of each file
    files = al_loop_files()
    for key, content in files.items():
        first_line = content.splitlines()[0]
        print(f"{key}: {first_line!r}  ({len(content)} chars)")
