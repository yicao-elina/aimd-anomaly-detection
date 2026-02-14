# CLAUDE.md — Project Context for AI Assistant

## Who You Are Helping

Alina is a PhD researcher at Johns Hopkins working on a Nature Partner Journal (npj) paper about
2D Sb₂Te₃ with Cr dopants. This `aimd/` folder contains AIMD trajectory analysis code that
**also doubles as a portfolio project** for a **Qualcomm Summer 2026 internship** interview.
The dual purpose shapes every design decision: scientifically rigorous *and* interview-demonstrable.

---

## Python Environment — CRITICAL

**Always use the `agentic` conda environment.**

The system `conda run` command is broken (library error with libmamba). Use the **full Python path**:

```bash
/Users/alina/anaconda3/envs/agentic/bin/python script.py
/Users/alina/anaconda3/envs/agentic/bin/pip install package
/Users/alina/anaconda3/envs/agentic/bin/streamlit run app/dashboard.py
```

Never use bare `python`, `pip`, or `conda run` — they will use the wrong environment.

---

## Project Root

```
/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25-npj_2D-Sb2Te3/aimd/
```

All scripts must be run from this directory. Relative paths in code assume this root.

---

## Project Architecture

```
aimd/
├── src/core/
│   ├── loaders.py             # TrajectoryLoader: parses .xyz AIMD files
│   ├── feature_extractors.py  # 22 features, sliding windows, atom-count-agnostic
│   ├── detectors.py           # Level 1 (3σ) + Level 2 (IF + SVM) ensemble
│   └── models.py              # LSTM Autoencoder (PyTorch, Level 3)
├── app/dashboard.py           # Streamlit 4-page dashboard (AIMD only)
├── scripts/run_full_pipeline.py  # Runs Steps 2–6 end-to-end
├── data/
│   ├── raw/temperature/       # AIMD at 300K, 600K, 1200K (Cr2 system)
│   ├── raw/concentration/     # 7 Cr configurations at 600K
│   ├── raw/mlff/              # MLFF trajectory = the "anomalous" test set
│   └── processed/             # Features (.npz), metadata (.csv), summary (.json)
├── results/
│   ├── figures/               # 6 PNGs (distributions, timeline, importance, etc.)
│   ├── models/                # anomaly_framework.pkl, lstm_autoencoder.pt
│   └── reports/               # ensemble_comparison.csv, aimd_vs_mlff_analysis.md
└── docs/                      # Step completion notes
```

---

## Data Facts

| Source | Files | Frames | Atom count | Role |
|--------|-------|--------|------------|------|
| AIMD temperature | 3 | ~3,500 | 82 | Training (normal) |
| AIMD concentration | 9 | ~14,500 | 80–86 | Training (normal) |
| MLFF | 1 | 8,038 | 82 | Test (anomalous) |

**Variable atom counts** (80–86) are handled by extracting only aggregate statistics per window
(mean/std/etc. over all atoms), so the feature dimension is always 22 regardless of atom count.
Never try to pad or align atom counts — the aggregate approach is by design.

The `aimd_merged_traj_bulk.xyz` file is a merged trajectory from `concentration/` — it is included
in training but is redundant with the individual files. Safe to use.

---

## Key Design Decisions (Do Not Change Without Discussion)

1. **Window size = 50 frames, stride = 10** — balances temporal context and sample count.
2. **22 features** — displacement stats (7) + dynamics (5) + frequency (3) + MSD (4) + energy (3).
3. **MLFF = fully anomalous test set** — we only have 1 MLFF file; it is treated as the anomalous
   class, not mixed into training.
4. **AIMD-only dashboard** — no synthetic GPU demo needed (the framework generalizability is
   demonstrated through the separate `/anomaly-demo` skill instead).
5. **L1+L2 ensemble is the primary result** — LSTM has limited training data (~131 sequences)
   so its 31% false-positive rate on AIMD is expected and noted. L1+L2 gives clean 3.6% vs 100%.
6. **Contamination rate = 5%** for Isolation Forest and One-Class SVM.

---

## Current Pipeline Results

```
AIMD anomaly rate (L1+L2):  3.6%   ← close to 5% contamination prior  ✅
MLFF anomaly rate (L1+L2): 100.0%  ← perfectly detected as anomalous  ✅
Detection ratio:             27.9×
LSTM val loss:               0.137 (converged at epoch 46)
LSTM anomaly threshold:      0.202 (95th percentile of val reconstruction error)
```

---

## LLM Component (glm-5:cloud via Ollama)

The dashboard Page 5 "🤖 AI Analysis" uses Ollama with `glm-5:cloud`.
**Ollama must be running** (`ollama serve` or the Ollama desktop app) for Page 5 to work.
Pages 1–4 work entirely offline with no LLM dependency.

Key file: `src/core/llm_analyst.py` — `OllamaAnalyst` class.

Anti-hallucination design:
- `build_data_context()` packages only actual computed statistics into the LLM prompt
- Code generation → `exec()` runs against real numpy arrays (figures cannot lie)
- Mechanism analysis prompt demands `► Claim: ... | Evidence: [value]` format
- The model `glm-5:cloud` was confirmed working: `ollama list` shows it available

If `glm-5:cloud` is unavailable, swap model name in `OllamaAnalyst(model='...')` in `app/dashboard.py:561`.

---

## How to Run

```bash
# Re-run full pipeline (Steps 2–6, ~2 minutes)
/Users/alina/anaconda3/envs/agentic/bin/python scripts/run_full_pipeline.py

# Launch dashboard (all 5 pages, Ollama must be running for Page 5)
/Users/alina/anaconda3/envs/agentic/bin/streamlit run app/dashboard.py

# Install new packages
/Users/alina/anaconda3/envs/agentic/bin/pip install <package>

# Check available Ollama models
ollama list
```

---

## Research Context (for the npj paper)

The science: Sb₂Te₃ is a topological insulator. Adding Cr dopants in 2D bilayer configurations
creates magnetic properties relevant for quantum computing / spintronics. AIMD simulates the true
DFT-level dynamics. MLFF (machine-learned force field via NEB optimization) is a faster
approximation — this project quantifies where MLFF deviates from AIMD.

The anomaly detection *is* the scientific result: it gives a data-driven, quantitative measure
of MLFF accuracy at different configurations and temperatures.

**Key physical features to watch:**
- `rms_velocity`, `crest_factor` — atomic velocity distribution (temperature proxy)
- `msd_slope` — diffusion coefficient (mobility of Cr dopants)
- `dominant_freq` — phonon frequencies (structural stability)
- `energy_mean`, `energy_trend` — thermodynamic consistency

---

## Interview / Qualcomm Context

**Role:** Summer 2026 Intern (11–14 weeks, May–September 2026)
**Team:** HLM (Heavy Lemon Media) game debugging team, within Qualcomm
**Problem they want solved:** The HLM team debugs a growing number of games manually. They want
AI to integrate existing tools and automate as many debugging steps as possible.

**What Qualcomm is looking for:**
- AI/ML and LLM knowledge
- Python and frameworks
- Debugging tools familiarity
- Computer architecture understanding
- Data Structures & Algorithms
- Master's/PhD level preferred

**How this project maps to the role:**
- The anomaly detection framework directly addresses "automate debugging workflows"
- The domain-agnostic design shows ability to apply ML to game profiling data (frame times,
  GPU utilization, draw calls, etc.) — the same tools HLM already uses, just automated
- LLM integration angle: the `/anomaly-demo` skill shows how to combine LLM-driven code
  generation with the ML pipeline to rapidly deploy to new domains

**The core pitch:** "Same detection algorithm, swap the data loader. I built this on AIMD
trajectories, but the framework applies directly to game profiler output — or any time-series
data your debugging tools produce."

**For Qualcomm interview demos, emphasize:**
- `FeatureExtractor` auto-adapts via aggregate statistics (no domain-specific code for the ML)
- `AnomalyDetectionFramework` is pretrained and loads in <1 second — production-ready
- 27.9× detection ratio demonstrates the system works with high confidence
- The `/anomaly-demo` skill shows how quickly a new domain can be onboarded
- Potential LLM integration: auto-generate anomaly reports in natural language, suggest root
  causes, or guide engineers through anomalous frames interactively

---

## Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| `conda run` fails with libarchive error | Use full path `/Users/alina/anaconda3/envs/agentic/bin/python` |
| `ModuleNotFoundError` | `pip install` via full path above |
| Pylance "cannot resolve import" warnings | Ignore — VS Code is not pointing at `agentic` env. Code runs fine. |
| Dashboard won't load | Run pipeline first: `python scripts/run_full_pipeline.py` |
| LSTM high AIMD false-positive rate | Known: small training set (131 sequences). Use L1+L2 for primary results. |
