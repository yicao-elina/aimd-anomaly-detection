# Streamlit Bug Fix & Launch Guide

## ✅ Issues Fixed

### Problem 1: Permission Denied on `app/dashboard.py`
**Cause:** File wasn't executable  
**Fix:**
```bash
chmod +x app/dashboard.py
```

### Problem 2: Streamlit Looking for Wrong File
**Cause:** Streamlit expected `streamlit_app.py` but found `app/dashboard.py`  
**Fix:**
```bash
ln -sf app/dashboard.py streamlit_app.py
```

## ✅ Status: RUNNING

The dashboard is now successfully running on port **8501**.

**View it here:** http://localhost:8501

---

## Quick Launch Reference

### One-Time Setup (first time only)
```bash
cd /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25-npj_2D-Sb2Te3/aimd
chmod +x app/dashboard.py
ln -sf app/dashboard.py streamlit_app.py
```

### Launch Command (every time)
```bash
cd /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25-npj_2D-Sb2Te3/aimd
/Users/alina/anaconda3/envs/agentic/bin/streamlit run app/dashboard.py
```

### Custom Port (if 8501 is busy)
```bash
/Users/alina/anaconda3/envs/agentic/bin/streamlit run app/dashboard.py --server.port 8502
```

---

## Dashboard Pages

1. **Data Overview** - File inventory, frame counts, temperature breakdown
2. **Feature Analysis** - Violin plots, distributions, interactive explorer
3. **Anomaly Detection** - Timeline, detector agreement, heatmaps
4. **AIMD vs MLFF Comparison** - Z-score analysis, feature comparison
5. **AI Analysis Assistant** - LLM-powered insights (requires Ollama)

---

## Verify Setup

```bash
# Check Ollama running (optional, for Page 5 only)
curl -s http://localhost:11434 && echo "✓ Ollama ready"

# Verify models exist
ls results/models/anomaly_framework.pkl results/models/lstm_autoencoder.pt

# Verify data processed
ls data/processed/features_aimd.npz data/processed/features_mlff.npz
```

---

## Next Steps

1. ✓ Open http://localhost:8501 in browser
2. Navigate through the 5 pages using the sidebar
3. Follow the [run_streamlit.md](run_streamlit.md) demo sequence for full walkthrough
4. Record video using QuickTime or take screenshots (see [run_streamlit.md](run_streamlit.md))

---

**Dashboard is live and ready for demo!** 🚀
