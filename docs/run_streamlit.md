
⏺ Good. Now the full guide for running, viewing, and      
  recording the dashboard:                             
                                                          
  ---                                                     
  How to view the dashboard in your browser
                                                          
  Step 0 — Fix Streamlit permissions (one-time setup):
  cd /Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25-npj_2D-Sb2Te3/aimd
  chmod +x app/dashboard.py
  ln -sf app/dashboard.py streamlit_app.py
                                                          
  Step 1 — Make sure Ollama is running (needed only for   
  Page 5):                                                
  # Check if Ollama is already running
  curl -s http://localhost:11434 && echo "Ollama is
  running"
  # If not, start it:
  ollama serve &

  Step 2 — Launch Streamlit (from the project root):
  cd /Users/alina/Library/CloudStorage/OneDrive-JohnsHopki
  ns/Research/25-npj_2D-Sb2Te3/aimd
  /Users/alina/anaconda3/envs/agentic/bin/streamlit run
  app/dashboard.py

  It will print:
    Local URL:  http://localhost:8501
    Network URL: http://192.168.x.x:8501

  → Open http://localhost:8501 in any browser. It
  auto-opens if not already.

  To run on a custom port (e.g. to avoid conflicts):
  /Users/alina/anaconda3/envs/agentic/bin/streamlit run
  app/dashboard.py --server.port 8502

  ---
  Full demo sequence — everything to execute

  Run these in order to show the full capability:

  Phase 1: Pre-flight (terminal)

  # Confirm models and data exist
  ls results/models/   # should show anomaly_framework.pkl
    lstm_autoencoder.pt
  ls results/figures/  # 6 PNGs
  ls data/processed/   # features_aimd.npz
  features_mlff.npz

  # Start Ollama
  curl -s http://localhost:11434 || ollama serve &
  sleep 2

  # Launch dashboard
  cd /Users/alina/Library/CloudStorage/OneDrive-JohnsHopki
  ns/Research/25-npj_2D-Sb2Te3/aimd
  /Users/alina/anaconda3/envs/agentic/bin/streamlit run
  app/dashboard.py

  Phase 2: Demo walk-through (browser at
  http://localhost:8501)

  Step: 1
  Page: Data Overview
  What to do: Load page
  What it shows: File inventory, 1,648 windows, 3
    temperatures
  ────────────────────────────────────────
  Step: 2
  Page: Feature Analysis → Distributions tab
  What to do: Load tab
  What it shows: Violin plots showing AIMD vs MLFF
    divergence
  ────────────────────────────────────────
  Step: 3
  Page: Feature Analysis → Interactive Explorer
  What to do: Select disp_median
  What it shows: z = +7213, AIMD 0.003 Å vs MLFF 17 Å
  ────────────────────────────────────────
  Step: 4
  Page: Anomaly Detection
  What to do: Switch to "MLFF (test)" radio
  What it shows: Timeline all red at confidence=3
  ────────────────────────────────────────
  Step: 5
  Page: Anomaly Detection → Window Zoom
  What to do: Drag slider to windows 0–100
  What it shows: Heatmap shows all 3 detectors firing
  ────────────────────────────────────────
  Step: 6
  Page: Anomaly Detection → Ask AI
  What to do: Click "Analyze selected region"
  What it shows: LLM explains mechanism with evidence
    citations
  ────────────────────────────────────────
  Step: 7
  Page: Anomaly Detection
  What to do: Switch to "AIMD (normal)", move slider
  What it shows: Mostly blue, only ~3.6% red
  ────────────────────────────────────────
  Step: 8
  Page: AIMD vs MLFF
  What to do: Scroll to Z-score chart
  What it shows: disp_median bar extends to z=+7213
  ────────────────────────────────────────
  Step: 9
  Page: AI Analysis → Quick buttons
  What to do: Click "Top deviating features"
  What it shows: LLM generates + runs code live
  ────────────────────────────────────────
  Step: 10
  Page: AI Analysis → text box
  What to do: Type: "Why does MLFF show 100% anomaly rate?
   
    Use both Figure + Analysis mode"
  What it shows: Full figure + grounded mechanism

  ---
  Taking screenshots (macOS)

  # Option A: macOS built-in (recommended)
  # Press Cmd+Shift+4, then click-drag over the browser
  window
  # Screenshot saves to Desktop automatically

  # Option B: Screenshot the whole screen
  # Press Cmd+Shift+3

  # Option C: Scripted screenshot of specific browser
  region
  /usr/sbin/screencapture -x -t png
  ~/Desktop/dashboard_page1.png
  # Then manually crop in Preview

  ---
  Recording a video demo (macOS — no install needed)

  QuickTime Player (built-in, simplest):
  1. Open QuickTime Player → File → New Screen Recording
  2. Click the dropdown arrow → select your microphone if
  narrating
  3. Click the record button → click-drag to select just
  the browser window
  4. Navigate through the demo sequence above
  5. Stop with the toolbar button → File → Save → exports
  .mov

  Convert to MP4 for sharing (if needed):
  # Built-in ffmpeg via Homebrew
  ffmpeg -i ~/Desktop/demo.mov -vcodec h264 -acodec aac
  ~/Desktop/demo.mp4
  # Or if ffmpeg not installed:
  /usr/local/bin/ffmpeg -i ~/Desktop/demo.mov
  ~/Desktop/demo.mp4

  ---
  Taking automated screenshots of every page (optional —
  for README/slides)

  /Users/alina/anaconda3/envs/agentic/bin/pip install -q
  playwright
  /Users/alina/anaconda3/envs/agentic/bin/python -m
  playwright install chromium

  # Then run this script while Streamlit is running:
  /Users/alina/anaconda3/envs/agentic/bin/python - <<'EOF'
  from playwright.sync_api import sync_playwright
  import time

  pages = {
      "data_overview":    "http://localhost:8501",
      "feature_analysis": None,   # need to click sidebar
      "anomaly":          None,
      "comparison":       None,
      "ai_analysis":      None,
  }

  with sync_playwright() as p:
      browser = p.chromium.launch()
      page = browser.new_page(viewport={"width": 1400,
  "height": 900})
      page.goto("http://localhost:8501")
      time.sleep(3)
      page.screenshot(path="results/figures/screenshot_pag
  e1.png", full_page=False)
      print("Saved screenshot_page1.png")
      browser.close()
  EOF

✻ Baked for 1m 37s
