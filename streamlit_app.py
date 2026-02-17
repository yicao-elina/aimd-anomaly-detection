# Entry point for Streamlit Cloud.
# In the Streamlit Cloud UI, set main file path to: app/dashboard.py
# For local use: streamlit run app/dashboard.py
import runpy, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
runpy.run_path("app/dashboard.py", run_name="__main__")
