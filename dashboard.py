"""
Root entry point for Streamlit dashboard.
Allows running: streamlit run dashboard.py
"""

from pathlib import Path
import runpy

# Forward execution to dashboard/dashboard.py
target_file = Path(__file__).resolve().parent / "dashboard" / "dashboard.py"
runpy.run_path(str(target_file), run_name="__main__")
