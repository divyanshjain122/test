"""Streamlit Dashboard Launcher

Launches the JSF dashboard with proper imports.
Run with: python run_dashboard.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Now run the dashboard
import streamlit.web.cli as stcli

if __name__ == "__main__":
    # Set the file to run
    dashboard_file = src_path / "jsf" / "dashboard" / "app.py"
    
    sys.argv = [
        "streamlit",
        "run",
        str(dashboard_file),
        "--server.headless=true",
    ]
    
    sys.exit(stcli.main())
