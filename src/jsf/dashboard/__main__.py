"""Allow running the dashboard as a module.

Usage:
    python -m jsf.dashboard

This launches the Streamlit dashboard without needing to know the file path.
"""

import sys
import os


def main():
    """Launch the Streamlit dashboard."""
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        print("Error: Streamlit is not installed.")
        print("Install it with: pip install jsf-core[dashboard]")
        sys.exit(1)

    # Get the path to app.py in this package
    app_path = os.path.join(os.path.dirname(__file__), "app.py")

    sys.argv = ["streamlit", "run", app_path, "--server.headless", "true"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
