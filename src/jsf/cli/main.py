"""JSF-Core Command Line Interface.

Main entry point for the `jsf` command after pip install.
Usage:
    jsf dashboard       Launch the Streamlit monitoring dashboard
    jsf setup-telegram  Configure Telegram bot for trading alerts
    jsf version         Show installed version
    jsf --help          Show all commands
"""

import sys
import argparse


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        print("Error: Streamlit is not installed.")
        print("Install it with: pip install jsf-core[dashboard]")
        sys.exit(1)

    # Get the path to app.py within the installed package
    import jsf.dashboard.app as dashboard_app
    import os

    app_path = os.path.abspath(dashboard_app.__file__)

    sys.argv = ["streamlit", "run", app_path, "--server.headless", "true"]
    sys.exit(stcli.main())


def show_version():
    """Show the installed jsf-core version."""
    try:
        from jsf import __version__
        print(f"jsf-core version {__version__}")
    except ImportError:
        print("jsf-core is not properly installed")
        sys.exit(1)


def setup_telegram():
    """Run the Telegram bot setup wizard."""
    try:
        from jsf.cli.setup_telegram import setup_telegram_bot
        setup_telegram_bot()
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure jsf-core[alerts] is installed.")
        sys.exit(1)


def cli():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="jsf",
        description="JSF-Core: Quantitative Research & Backtesting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  dashboard       Launch the real-time Streamlit monitoring dashboard
  setup-telegram  Configure Telegram bot for trading alerts
  version         Show installed jsf-core version

Examples:
  jsf dashboard                 # Launch dashboard at http://localhost:8501
  jsf setup-telegram            # Interactive Telegram setup wizard
  jsf version                   # Print version number

Documentation: https://github.com/JaiAnshSB26/JBAC-Strategy-Foundry
        """,
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=["dashboard", "setup-telegram", "version"],
        help="Command to run",
    )

    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Show version and exit",
    )

    args = parser.parse_args()

    if args.version:
        show_version()
        return

    if args.command is None:
        parser.print_help()
        return

    if args.command == "dashboard":
        launch_dashboard()
    elif args.command == "setup-telegram":
        setup_telegram()
    elif args.command == "version":
        show_version()
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
