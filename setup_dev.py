"""
Development setup script for JSF-Core.

Run this after cloning the repository to set up your development environment.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a shell command and report status."""
    print(f"\n{'='*60}")
    print(f"⚙️  {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Set up the development environment."""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║          JSF-CORE DEVELOPMENT SETUP                      ║
    ║          JBAC Strategy Foundry                           ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    project_root = Path(__file__).parent
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python version: {sys.version}")
    
    steps = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -e .[dev]", "Installing JSF-Core in development mode"),
        ("pre-commit install", "Installing pre-commit hooks"),
        ("pytest --version", "Verifying pytest installation"),
    ]
    
    success_count = 0
    for cmd, description in steps:
        if run_command(cmd, description):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"SETUP SUMMARY: {success_count}/{len(steps)} steps completed")
    print(f"{'='*60}")
    
    if success_count == len(steps):
        print("""
        ✅ Setup complete! You're ready to develop JSF-Core.
        
        Next steps:
        1. Run tests:           make test
        2. Check code style:    make lint
        3. Format code:         make format
        4. Start coding!
        
        See CONTRIBUTING.md for development guidelines.
        """)
        return 0
    else:
        print("""
        ⚠️  Setup incomplete. Please check errors above.
        
        Try running the failed commands manually or check:
        - Python version (requires 3.9+)
        - Internet connection
        - Disk space
        """)
        return 1


if __name__ == "__main__":
    sys.exit(main())
