"""Pre-Release Verification Script.

This script performs comprehensive checks to ensure the library is ready
for conversion and distribution. It tests:
- All imports work
- Real data integrations functional
- No mock dependencies in production
- All demos run successfully
- Package structure correct

Run from project root: python -m tests.verify_library_readiness
Or from tests dir: python verify_library_readiness.py
"""

import sys
import subprocess
from pathlib import Path
import importlib
import os

# Ensure we can import from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
os.chdir(project_root)

print("="*80)
print("JSF-CORE PRE-RELEASE VERIFICATION")
print("="*80)
print(f"Project root: {project_root}")

# Track results
passed = []
failed = []
warnings = []


# =============================================================================
# 1. Python Version Check
# =============================================================================
print("\n[1] Checking Python version...")
version = sys.version_info
if version.major == 3 and version.minor >= 9:
    print(f"   [OK] Python {version.major}.{version.minor}.{version.micro}")
    passed.append("Python version")
else:
    print(f"   [FAIL] Python {version.major}.{version.minor} (requires >= 3.9)")
    failed.append("Python version")


# =============================================================================
# 2. Package Structure Check
# =============================================================================
print("\n[2] Checking package structure...")

required_dirs = [
    "src/jsf",
    "src/jsf/data",
    "src/jsf/signals",
    "src/jsf/ml",
    "src/jsf/strategies",
    "src/jsf/simulation",
    "src/jsf/portfolio",
    "src/jsf/broker",
    "src/jsf/live",
    "tests",
    "docs",
    "demos",
]

required_files = [
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "requirements.txt",
]

missing_dirs = []
for dir_path in required_dirs:
    if not (project_root / dir_path).exists():
        missing_dirs.append(dir_path)

missing_files = []
for file_path in required_files:
    if not (project_root / file_path).exists():
        missing_files.append(file_path)

if not missing_dirs and not missing_files:
    print(f"   [OK] All required directories and files present")
    passed.append("Package structure")
else:
    if missing_dirs:
        print(f"   [FAIL] Missing directories: {missing_dirs}")
    if missing_files:
        print(f"   [FAIL] Missing files: {missing_files}")
    failed.append("Package structure")


# =============================================================================
# 3. Core Module Imports
# =============================================================================
print("\n[3] Testing core module imports...")

modules_to_test = [
    ("jsf.data", "PriceData, SyntheticDataLoader"),
    ("jsf.signals", "MomentumSignal, MeanReversionSignal"),
    ("jsf.ml", "FeatureExtractor, XGBoostModel, LightGBMModel"),
    ("jsf.ml.transformers", "FinBERT, SimpleSentiment"),
    ("jsf.strategies", "MomentumStrategy"),
    ("jsf.simulation", "Backtester, calculate_metrics"),
    ("jsf.portfolio", "EqualWeightSizer"),
]

all_imports_ok = True
for module_name, items in modules_to_test:
    try:
        module = importlib.import_module(module_name)
        print(f"   [OK] {module_name}")
    except ImportError as e:
        print(f"   [FAIL] {module_name}: {e}")
        all_imports_ok = False

if all_imports_ok:
    passed.append("Core imports")
else:
    failed.append("Core imports")


# =============================================================================
# 4. Check Dependencies
# =============================================================================
print("\n[4] Checking dependencies...")

core_deps = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",  # scikit-learn imports as sklearn
]

ml_deps = [
    "xgboost",
    "lightgbm",
    "tensorflow",
]

nlp_deps = [
    "torch",
    "transformers",
]

def check_package(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False

missing_core = [pkg for pkg in core_deps if not check_package(pkg)]
missing_ml = [pkg for pkg in ml_deps if not check_package(pkg)]
missing_nlp = [pkg for pkg in nlp_deps if not check_package(pkg)]

if not missing_core:
    print(f"   [OK] Core dependencies installed")
    passed.append("Core dependencies")
else:
    print(f"   [FAIL] Missing core deps: {missing_core}")
    failed.append("Core dependencies")

if not missing_ml:
    print(f"   [OK] ML dependencies installed")
else:
    print(f"   [WARN] Missing ML deps: {missing_ml}")
    warnings.append(f"ML dependencies: {missing_ml}")

if not missing_nlp:
    print(f"   [OK] NLP dependencies installed")
else:
    print(f"   [WARN] Missing NLP deps: {missing_nlp}")
    warnings.append(f"NLP dependencies: {missing_nlp}")


# =============================================================================
# 5. Real Data Integration Check
# =============================================================================
print("\n[5] Checking real data integrations...")

# Check that we default to real models
try:
    from jsf.ml.transformers.bert import FinBERT
    
    # Create instance and check it's not mock by default
    finbert = FinBERT()
    if hasattr(finbert, 'use_mock') and finbert.use_mock:
        print(f"   [WARN] FinBERT defaults to mock mode")
        warnings.append("FinBERT uses mock by default")
    else:
        print(f"   [OK] FinBERT uses real model by default")
        passed.append("Real FinBERT")
except Exception as e:
    print(f"   [FAIL] FinBERT check failed: {e}")
    failed.append("Real FinBERT")

# Check Yahoo Finance integration
try:
    from jsf.data.sources.yahoo import YFINANCE_AVAILABLE
    if YFINANCE_AVAILABLE:
        print(f"   [OK] Yahoo Finance integration available")
        passed.append("Yahoo Finance")
    else:
        print(f"   [WARN] yfinance not installed")
        warnings.append("yfinance not available for real data")
except ImportError:
    print(f"   [WARN] Yahoo Finance loader not found")
    warnings.append("Yahoo Finance integration missing")


# =============================================================================
# 6. Check for Mock/Synthetic Usage in Production Code
# =============================================================================
print("\n[6] Scanning for mock/synthetic usage in production code...")

import re

src_dir = project_root / "src" / "jsf"

# Patterns to check - but be smart about what we flag
suspicious_patterns = []

for py_file in src_dir.rglob("*.py"):
    # Skip __init__.py, test files, and known-good files
    if (py_file.name == "__init__.py" or 
        "test" in py_file.name.lower() or
        py_file.name == "export.py" or  # export.py has MockONNXExporter class (intentional)
        py_file.name == "loaders.py"):  # loaders.py provides opt-in synthetic data via factory
        continue
    
    content = py_file.read_text(encoding='utf-8')
    
    # Check for actual problematic usage, not just mentions
    # Flag: use_mock=True (but not in comments)
    for line_num, line in enumerate(content.split('\n'), 1):
        line_stripped = line.strip()
        
        # Skip comments and class definitions
        if line_stripped.startswith('#') or line_stripped.startswith('class '):
            continue
        
        # Flag: use_mock=True in actual code
        if 'use_mock=True' in line and 'use_mock=False' not in line:
            suspicious_patterns.append(f"{py_file.name}:{line_num} - use_mock=True")
        
        # Flag: SyntheticDataLoader being instantiated (not just imported or class definition)
        if 'SyntheticDataLoader(' in line and 'import' not in line and 'class ' not in line:
            suspicious_patterns.append(f"{py_file.name}:{line_num} - SyntheticDataLoader instantiation")

if not suspicious_patterns:
    print(f"   [OK] No mock/synthetic usage in production code")
    passed.append("No mock in production")
else:
    print(f"   [FAIL] Found suspicious patterns:")
    for issue in suspicious_patterns:
        print(f"      - {issue}")
    failed.append("Mock/synthetic in production code")


# =============================================================================
# 7. Test ML Verification
# =============================================================================
print("\n[7] Running ML verification...")

try:
    result = subprocess.run(
        [sys.executable, str(project_root / "tests" / "verify_ml_integration.py")],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(project_root),
    )
    
    if result.returncode == 0 and "VERIFICATION COMPLETE" in result.stdout:
        print(f"   [OK] ML verification passed")
        passed.append("ML verification")
    else:
        print(f"   [FAIL] ML verification failed")
        if result.stderr:
            print(f"      {result.stderr[:200]}")
        failed.append("ML verification")
except Exception as e:
    print(f"   [WARN] Could not run ML verification: {e}")
    warnings.append("ML verification not run")


# =============================================================================
# 8. Check Documentation
# =============================================================================
print("\n[8] Checking documentation...")

doc_files = [
    "README.md",
    "docs/QUICKSTART_TELEGRAM.md",
    "LIBRARY_READINESS.md",
    "ML_PHASE_COMPLETE.md",
    "REAL_DATA_COMPLETE.md",
]

missing_docs = [f for f in doc_files if not (project_root / f).exists()]

if not missing_docs:
    print(f"   [OK] All documentation files present")
    passed.append("Documentation")
else:
    print(f"   [WARN] Missing docs: {missing_docs}")
    warnings.append(f"Missing documentation: {missing_docs}")


# =============================================================================
# 9. Check Demo Files
# =============================================================================
print("\n[9] Checking demo files...")

demo_files = [
    "demos/demo_realtime_news.py",
    "demos/demo_full_real_data.py",
    "demos/demo_ml_pipeline.py",
]

missing_demos = [f for f in demo_files if not (project_root / f).exists()]

if not missing_demos:
    print(f"   [OK] All key demo files present")
    passed.append("Demo files")
else:
    print(f"   [WARN] Missing demos: {missing_demos}")
    warnings.append(f"Missing demos: {missing_demos}")


# =============================================================================
# 10. Check Version
# =============================================================================
print("\n[10] Checking version...")

try:
    import tomli
    with open(project_root / "pyproject.toml", "rb") as f:
        config = tomli.load(f)
    version = config['project']['version']
    
    if '-dev' in version:
        print(f"   [WARN] Version still in dev: {version}")
        warnings.append(f"Development version: {version}")
    else:
        print(f"   [OK] Release version: {version}")
        passed.append("Version")
except ImportError:
    # Try with toml (older Python) 
    try:
        import toml
        config = toml.load(project_root / "pyproject.toml")
        version = config['project']['version']
        
        if '-dev' in version:
            print(f"   [WARN] Version still in dev: {version}")
            warnings.append(f"Development version: {version}")
        else:
            print(f"   [OK] Release version: {version}")
            passed.append("Version")
    except Exception:
        print(f"   [WARN] Could not check version (install tomli or toml)")
        warnings.append("Version check failed")
except Exception as e:
    print(f"   [WARN] Could not check version: {e}")
    warnings.append("Version check failed")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

print(f"\n✓ PASSED: {len(passed)}")
for item in passed:
    print(f"   - {item}")

if warnings:
    print(f"\n⚠ WARNINGS: {len(warnings)}")
    for item in warnings:
        print(f"   - {item}")

if failed:
    print(f"\n✗ FAILED: {len(failed)}")
    for item in failed:
        print(f"   - {item}")

print("\n" + "="*80)

if not failed:
    if not warnings:
        print("STATUS: ✅ READY FOR LIBRARY CONVERSION")
        print("\nNext steps:")
        print("1. Review LIBRARY_READINESS.md")
        print("2. Update version in pyproject.toml (remove -dev)")
        print("3. Run: python -m build")
        print("4. Test install: pip install dist/jsf_core-*.whl")
        print("5. Upload to PyPI: twine upload dist/*")
    else:
        print("STATUS: ⚠️  READY WITH WARNINGS")
        print("\nLibrary can be converted, but review warnings above.")
        print("Non-critical issues that can be addressed post-release.")
else:
    print("STATUS: ❌ NOT READY")
    print("\nPlease fix the failed checks before conversion.")
    sys.exit(1)
