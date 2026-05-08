import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def run_single_year(year, target_script):
    """Function to be executed in a separate process for a specific year."""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    print(f"🚀 Starting Year {year}...")

    cmd = [
        sys.executable,
        str(target_script),
        "--start", start_date,
        "--end", end_date
    ]

    try:
        # We use capture_output=True to keep the logs from overlapping in the console
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return f"✅ Year {year} completed."
    except subprocess.CalledProcessError as e:
        return f"❌ Error during year {year}: {e.stderr}"

def run_batch_backtests_parallel():
    current_dir = Path(__file__).parent
    target_script = current_dir.parent / "hugging_face_ml.py"
    years = [2019, 2020, 2021, 2022, 2023]
    
    # max_workers=None will default to the number of processors on your machine
    print(f"🔥 Starting Parallel Execution for: {target_script.name}")
    
    with ProcessPoolExecutor(max_workers=None) as executor:
        # Submit all tasks to the pool
        futures = [executor.submit(run_single_year, year, target_script) for year in years]
        
        # Collect results as they finish
        for future in futures:
            print(future.result())

if __name__ == "__main__":
    run_batch_backtests_parallel()