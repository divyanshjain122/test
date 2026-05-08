import os
import sys
from huggingface_hub import snapshot_download

def download_fnspid_dataset(repo_id="Zihan1004/FNSPID"):
    """
    Downloads the FNSPID dataset from Hugging Face.
    FNSPID is a large-scale Financial News dataset.
    """
    print("\n" + "="*50)
    print(f"STARTING DOWNLOAD: {repo_id}")
    print("ESTIMATED SIZE: ~30GB")
    print("="*50 + "\n", flush=True)

    try:
        # snapshot_download is the recommended way to download full HF datasets
        path = snapshot_download(
            repo_id=repo_id, 
            repo_type="dataset",
            # This allows the script to pick up where it left off if your Wi-Fi drops
            resume_download=True,
            # For 30GB, using more threads can speed up the process
            max_workers=8 
        )
        
        print("\n" + "-"*50)
        print(f"SUCCESS!")
        print(f"DATASET LOCATION: {path}")
        print("-"*50 + "\n")
        
        return path

    except KeyboardInterrupt:
        print("\n\nDownload paused by user. Run the script again to resume.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred:\n{e}")
        return None

if __name__ == "__main__":
    # Ensure the script is being run inside the project environment
    dataset_path = download_fnspid_dataset()