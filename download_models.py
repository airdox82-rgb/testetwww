import os
from huggingface_hub import hf_hub_download
import shutil

# --- Configuration ---
repo_id = "kevinwang676/GPT-SoVITS-v4"

# Dictionary mapping the file path in the repo to the desired local filename
files_to_download = {
    "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth": "s2Gv4.pth",
    "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth": "vocoder.pth"
}

local_dir = "GPT_SoVITS/pretrained_models"
# -------------------

def download_files_from_hf():
    """Downloads specified files from a Hugging Face repo to a local directory."""
    print(f"Ensuring local directory exists: {local_dir}")
    os.makedirs(local_dir, exist_ok=True)

    for repo_filepath, local_filename in files_to_download.items():
        local_filepath = os.path.join(local_dir, local_filename)
        
        if os.path.exists(local_filepath) and os.path.getsize(local_filepath) > 0:
            print(f"File {local_filename} already exists and is not empty. Skipping.")
            continue

        print(f"Downloading {local_filename} from {repo_id}...")
        try:
            # Download the file to a cache directory
            downloaded_file_cache_path = hf_hub_download(
                repo_id=repo_id,
                filename=repo_filepath,
                resume_download=True
            )
            # Copy the file from the cache to the desired local path
            shutil.copy(downloaded_file_cache_path, local_filepath)
            print(f"Successfully downloaded and placed {local_filename} in {local_dir}")

        except Exception as e:
            print(f"Error downloading {local_filename}: {e}")
            print("Please ensure you have accepted the terms on the Hugging Face model page and are logged in via 'huggingface-cli login' if required.")
            return False
            
    print("\nAll required models have been successfully downloaded.")
    return True

if __name__ == "__main__":
    download_files_from_hf()
