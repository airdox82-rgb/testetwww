
from huggingface_hub import list_repo_files

repo_id = "kevinwang676/GPT-SoVITS-v4"

try:
    print(f"Listing files in repository: {repo_id}")
    files = list_repo_files(repo_id)
    for file in files:
        print(file)
except Exception as e:
    print(f"An error occurred: {e}")
