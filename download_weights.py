import os
from huggingface_hub import snapshot_download

# Create a weights directory in the current folder
WEIGHTS_DIR = os.path.abspath("./weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

print("Downloading Qwen2.5-VL-7B-Instruct...")
snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct", 
    local_dir=os.path.join(WEIGHTS_DIR, "qwen_vl"), 
    local_dir_use_symlinks=False
)

print("All weights successfully downloaded for offline use.")