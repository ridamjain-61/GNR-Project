import os
from huggingface_hub import snapshot_download
import easyocr

# Define a local directory to store all weights safely
WEIGHTS_DIR = os.path.abspath("./weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

print("Downloading Qwen2.5-VL-3B-Instruct...")
snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-3B-Instruct", 
    local_dir=f"{WEIGHTS_DIR}/qwen_vl", 
    local_dir_use_symlinks=False
)

print("Downloading Qwen2.5-0.5B-Instruct...")
snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct", 
    local_dir=f"{WEIGHTS_DIR}/qwen_text", 
    local_dir_use_symlinks=False
)

print("Downloading EasyOCR models...")
# This forces EasyOCR to download its models to our custom folder
easyocr.Reader(["en"], gpu=False, model_storage_directory=f"{WEIGHTS_DIR}/easyocr", download_enabled=True)

print("All weights downloaded successfully to local directory.")