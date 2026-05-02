import argparse
import os
import sys
import time
import traceback
import random
import pandas as pd

# ==========================================
# 1. ENFORCE STRICT OFFLINE MODE
# ==========================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from models import VLMAnswerer, INT_TO_OPTION


IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp"]


def random_prediction():
    return random.randint(1, 4)


def clean_base_id(value, fallback):
    """
    Robustly normalize an ID / image name coming from CSV.
    Handles values like:
      - 123
      - 123.png
      - "123.png "
      - Path-like strings
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return str(fallback)

    s = str(value).strip()
    if not s:
        return str(fallback)

    s = os.path.basename(s)
    for ext in IMAGE_EXTS:
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
            break
    return s


def resolve_image_path(images_dir, base_id):
    """
    Try multiple common extensions and return the first existing path.
    """
    for ext in IMAGE_EXTS:
        candidate = os.path.join(images_dir, base_id + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="GNR Project Inference Script")
    parser.add_argument("--test_dir", type=str, required=True, help="Absolute path to the test directory")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print("Starting VLM-only self-consistency inference pipeline...")
    print(f"Test directory provided: {args.test_dir}")
    overall_start = time.perf_counter()

    # ==========================================
    # 2. CONSTRUCT PATHS
    # ==========================================
    test_csv_path = os.path.join(args.test_dir, "test.csv")
    images_dir = os.path.join(args.test_dir, "images")
    weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "weights"))

    os.environ["GNR_VLM_MODEL_DIR"] = os.path.join(weights_dir, "qwen_vl")

    if not os.path.exists(test_csv_path):
        print(f"CRITICAL ERROR: test.csv not found at {test_csv_path}")
        sys.exit(1)

    if not os.path.isdir(images_dir):
        print(f"CRITICAL ERROR: images directory not found at {images_dir}")
        sys.exit(1)

    # ==========================================
    # 3. INITIALIZE MODEL (OFFLINE)
    # ==========================================
    print("Loading Qwen2.5-VL strictly from local weights directory...")
    load_start = time.perf_counter()

    try:
        vlm = VLMAnswerer()
        print(f"Model loaded in {time.perf_counter() - load_start:.1f}s")
        print(f"VLM Status: {vlm.available}")
    except Exception as e:
        print(f"CRITICAL ERROR loading model: {e}")
        traceback.print_exc()
        vlm = None

    # ==========================================
    # 4. READ DATA AND RUN INFERENCE
    # ==========================================
    test_df = pd.read_csv(test_csv_path)
    total_images = len(test_df)
    print(f"Found {total_images} test rows in {test_csv_path}")

    predictions = []

    for index, row in test_df.iterrows():
        raw_id = row.get("id", row.get("image_name", f"unknown_{index}"))
        base_id = clean_base_id(raw_id, fallback=f"unknown_{index}")

        image_path = resolve_image_path(images_dir, base_id)

        print(f"\n[{index + 1}/{total_images}] Processing {base_id}...")
        img_start = time.perf_counter()

        if image_path is None:
            final_pred = random_prediction()
            debug_info = f"Image not found in {IMAGE_EXTS}. Random fallback: {final_pred}"
            print(f"  -> WARNING: {debug_info}")
        else:
            try:
                if vlm is None or not getattr(vlm, "available", False):
                    final_pred = random_prediction()
                    debug_info = f"Model unavailable. Random fallback: {final_pred}"
                else:
                    final_pred, debug_info = vlm.answer_image(image_path)

                    # Safety guard: never let invalid labels leak into submission
                    if final_pred not in (1, 2, 3, 4):
                        final_pred = random_prediction()
                        debug_info += f" | Invalid model output -> random fallback: {final_pred}"

                print(f"  - Debug: {debug_info}")

            except Exception as e:
                print(f"  -> ERROR during inference for {base_id}: {e}")
                traceback.print_exc()
                final_pred = random_prediction()
                print(f"  -> Random fallback after error: {final_pred}")

        elapsed = time.perf_counter() - img_start
        option_str = INT_TO_OPTION.get(final_pred, "Option ?")
        print(f"  -> Final Prediction: {final_pred} ({option_str}) | Time: {elapsed:.1f}s")

        predictions.append(
            {
                "id": base_id,
                "image_name": base_id,
                "option": final_pred,
            }
        )

    # ==========================================
    # 5. SAVE SUBMISSION
    # ==========================================
    submission_path = os.path.join(os.getcwd(), "submission.csv")
    submission_df = pd.DataFrame(predictions)

    # Ensure column order is stable
    submission_df = submission_df[["id", "image_name", "option"]]
    submission_df.to_csv(submission_path, index=False)

    print(f"\nInference completed in {time.perf_counter() - overall_start:.1f}s")
    print(f"Successfully saved {len(predictions)} predictions to {submission_path}")


if __name__ == "__main__":
    main()