import argparse
import os
import sys
import time
import traceback
import pandas as pd

# ==========================================
# 1. ENFORCE STRICT OFFLINE MODE
# ==========================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Import our custom robust offline models
from models import (
    VLMAnswerer, 
    TextAnswerer, 
    EasyOCRReader, 
    keyword_rule_fallback, 
    choose_prediction,
    make_preprocessed_copy, 
    INT_TO_OPTION
)

def main():
    parser = argparse.ArgumentParser(description="GNR Project Inference Script")
    parser.add_argument('--test_dir', type=str, required=True, help="Absolute path to the test directory")
    args = parser.parse_args()

    print(f"Starting inference pipeline...")
    print(f"Test directory provided: {args.test_dir}")
    overall_start = time.perf_counter()

    # ==========================================
    # 2. CONSTRUCT PATHS
    # ==========================================
    test_csv_path = os.path.join(args.test_dir, 'test.csv')
    images_dir = os.path.join(args.test_dir, 'images')
    weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "weights"))

    os.environ["GNR_VLM_MODEL_DIR"] = os.path.join(weights_dir, "qwen_vl")
    os.environ["GNR_TEXT_MODEL_DIR"] = os.path.join(weights_dir, "qwen_text")
    os.environ["GNR_EASYOCR_DIR"] = os.path.join(weights_dir, "easyocr")

    if not os.path.exists(test_csv_path):
        print(f"CRITICAL ERROR: test.csv not found at {test_csv_path}")
        sys.exit(1)

    # ==========================================
    # 3. INITIALIZE MODELS (OFFLINE)
    # ==========================================
    print("Loading models strictly from local weights directory...")
    load_start = time.perf_counter()
    
    try:
        vlm = VLMAnswerer()
        text = TextAnswerer()
        ocr = EasyOCRReader()
        
        print(f"Models loaded in {time.perf_counter() - load_start:.1f}s")
        print(f"VLM Status: {vlm.available} | Text Status: {text.available} | OCR Status: {ocr.available}")
    except Exception as e:
        print(f"CRITICAL ERROR loading models: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ==========================================
    # 4. READ DATA AND RUN INFERENCE
    # ==========================================
    test_df = pd.read_csv(test_csv_path)
    total_images = len(test_df)
    print(f"Found {total_images} test rows in {test_csv_path}")

    predictions = []

    for index, row in test_df.iterrows():
        image_name = row['image_name']
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path) and os.path.exists(image_path + ".png"):
            image_path = image_path + ".png"

        print(f"\n[{index + 1}/{total_images}] Processing {image_name}...")
        img_start = time.perf_counter()
        final_pred = 5  # Fallback skip value

        if not os.path.exists(image_path):
            print(f"  -> ERROR: Image {image_name} not found. Skipping.")
        else:
            try:
                print("  - Preprocessing...")
                pre_path = make_preprocessed_copy(image_path)
                
                print("  - Running OCR...")
                ocr_text = ocr.read(pre_path)
                
                print("  - Rule Check...")
                rule_pred = keyword_rule_fallback(ocr_text)
                
                print("  - Running VLM...")
                vlm_pred, _ = vlm.answer_image(image_path, ocr_text)
                
                print("  - Running Text Model Fallback...")
                text_pred, _ = text.answer_text(ocr_text)
                
                final_pred = choose_prediction(vlm_pred, text_pred, rule_pred)
                
                # Cleanup temp file
                if os.path.exists(pre_path):
                    os.remove(pre_path)
                    
            except Exception as e:
                print(f"  -> ERROR during inference for {image_name}: {e}")
                traceback.print_exc()
                final_pred = 5

        elapsed = time.perf_counter() - img_start
        option_str = INT_TO_OPTION.get(final_pred, "Skip/Unanswered")
        print(f"  -> Final Prediction: {final_pred} ({option_str}) | Time: {elapsed:.1f}s")
        
        predictions.append({
            'id': image_name, 
            'image_name': image_name, 
            'option': final_pred
        })

    # ==========================================
    # 5. SAVE SUBMISSION
    # ==========================================
    submission_path = os.path.join(os.getcwd(), 'submission.csv')
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\nInference completed in {time.perf_counter() - overall_start:.1f}s")
    print(f"Successfully saved {len(predictions)} predictions to {submission_path}")

if __name__ == '__main__':
    main()