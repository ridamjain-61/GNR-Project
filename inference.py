import argparse
import os
import sys
import time
import traceback
import pandas as pd

# ==========================================
# 1. ENFORCE STRICT OFFLINE MODE
# ==========================================
# These environment variables prevent Hugging Face and EasyOCR 
# from attempting to connect to the internet.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# ==========================================
# 2. IMPORT YOUR CUSTOM LOGIC HERE
# ==========================================
# If your VLMAnswerer, TextAnswerer, and EasyOCRReader classes are in a 
# separate file (e.g., src/models.py), import them here. 
# Otherwise, paste their class definitions directly above the main() function.
#
# from src.models import VLMAnswerer, TextAnswerer, EasyOCRReader, keyword_rule_fallback, choose_prediction
# make_preprocessed_copy, INT_TO_OPTION

def main():
    parser = argparse.ArgumentParser(description="GNR Project Inference Script")
    # The grading script passes the absolute path to the test dataset
    parser.add_argument('--test_dir', type=str, required=True, help="Absolute path to the test directory")
    args = parser.parse_args()

    print(f"Starting inference pipeline...")
    print(f"Test directory provided: {args.test_dir}")
    overall_start = time.perf_counter()

    # ==========================================
    # 3. CONSTRUCT PATHS
    # ==========================================
    test_csv_path = os.path.join(args.test_dir, 'test.csv')
    images_dir = os.path.join(args.test_dir, 'images')
    
    # This points to the folder created by setup.bash / download_weights.py
    weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "weights"))

    # Set environment variables so your custom classes know where to look for offline weights
    os.environ["GNR_VLM_MODEL_DIR"] = os.path.join(weights_dir, "qwen_vl")
    os.environ["GNR_TEXT_MODEL_DIR"] = os.path.join(weights_dir, "qwen_text")
    os.environ["GNR_EASYOCR_DIR"] = os.path.join(weights_dir, "easyocr")

    # Verify test.csv exists
    if not os.path.exists(test_csv_path):
        print(f"CRITICAL ERROR: test.csv not found at {test_csv_path}")
        sys.exit(1)

    # ==========================================
    # 4. INITIALIZE MODELS (OFFLINE)
    # ==========================================
    print("Loading models strictly from local weights directory...")
    load_start = time.perf_counter()
    
    try:
        # Initialize your classes here. Ensure they are coded to read from the 
        # local paths specified in the os.environ variables above.
        
        # vlm = VLMAnswerer()
        # text = TextAnswerer()
        # ocr = EasyOCRReader()
        
        print("Models loaded in {:.1f}s".format(time.perf_counter() - load_start))
    except Exception as e:
        print(f"CRITICAL ERROR loading models: {e}")
        traceback.print_exc()
        sys.exit(1) # If models fail to load, we can't do anything.

    # ==========================================
    # 5. READ DATA AND RUN INFERENCE
    # ==========================================
    test_df = pd.read_csv(test_csv_path)
    total_images = len(test_df)
    print(f"Found {total_images} test rows in {test_csv_path}")

    predictions = []

    for index, row in test_df.iterrows():
        image_name = row['image_name']
        image_path = os.path.join(images_dir, image_name)
        
        print(f"[{index + 1}/{total_images}] Processing {image_name}...")
        img_start = time.perf_counter()

        # Fallback prediction is 5 (unanswered/skip) to avoid negative marking if something breaks
        final_pred = 5 

        if not os.path.exists(image_path):
            print(f"  -> ERROR: Image {image_name} not found at {image_path}. Skipping.")
        else:
            try:
                # --- YOUR INFERENCE PIPELINE GOES HERE ---
                # pre_path = make_preprocessed_copy(image_path)
                # ocr_text = ocr.read(pre_path)
                # rule_pred = keyword_rule_fallback(ocr_text)
                # vlm_pred, _ = vlm.answer_image(image_path, ocr_text)
                # text_pred, _ = text.answer_text(ocr_text)
                # final_pred = choose_prediction(vlm_pred, text_pred, rule_pred)
                
                # [REMOVE THIS LINE ONCE YOUR LOGIC IS IN PLACE]
                final_pred = 0 
                
            except Exception as e:
                # Safely catch any per-image errors so the whole script doesn't crash
                print(f"  -> ERROR during inference for {image_name}: {e}")
                traceback.print_exc()
                final_pred = 5

        elapsed = time.perf_counter() - img_start
        print(f"  -> Prediction: {final_pred} | Time: {elapsed:.1f}s")
        
        # Append strictly in the format requested: id, image_name, option
        predictions.append({
            'id': image_name, 
            'image_name': image_name, 
            'option': final_pred
        })

    # ==========================================
    # 6. SAVE SUBMISSION TO CURRENT DIRECTORY
    # ==========================================
    # The grading script requires submission.csv to be in the root of your repo folder, 
    # NOT in the test_dir.
    submission_path = os.path.join(os.getcwd(), 'submission.csv')
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\nInference completed in {:.1f}s".format(time.perf_counter() - overall_start))
    print(f"Successfully saved {len(predictions)} predictions to {submission_path}")

if __name__ == '__main__':
    main()