import os
import shutil
import torch
import re
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
import easyocr

# Mapping dictionary for final output
INT_TO_OPTION = {0: "Option 1", 1: "Option 2", 2: "Option 3", 3: "Option 4", 5: "Skip/Unanswered"}

def make_preprocessed_copy(image_path):
    """
    Creates a preprocessed copy of the image for better OCR/VLM reading.
    In a real scenario, you might add contrast/grayscale here.
    For safety, we just return the path or make a direct copy to a tmp folder.
    """
    tmp_path = "tmp_" + os.path.basename(image_path)
    # Simple copy for now, replace with cv2 preprocessing if you have it
    shutil.copy(image_path, tmp_path)
    return tmp_path

def keyword_rule_fallback(ocr_text):
    """
    Simple regex/rule-based fallback if models fail.
    Returns 5 (skip) by default unless a very obvious pattern is found.
    """
    # Add your regex logic here if you have specific rules
    return 5 

def extract_mcq_answer(text):
    """
    Safely extracts the predicted option (0, 1, 2, or 3) from the model's text output.
    Returns 5 if it cannot confidently determine the answer.
    """
    text = text.lower()
    # Look for exact option formats or letters A/B/C/D mapped to 0/1/2/3
    if "option 1" in text or "option a" in text or text.startswith("a") or text.startswith("1"): return 0
    if "option 2" in text or "option b" in text or text.startswith("b") or text.startswith("2"): return 1
    if "option 3" in text or "option c" in text or text.startswith("c") or text.startswith("3"): return 2
    if "option 4" in text or "option d" in text or text.startswith("d") or text.startswith("4"): return 3
    return 5

def choose_prediction(vlm_pred, text_pred, rule_pred):
    """
    Ensemble logic: VLM > Text Model > Rule Fallback.
    Returns 5 to avoid negative marking if all fail.
    """
    if vlm_pred in [0, 1, 2, 3]: return vlm_pred
    if text_pred in [0, 1, 2, 3]: return text_pred
    if rule_pred in [0, 1, 2, 3]: return rule_pred
    return 5


class EasyOCRReader:
    def __init__(self):
        model_dir = os.environ.get("GNR_EASYOCR_DIR", "./weights/easyocr")
        try:
            # STRICT OFFLINE MODE
            self.reader = easyocr.Reader(['en'], gpu=True, model_storage_directory=model_dir, download_enabled=False)
            self.available = True
        except Exception as e:
            print(f"EasyOCR failed to load offline: {e}")
            self.available = False

    def read(self, img_path):
        if not self.available: return ""
        try:
            results = self.reader.readtext(img_path, detail=0)
            return " ".join(results)
        except:
            return ""


class VLMAnswerer:
    def __init__(self):
        model_dir = os.environ.get("GNR_VLM_MODEL_DIR", "./weights/qwen_vl")
        try:
            # STRICT OFFLINE MODE + Optimal L40s GPU loading (bfloat16)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir, 
                local_files_only=True, 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
            self.available = True
        except Exception as e:
            print(f"VLM failed to load offline: {e}")
            self.available = False

    def answer_image(self, img_path, ocr_text=""):
        if not self.available: return 5, ""
        
        prompt = "This is a multiple-choice question about deep learning. Analyze the image. "
        if ocr_text:
            prompt += f"Here is the extracted OCR text for context: '{ocr_text}'. "
        prompt += "Which option is correct? Reply ONLY with 'Option 1', 'Option 2', 'Option 3', or 'Option 4'."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            ).to("cuda")

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=50)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            return extract_mcq_answer(output_text), output_text
        except Exception as e:
            print(f"VLM Inference error: {e}")
            return 5, ""


class TextAnswerer:
    def __init__(self):
        model_dir = os.environ.get("GNR_TEXT_MODEL_DIR", "./weights/qwen_text")
        try:
            # STRICT OFFLINE MODE
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, 
                local_files_only=True, 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            self.available = True
        except Exception as e:
            print(f"Text Model failed to load offline: {e}")
            self.available = False

    def answer_text(self, ocr_text):
        if not self.available or not ocr_text.strip(): return 5, ""

        prompt = f"Given the following text from a deep learning MCQ: '{ocr_text}'. Which option is correct? Reply ONLY with 'Option 1', 'Option 2', 'Option 3', or 'Option 4'."
        messages = [{"role": "user", "content": prompt}]

        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to("cuda")

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=50)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                output_text = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

            return extract_mcq_answer(output_text), output_text
        except Exception as e:
            print(f"Text Inference error: {e}")
            return 5, ""