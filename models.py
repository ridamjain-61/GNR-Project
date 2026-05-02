import os
import torch
from collections import Counter
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Mapping dictionary for final output
INT_TO_OPTION = {0: "Option 1", 1: "Option 2", 2: "Option 3", 3: "Option 4", 5: "Skip/Unanswered"}

def extract_mcq_answer(text):
    """
    Safely extracts the predicted option (0, 1, 2, or 3) from the model's text output.
    Returns 5 if it cannot confidently determine the answer.
    """
    text = text.lower()
    if "option 1" in text or "option a" in text or text.startswith("a") or text.startswith("1"): return 0
    if "option 2" in text or "option b" in text or text.startswith("b") or text.startswith("2"): return 1
    if "option 3" in text or "option c" in text or text.startswith("c") or text.startswith("3"): return 2
    if "option 4" in text or "option d" in text or text.startswith("d") or text.startswith("4"): return 3
    return 5

class VLMAnswerer:
    def __init__(self):
        model_dir = os.environ.get("GNR_VLM_MODEL_DIR", "./weights/qwen_vl")
        # Dynamic device detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            # STRICT OFFLINE MODE + Auto device mapping
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir, 
                local_files_only=True, 
                torch_dtype="auto", 
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
            self.available = True
        except Exception as e:
            print(f"VLM failed to load offline: {e}")
            self.available = False

    def answer_image(self, img_path):
        if not self.available: return 5, "Model offline"
        
        prompt = "This is a multiple-choice question about deep learning. Analyze the image carefully. Which option is correct? Reply ONLY with 'Option 1', 'Option 2', 'Option 3', or 'Option 4'."

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
            ).to(self.device)

            predictions = []
            raw_outputs = []

            # SELF-CONSISTENCY: Generate 3 samples at temperature 0.7
            for i in range(3):
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                    output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    
                    pred = extract_mcq_answer(output_text)
                    predictions.append(pred)
                    raw_outputs.append(output_text)

            # MAJORITY VOTE
            # Filter out 5s (unanswered) from the vote, unless all failed
            valid_votes = [p for p in predictions if p != 5]
            
            if not valid_votes:
                final_pred = 5
            else:
                # Get the most common valid prediction
                final_pred = Counter(valid_votes).most_common(1)[0][0]

            debug_info = f"Votes: {predictions} -> Raw: {raw_outputs}"
            return final_pred, debug_info
            
        except Exception as e:
            print(f"VLM Inference error: {e}")
            return 5, f"Error: {e}"