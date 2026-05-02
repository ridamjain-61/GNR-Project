import os
import torch
import re
from collections import Counter
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Mapping dictionary for final output
INT_TO_OPTION = {0: "Option 1", 1: "Option 2", 2: "Option 3", 3: "Option 4", 5: "Skip/Unanswered"}

def extract_mcq_answer(text):
    """
    Highly robust regex extractor. Hunts for the anchored 'Final Answer' 
    but falls back to aggressive pattern matching if the model disobeys.
    """
    text = text.lower()
    
    # 1st Pass: Look for our explicit requested format (e.g., "final answer: option 2")
    # This regex looks for "final answer" followed by anything, then "option" or just a number/letter
    match = re.search(r'final answer.*?([1-4a-d])', text)
    if match:
        val = match.group(1)
        if val in ['1', 'a']: return 0
        if val in ['2', 'b']: return 1
        if val in ['3', 'c']: return 2
        if val in ['4', 'd']: return 3

    # 2nd Pass: Look for standard concluding phrases
    # (e.g., "the correct option is 3", "answer is option c")
    fallback_match = re.search(r'(?:correct option|answer is|therefore).*?([1-4a-d])\b', text)
    if fallback_match:
        val = fallback_match.group(1)
        if val in ['1', 'a']: return 0
        if val in ['2', 'b']: return 1
        if val in ['3', 'c']: return 2
        if val in ['4', 'd']: return 3

    # 3rd Pass: Absolute last resort. Find the VERY LAST option mentioned in the text.
    # LLMs usually state the correct answer at the very end of their reasoning.
    last_resort = re.findall(r'option\s*([1-4a-d])', text)
    if last_resort:
        val = last_resort[-1] # Grab the last one mentioned
        if val in ['1', 'a']: return 0
        if val in ['2', 'b']: return 1
        if val in ['3', 'c']: return 2
        if val in ['4', 'd']: return 3

    # If it absolutely cannot be parsed, protect the score with 5
    return 5

class VLMAnswerer:
    def __init__(self):
        model_dir = os.environ.get("GNR_VLM_MODEL_DIR", "./weights/qwen_vl")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
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
        
        # UPGRADED PROMPT: Chain of Thought + Strict Anchoring
        prompt = (
            "This is a multiple-choice question about deep learning concepts, math, or architectures. "
            "Analyze the image carefully. Think step-by-step about the formulas, code, or diagrams shown. "
            "After your reasoning, you MUST conclude your response with the exact phrase: 'Final Answer: Option X' "
            "(where X is 1, 2, 3, or 4)."
        )

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

            # Generate 3 samples at temperature 0.6 for structural consistency + creativity
            for i in range(3):
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=512,  # INCREASED TO ALLOW THINKING
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9
                    )
                    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                    output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                    
                    pred = extract_mcq_answer(output_text)
                    predictions.append(pred)
                    raw_outputs.append(output_text)

            # MAJORITY VOTE LOGIC UPGRADE
            valid_votes = [p for p in predictions if p != 5]
            
            if not valid_votes:
                # Total failure to extract anything useful
                final_pred = 5
            else:
                vote_counts = Counter(valid_votes)
                most_common = vote_counts.most_common()
                
                # If there's a tie (e.g., [1, 2]), most_common[0] just picks the first one it saw.
                # Since expected value of guessing between 2 options is positive (+1.0 * 0.5 + -0.25 * 0.5 = +0.375), 
                # we will happily accept the tie-breaker guess rather than skipping.
                final_pred = most_common[0][0]

            # We format the debug info so you can actually read the model's thoughts if you test locally
            debug_info = f"Votes: {predictions}\n--- Thought 1 ---\n{raw_outputs[0][:200]}...\n-----------------"
            return final_pred, debug_info
            
        except Exception as e:
            print(f"VLM Inference error: {e}")
            return 5, f"Error: {e}"