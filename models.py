import os
import re
import random
import unicodedata
from collections import Counter, defaultdict
from typing import Optional, Tuple, List

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Mapping dictionary for final output (5 is kept in dict for safety/debugging)
INT_TO_OPTION = {
    1: "Option 1",
    2: "Option 2",
    3: "Option 3",
    4: "Option 4",
    5: "Skip/Unanswered",
}

# High-confidence first. The extractor will choose the highest-score match;
# if tied, the rightmost/highest-position match wins.
PATTERN_SPECS = [
    # XML / markup tags
    (100, r"<\s*(?:answer|final_answer|final|ans|choice)\s*>\s*([1-4a-d]|one|two|three|four)\s*<\s*/\s*(?:answer|final_answer|final|ans|choice)\s*>", re.IGNORECASE),
    (98,  r"<\s*(?:answer|final_answer|final|ans|choice)\s*>(?:\s*[^<]{0,40}?)\s*([1-4a-d]|one|two|three|four)\s*(?:[^<]{0,40}?)<\s*/\s*(?:answer|final_answer|final|ans|choice)\s*>", re.IGNORECASE),

    # Explicit answer declarations
    (95, r"\b(?:final\s+answer|correct\s+answer|answer|chosen\s+answer|selected\s+answer)\b\s*(?:is|=|:|-|->)?\s*([1-4a-d]|one|two|three|four)\b", re.IGNORECASE),
    (94, r"\b(?:option|choice)\b\s*(?:is|=|:|-|->)?\s*([1-4a-d]|one|two|three|four)\b", re.IGNORECASE),
    (92, r"\b(?:therefore|thus|hence|so)\b[^\n\.]{0,80}?\b([1-4a-d]|one|two|three|four)\b", re.IGNORECASE),

    # Short end-of-text declarations
    (85, r"(?:^|\n|\.|!|\?)\s*(?:answer|final answer|option|choice)\s*[:=\-]?\s*([1-4a-d]|one|two|three|four)\b\s*$", re.IGNORECASE),
    (80, r"\b([1-4a-d]|one|two|three|four)\s*[)\].,;:!?]\s*$", re.IGNORECASE),
    (78, r"[\(\[]\s*([1-4a-d]|one|two|three|four)\s*[\)\]]\s*$", re.IGNORECASE),

    # Qwen-ish / common model formats
    (72, r"\b(?:A|B|C|D|1|2|3|4)\b", re.IGNORECASE),
]

WORD_TO_INT = {
    "1": 1, "a": 1, "one": 1,
    "2": 2, "b": 2, "two": 2,
    "3": 3, "c": 3, "three": 3,
    "4": 4, "d": 4, "four": 4,
}


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    # Normalize common smart punctuation
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


def _strip_markup_noise(text: str) -> str:
    """Remove code fences and collapse obvious formatting noise."""
    text = re.sub(r"```(?:\w+)?", " ", text)
    text = text.replace("```", " ")
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _map_to_int(token: str) -> Optional[int]:
    if token is None:
        return None
    t = _normalize_text(token).strip().lower()
    t = re.sub(r"[^1-4a-d]", "", t)
    return WORD_TO_INT.get(t)


def extract_mcq_answer(text: str) -> int:
    """
    Robust MCQ extractor.

    Strategy:
      1) Prefer explicit tags such as <answer>...</answer>.
      2) Prefer explicit answer statements such as "final answer: C".
      3) Prefer tail-end declarations.
      4) Fall back to the rightmost standalone candidate in a short tail window.
      5) Return 5 only when nothing can be parsed.
    """
    cleaned = _strip_markup_noise(_normalize_text(text).lower())
    if not cleaned:
        return 5

    candidates: List[Tuple[int, int, int, str]] = []  # (score, position, answer, pattern_name)

    def add_matches(score: int, pattern: str, flags: int, name: str, source_text: str):
        for m in re.finditer(pattern, source_text, flags):
            token = None
            if m.lastindex:
                token = m.group(1)
            else:
                token = m.group(0)
            ans = _map_to_int(token)
            if ans is None:
                continue
            candidates.append((score, m.start(), ans, name))

    # First pass: full text.
    for score, pattern, flags in PATTERN_SPECS[:-3]:
        add_matches(score, pattern, flags, f"full:{score}", cleaned)

    # Second pass: only the tail window for weaker patterns.
    tail_window = cleaned[-350:] if len(cleaned) > 350 else cleaned
    for score, pattern, flags in PATTERN_SPECS[-3:]:
        add_matches(score, pattern, flags, f"tail:{score}", tail_window)

    if not candidates:
        return 5

    # Sort by: score asc, position asc, then pick the last item.
    # This prefers higher-confidence matches; ties go to later mentions.
    candidates.sort(key=lambda x: (x[0], x[1]))
    best_score = candidates[-1][0]
    best = [c for c in candidates if c[0] == best_score]
    best.sort(key=lambda x: x[1])
    return best[-1][2]


class VLMAnswerer:
    def __init__(self):
        model_dir = os.environ.get("GNR_VLM_MODEL_DIR", "./weights/qwen_vl")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available = False
        self.model = None
        self.processor = None

        try:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir,
                local_files_only=True,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )
            self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
            self.available = True
        except Exception as e:
            print(f"VLM failed to load offline: {e}")
            self.available = False

        # Prompt ensemble. Keep the output extremely constrained.
        self.prompt_variants = [
            (
                "This is a multiple-choice question about deep learning concepts, math, or architectures. "
                "Analyze the image carefully. Think step-by-step about the formulas, code, or diagrams shown. "
                "Inspect the image carefully. Choose exactly one option from 1, 2, 3, or 4. "
                "Do not provide reasoning. Do not add extra words. "
                "Return only this format: <answer>2</answer>."
            )
        ]

    def _generate_once(self, img_path: str, prompt: str, temperature: float = 0.2) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move tensors safely.
        if self.device == "cuda":
            inputs = inputs.to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.05,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text

    def answer_image(self, img_path: str, n_samples: int = 5):
        """
        Returns:
            final_pred: int in {1,2,3,4}; random guess if everything fails.
            debug_info: compact trace of prompts, raw outputs, and votes.
        """
        if not self.available:
            guess = random.randint(1, 4)
            return guess, "Model offline -> random guess"

        raw_outputs = []
        parsed_preds = []
        parsed_sources = []

        # Mix prompt variants and temperatures for self-consistency.
        temps = [0.15, 0.25, 0.45]
        for i in range(n_samples):
            prompt = self.prompt_variants[i % len(self.prompt_variants)]
            temp = temps[i % len(temps)]
            try:
                output_text = self._generate_once(img_path, prompt, temperature=temp)
                pred = extract_mcq_answer(output_text)
            except Exception as e:
                output_text = f"<generation_error>{e}</generation_error>"
                pred = 5

            raw_outputs.append(output_text)
            parsed_preds.append(pred)
            parsed_sources.append((i + 1, prompt[:60] + "...", temp))

        valid_votes = [p for p in parsed_preds if p in (1, 2, 3, 4)]
        if not valid_votes:
            final_pred = random.randint(1, 4)
            debug_info = (
                f"Votes: {parsed_preds} -> all unparseable. "
                f"Random guess: {final_pred}\n"
                f"Raw1: {raw_outputs[0][:250]}"
            )
            return final_pred, debug_info

        vote_counts = Counter(valid_votes)
        top_count = max(vote_counts.values())
        tied = [k for k, v in vote_counts.items() if v == top_count]

        if len(tied) == 1:
            final_pred = tied[0]
        else:
            # Tie-break: prefer the prediction from the strongest parsed output.
            score_rank = {1: 0, 2: 0, 3: 0, 4: 0}
            for pred in valid_votes:
                score_rank[pred] += 1
            # Deterministic tie-break by earliest occurrence among tied options.
            for pred in parsed_preds:
                if pred in tied:
                    final_pred = pred
                    break
            else:
                final_pred = random.choice(tied)

        debug_info = (
            f"Votes: {parsed_preds} | counts={dict(vote_counts)} | final={final_pred}\n"
            f"Raw1: {raw_outputs[0][:220]}\n"
            f"Raw2: {raw_outputs[1][:220] if len(raw_outputs) > 1 else ''}\n"
            f"Raw3: {raw_outputs[2][:220] if len(raw_outputs) > 2 else ''}"
        )
        return final_pred, debug_info
