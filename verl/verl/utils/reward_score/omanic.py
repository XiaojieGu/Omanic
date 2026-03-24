"""
Reward function for multiple-choice and open-ended QA tasks (Qwen3).

Expected model output format:
  <think> reasoning </think> <answer>...</answer>

Two modes, determined by ground_truth:
  - MC mode   : ground_truth is a single letter (A/B/C/D)
                Extracts the option letter and compares exactly.
  - Open mode : ground_truth is answer text
                Extracts <answer> content and compares case-insensitively
                after stripping whitespace/punctuation.

Binary scoring: 1.0 for correct, 0.0 for wrong.
"""

import re
import string

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.DOTALL)
_MC_LETTER_RE = re.compile(r"^[A-D]$", re.I)
_LAST_MC_RE = re.compile(r"\b([A-D])\b")


def _extract_answer_tag(text: str) -> str | None:
    """Return content inside the last <answer>...</answer> tag, or None."""
    matches = _ANSWER_TAG_RE.findall(text)
    return matches[-1].strip() if matches else None


def _normalize(text: str) -> str:
    """Lowercase and strip leading/trailing whitespace and punctuation."""
    return text.lower().strip().strip(string.punctuation).strip()


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> float:
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0]

    gt = ground_truth.strip()

    if _MC_LETTER_RE.match(gt):
        # --- MC mode: ground_truth is A/B/C/D ---
        pred = _extract_answer_tag(solution_str)
        if pred is None:
            # fallback: last standalone A-D letter in output
            matches = _LAST_MC_RE.findall(solution_str)
            pred = matches[-1].upper() if matches else None
        else:
            pred = pred[0].upper() if pred else None

        return 1.0 if (pred is not None and pred == gt.upper()) else 0.0

    else:
        # --- Open-ended mode: ground_truth is answer text ---
        pred = _extract_answer_tag(solution_str)
        if pred is None:
            return 0.0
        return 1.0 if _normalize(pred) == _normalize(gt) else 0.0
