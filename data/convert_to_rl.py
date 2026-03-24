#!/usr/bin/env python3
"""
Convert Omanic raw JSONL files into verl RL JSON format.

Input files:
  - OmanicSynth.jsonl
  - OmanicBench.jsonl

Output files:
  - OmanicSynth_rl.json
  - OmanicBench_rl.json

Each sample is converted into either:
  - an open-ended QA example, or
  - a multiple-choice example with A/B/C/D options.

The mix is controlled by a fixed random seed so the conversion is reproducible.
The output data_source is set to "omanic" so verl will route rewards to
verl.utils.reward_score.omanic.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OPEN_SYSTEM = (
    "You are a helpful assistant. "
    "Think step by step if needed, then provide only your final answer inside "
    "<answer></answer> tags. "
    "Example: <answer>200</answer>"
)

MC_SYSTEM = (
    "You are a helpful assistant. "
    "Think step by step if needed, then provide only the correct option letter "
    "(A, B, C, or D) inside <answer></answer> tags. "
    "Example: <answer>C</answer>"
)

FILE_MAP = {
    "OmanicSynth.jsonl": "OmanicSynth_rl.json",
    "OmanicBench.jsonl": "OmanicBench_rl.json",
}


def format_mc_options(options: dict) -> str:
    lines = []
    for key in ("A", "B", "C", "D"):
        value = options.get(key)
        if value:
            lines.append(f"{key}. {value}")
    return "\n".join(lines)


def convert_record(obj: dict, as_mc: bool) -> dict:
    question = str(obj["multi_hop_question"]).strip()
    answer = str(obj["multi_hop_answer"]).strip()

    if as_mc:
        options = obj.get("multiple_choice_options") or {}
        correct_label = str(obj.get("correct_answer_label", "")).strip().upper()
        option_block = format_mc_options(options)
        if option_block and correct_label in {"A", "B", "C", "D"}:
            user_content = question + "\n\n" + option_block
            return {
                "prompt": [
                    {"role": "system", "content": MC_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                "data_source": "omanic",
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": correct_label,
                },
            }

    return {
        "prompt": [
            {"role": "system", "content": OPEN_SYSTEM},
            {"role": "user", "content": question},
        ],
        "data_source": "omanic",
        "ability": "reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
    }


def convert_file(input_path: Path, output_path: Path, seed: int = 42) -> tuple[int, int]:
    rng = random.Random(seed)
    records = []

    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(convert_record(obj, as_mc=(rng.random() < 0.5)))

    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(records, fout, ensure_ascii=False, indent=2)

    return len(records), len(records)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    for src_name, dst_name in FILE_MAP.items():
        src = script_dir / src_name
        dst = script_dir / dst_name
        if not src.exists():
            print(f"[{src_name}] not found, skipped.")
            continue
        total, kept = convert_file(src, dst)
        print(f"[{src_name}] {kept}/{total} entries -> {dst}")
    print('Format: mixed open-ended and multiple-choice RL JSON with data_source="omanic".')


if __name__ == "__main__":
    main()
