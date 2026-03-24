"""
Convert OmanicSynth.jsonl and OmanicBench.jsonl into a mixed SFT format
with open-ended QA and multiple-choice examples.
Roughly half of the samples are open-ended and half are 4-option
multiple-choice questions (A/B/C/D).
Outputs: OmanicSynth_sft.json and OmanicBench_sft.json as JSON arrays.
"""

import json
import os
import random

SYSTEM_OPEN = (
    "You are a helpful assistant. Answer the question directly and concisely."
)
SYSTEM_MC = (
    "You are a helpful assistant. Answer the multiple-choice question by selecting "
    "the correct option (A, B, C, or D) and providing the answer."
)
FILE_MAP = {
    "OmanicSynth.jsonl": "OmanicSynth_sft.json",
    "OmanicBench.jsonl": "OmanicBench_sft.json",
}


def format_mc_options(options: dict) -> str:
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()) if v)


def convert_record(obj: dict, as_mc: bool) -> dict:
    question = obj["multi_hop_question"]
    answer = str(obj["multi_hop_answer"])

    if as_mc:
        options = obj.get("multiple_choice_options") or {}
        correct_label = obj.get("correct_answer_label", "A")
        if options:
            option_text = options.get(correct_label, answer)
            return {
                "instruction": question + "\n\n" + format_mc_options(options),
                "input": "",
                "output": f"{correct_label}. {option_text}",
                "system": SYSTEM_MC,
            }

    return {
        "instruction": question,
        "input": "",
        "output": answer,
        "system": SYSTEM_OPEN,
    }


def convert(input_path: str, output_path: str, seed: int = 42) -> int:
    rng = random.Random(seed)
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(convert_record(obj, rng.random() < 0.5))

    rng.shuffle(records)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return len(records)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for src_name, dst_name in FILE_MAP.items():
        src = os.path.join(script_dir, src_name)
        dst = os.path.join(script_dir, dst_name)
        if os.path.exists(src):
            n = convert(src, dst)
            print(f"[{src_name}] {n} samples -> {dst}")
        else:
            print(f"[{src_name}] not found, skipped.")
    print("Format: ~50% open-ended QA, ~50% multiple-choice (A/B/C/D).")


if __name__ == "__main__":
    main()
