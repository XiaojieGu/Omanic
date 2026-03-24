#!/usr/bin/env python3
"""Evaluate OmanicBench with either an in-memory merged LoRA model or a full fine-tuned model."""

import argparse
import gc
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

BASE_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_LORA_PATH = "LlamaFactory/saves/llama3.3-70b/lora"
DEFAULT_MODEL_PATH = None
DEFAULT_INPUT = "data/OmanicBench.jsonl"
DEFAULT_MODE = "direct"


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def is_qwen3(name_or_path: str | None) -> bool:
    return bool(name_or_path) and "qwen3" in name_or_path.lower()


def build_model_tag(base_model_name: str, model_path: str | None, lora_path: str | None) -> str:
    if model_path:
        raw = model_path
        if "saves/" in raw:
            raw = raw.split("saves/", 1)[1]
        raw = raw.strip("/")
        return raw.replace("/", "_").replace(".", "-").lower()

    raw = base_model_name.replace("/", "_").replace(".", "-").lower()
    if lora_path:
        return raw
    return raw


def resolve_default_paths(base_model_name: str, model_path: str | None, lora_path: str | None, mode: str) -> tuple[str, str]:
    tag = build_model_tag(base_model_name, model_path, lora_path)
    output = f"eval/results/{tag}_{mode}_eval_results.jsonl"
    summary = f"eval/results/{tag}_{mode}_eval_summary.json"
    return output, summary


def apply_template(tokenizer, messages: List[Dict[str, str]], model_ref: str, add_generation_prompt: bool = True) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    if is_qwen3(model_ref):
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def load_data(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {path}")
    return data


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not gold_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def get_mc_options(sample: Dict[str, Any]) -> Tuple[Dict[str, str], str]:
    options = sample.get("multiple_choice_options") or {}
    label = str(sample.get("correct_answer_label", "")).strip().upper()
    if options and label:
        return options, label

    hops = sample.get("repaired_single_hop") or sample.get("single_hop") or []
    if hops:
        last_hop = hops[-1]
        options = last_hop.get("multiple_choice_options") or {}
        label = str(last_hop.get("correct_answer_label", "")).strip().upper()
    return options, label


def build_mc_prompt(sample: Dict[str, Any], mode: str = DEFAULT_MODE) -> str:
    question = sample["multi_hop_question"]
    options, _ = get_mc_options(sample)

    if mode == "cot":
        instruction = 'Think step by step and end the response with "The answer is X", where X is A, B, C, or D.'
    else:
        instruction = 'Select the correct option from four candidates and return only the answer letter (A/B/C/D).'

    prompt = f"{instruction}\n\nQuestion: {question}\n\n"
    for label in sorted(options.keys()):
        prompt += f"{label}. {options[label]}\n"
    prompt += "\nResponse:" if mode == "cot" else "\nAnswer:"
    return prompt


def build_qa_prompt(sample: Dict[str, Any], force_answer: bool = False, mode: str = DEFAULT_MODE) -> str:
    question = sample["multi_hop_question"]
    if mode == "cot":
        instruction = (
            'Think step by step and write the final answer on a separate line in the format "FINAL ANSWER: <answer>".'
        )
        if force_answer:
            instruction += " Even if the question seems flawed, give your best answer and do not refuse."
        return f"{instruction}\n\nQuestion: {question}\n\nResponse:"

    instruction = 'Answer as concisely as possible and provide only the final answer without explanation.'
    if force_answer:
        instruction += " Even if the question seems flawed, give your best short answer and do not refuse."
    return f"{instruction}\n\nQuestion: {question}\n\nAnswer:"


REFUSAL_PATTERNS = [
    r"(?i)unanswerable",
    r"(?i)cannot\s+(be\s+)?(determine|answer|verify|confirm)",
    r"(?i)can'?t\s+(be\s+)?(determine|answer|verify|confirm)",
    r"(?i)not\s+enough\s+information",
    r"(?i)insufficient\s+information",
    r"(?i)unable\s+to",
    r"(?i)impossible\s+to\s+(determine|answer)",
    r"(?i)false\s+premise",
    r"(?i)incorrect\s+(premise|assumption)",
    r"(?i)does\s+not\s+exist",
    r"(?i)doesn'?t\s+exist",
    r"(?i)never\s+existed",
]


def is_refusal(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in REFUSAL_PATTERNS)


def extract_final_answer_for_cot(text: str) -> str:
    text = text.strip()
    patterns = [
        r"(?is)final\s*answer\s*[:：]\s*(.+)$",
        r"(?is)the\s+answer\s+is\s+(.+)$",
        r"(?is)answer\s*[:：]\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            ans = match.group(1).strip().splitlines()[0].strip()
            if ans:
                return ans
    return text.strip()


def extract_short_answer(text: str) -> str:
    text = text.strip()
    if len(text.split()) <= 10:
        return text

    m = re.search(r"(?i)(?:the\s+)?answer\s+(?:is|would\s+be|could\s+be)\s*[:\-]?\s*(.+?)(?:\.|$)", text)
    if m:
        ans = m.group(1).strip().rstrip(".")
        if ans and len(ans.split()) <= 15:
            return ans

    m = re.search(r"(?i)\bit\s+(?:is|was|would\s+be)\s+(.+?)(?:\.|,|;|$)", text)
    if m:
        ans = m.group(1).strip().rstrip(".")
        if ans and len(ans.split()) <= 10:
            return ans

    if ";" in text:
        last_part = text.split(";")[-1].strip()
        if last_part and len(last_part.split()) <= 15 and not is_refusal(last_part):
            return last_part

    sentences = re.split(r"[.!?]\s+", text)
    if sentences:
        first = sentences[0].strip().rstrip(".")
        if first and not is_refusal(first) and len(first.split()) <= 15:
            return first
    return text


def needs_answer_extraction(text: str) -> bool:
    text = text.strip()
    return len(text.split()) > 15 or is_refusal(text)


def extract_mc_answer(text: str) -> str:
    text = text.strip()
    if text.upper() in {"A", "B", "C", "D"}:
        return text.upper()
    patterns = [
        r"(?:the\s+)?answer\s*(?:is|:)\s*\(?([A-Da-d])\)?",
        r"^\s*\(?([A-Da-d])\)?[\s\.\,\:]",
        r"\b([A-Da-d])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    for char in text:
        if char.upper() in {"A", "B", "C", "D"}:
            return char.upper()
    return ""


class MCChoiceLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: List[int]):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0.0
        return scores + mask


def get_abcd_token_ids(tokenizer) -> List[int]:
    candidates = ["A", "B", "C", "D", "a", "b", "c", "d", " A", " B", " C", " D", " a", " b", " c", " d"]
    token_ids = set()
    for c in candidates:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) == 1:
            token_ids.add(ids[0])
    print(f"Allowed ABCD token ids: {sorted(token_ids)}")
    return list(token_ids)


def load_model_and_tokenizer(base_model_name: str, lora_path: str | None, model_path: str | None):
    model_ref = model_path or base_model_name
    tokenizer_path = model_path or base_model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_path:
        print("=" * 60)
        print(f"Loading full model from: {model_path}")
        print("=" * 60)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        return model, tokenizer, model_ref

    if not lora_path:
        raise ValueError("Either --lora-path or --model-path must be provided.")
    if not Path(lora_path).exists():
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")

    print("=" * 60)
    print(f"Loading base model: {base_model_name}")
    print(f"Merging LoRA in memory from: {lora_path}")
    print("=" * 60)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer, model_ref


def batch_generate(model, tokenizer, prompts: List[str], batch_size: int, max_new_tokens: int, desc: str, logits_processor=None) -> List[str]:
    results: List[str] = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    print(f"Running inference for {len(prompts)} samples (batch_size={batch_size}, {num_batches} batches)")

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None)
    if logits_processor is not None:
        gen_kwargs["logits_processor"] = logits_processor

    for batch_idx in tqdm(range(num_batches), desc=desc):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        try:
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            prompt_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[:, prompt_length:]
            batch_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            batch_results = [t.strip() for t in batch_texts]
        except Exception as exc:
            print(f"[Warning] Batch {batch_idx} failed ({exc}), falling back to per-sample generation.")
            torch.cuda.empty_cache()
            batch_results = []
            for prompt in batch_prompts:
                try:
                    single_input = tokenizer(
                        [prompt],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=4096,
                    ).to(model.device)
                    with torch.no_grad():
                        single_output = model.generate(**single_input, **gen_kwargs)
                    input_length = single_input["input_ids"].shape[1]
                    text = tokenizer.decode(single_output[0][input_length:], skip_special_tokens=True)
                    batch_results.append(text.strip())
                except Exception as exc2:
                    print(f"[Warning] Single-sample generation failed ({exc2}), using empty output.")
                    batch_results.append("")
                finally:
                    torch.cuda.empty_cache()
        results.extend(batch_results)

    if len(results) != len(prompts):
        print(f"[Warning] Output count ({len(results)}) does not match input count ({len(prompts)}), padding with empty strings.")
        while len(results) < len(prompts):
            results.append("")
    return results


def evaluate(base_model_name: str, lora_path: str | None, model_path: str | None, input_path: str, output_path: str, summary_path: str, batch_size: int, mode: str) -> None:
    ensure_parent(output_path)
    ensure_parent(summary_path)
    samples = load_data(input_path)
    model, tokenizer, model_ref = load_model_and_tokenizer(base_model_name, lora_path, model_path)

    mc_prompts = []
    qa_prompts = []
    for sample in samples:
        mc_prompt = apply_template(tokenizer, [{"role": "user", "content": build_mc_prompt(sample, mode=mode)}], model_ref)
        qa_prompt = apply_template(tokenizer, [{"role": "user", "content": build_qa_prompt(sample, mode=mode)}], model_ref)
        mc_prompts.append(mc_prompt)
        qa_prompts.append(qa_prompt)

    abcd_ids = get_abcd_token_ids(tokenizer)
    mc_logits_processor = None if mode == "cot" else LogitsProcessorList([MCChoiceLogitsProcessor(abcd_ids)])
    mc_max_new_tokens = 32 if mode == "cot" else 1

    mc_raw_outputs = batch_generate(model, tokenizer, mc_prompts, batch_size=batch_size, max_new_tokens=mc_max_new_tokens, desc=f"MC inference ({mode})", logits_processor=mc_logits_processor)
    qa_raw_outputs = batch_generate(model, tokenizer, qa_prompts, batch_size=batch_size, max_new_tokens=256 if mode == "cot" else 128, desc=f"QA inference ({mode})")

    refusal_indices = [i for i, out in enumerate(qa_raw_outputs) if is_refusal(out)]
    if refusal_indices:
        retry_prompts = []
        for idx in refusal_indices:
            retry_prompt = apply_template(
                tokenizer,
                [{"role": "user", "content": build_qa_prompt(samples[idx], force_answer=True, mode=mode)}],
                model_ref,
            )
            retry_prompts.append(retry_prompt)
        retry_outputs = batch_generate(model, tokenizer, retry_prompts, batch_size=batch_size, max_new_tokens=128 if mode == "cot" else 64, desc=f"QA retry ({mode})")
        for i, idx in enumerate(refusal_indices):
            qa_raw_outputs[idx] = retry_outputs[i]

    for i, text in enumerate(qa_raw_outputs):
        if mode == "cot":
            text = extract_final_answer_for_cot(text)
        if needs_answer_extraction(text):
            text = extract_short_answer(text)
        qa_raw_outputs[i] = text

    mc_correct_count = 0
    total_em = 0.0
    total_f1 = 0.0
    results = []
    for i, sample in enumerate(samples):
        _, correct_label = get_mc_options(sample)
        predicted_label = extract_mc_answer(mc_raw_outputs[i])
        mc_is_correct = predicted_label == correct_label
        if mc_is_correct:
            mc_correct_count += 1

        gold_answer = str(sample.get("multi_hop_answer", ""))
        qa_pred = qa_raw_outputs[i]
        em = compute_exact_match(qa_pred, gold_answer)
        f1 = compute_f1(qa_pred, gold_answer)
        total_em += em
        total_f1 += f1

        results.append(
            {
                "id": sample.get("id", i),
                "multi_hop_question": sample["multi_hop_question"],
                "multi_hop_answer": gold_answer,
                "correct_label": correct_label,
                "predicted_label": predicted_label,
                "mc_raw_output": mc_raw_outputs[i],
                "mc_is_correct": mc_is_correct,
                "qa_raw_output": qa_pred,
                "em": em,
                "f1": f1,
                "mode": mode,
            }
        )

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    n = len(samples)
    summary = {
        "base_model_name": base_model_name,
        "model_path": model_path,
        "lora_path": lora_path,
        "input_path": input_path,
        "output_path": output_path,
        "total_samples": n,
        "mc_correct_count": mc_correct_count,
        "mc_accuracy": mc_correct_count / n * 100 if n else 0.0,
        "exact_match": total_em / n * 100 if n else 0.0,
        "f1": total_f1 / n * 100 if n else 0.0,
        "mode": mode,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Evaluation summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OmanicBench using either in-memory LoRA merge or a full model checkpoint")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL_NAME)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--lora-path", type=str, default=DEFAULT_LORA_PATH)
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--summary-output", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mode", choices=["direct", "cot"], default=DEFAULT_MODE)
    args = parser.parse_args()

    if args.model_path:
        args.lora_path = None

    if args.output is None or args.summary_output is None:
        default_output, default_summary = resolve_default_paths(args.base_model, args.model_path, args.lora_path, args.mode)
        if args.output is None:
            args.output = default_output
        if args.summary_output is None:
            args.summary_output = default_summary

    print(f"Visible GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")

    evaluate(args.base_model, args.lora_path, args.model_path, args.input, args.output, args.summary_output, args.batch_size, args.mode)


if __name__ == "__main__":
    main()
