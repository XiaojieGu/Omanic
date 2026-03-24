"""
Evaluate multiple models through the OpenRouter API with direct / cot modes.
This script reports multi-hop MC accuracy, QA exact match, QA F1, and average output length.

Usage:
  python eval/open_eval.py --model all --mode direct
  python eval/open_eval.py --model all --mode cot
  python eval/open_eval.py --model qwen/qwen3-8b --mode direct

Environment variables:
  OPENROUTER_API_KEY: OpenRouter API key
"""

import json
import asyncio
import os
import re
import string
import argparse
from collections import Counter
from pathlib import Path

import aiohttp
from aiohttp import ClientSession
from tqdm.asyncio import tqdm

# ======================================================
# CONFIGURATION
# ======================================================

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

INPUT_FILE = "data/OmanicBench.jsonl"
ANALYSIS_DIR = "eval/results"
EVAL_LIMIT = None
MAX_CONCURRENCY = 50

ALL_MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3-8b-instruct",
    "meta-llama/llama-3-70b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen3-32b",
    "qwen/qwen3-8b",
    "qwen/qwen3-max",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek/deepseek-r1-distill-qwen-32b",
    "openai/gpt-5.4",
    "openai/gpt-5.2",
    "openai/gpt-5.1",
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.5",
    "anthropic/claude-opus-4.1",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-opus-4",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
]

MODEL_ALIASES = {
    "GPT-5.4": "openai/gpt-5.4",
    "GPT-5.2": "openai/gpt-5.2",
    "GPT-5.1": "openai/gpt-5.1",
    "GPT-4o": "openai/gpt-4o",
    "Claude-Sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "Claude-Sonnet-4.5": "anthropic/claude-sonnet-4.5",
    "Claude-Opus-4.5": "anthropic/claude-opus-4.5",
    "Claude-Opus-4.1": "anthropic/claude-opus-4.1",
    "Claude-Sonnet-4": "anthropic/claude-sonnet-4",
    "Claude-Opus-4": "anthropic/claude-opus-4",
    "Gemini-3.1-flash-lite": "google/gemini-3.1-flash-lite-preview",
    "Gemini-3-Flash-Preview": "google/gemini-3-flash-preview",
    "Gemini-3.5-Flash-Preview": "google/gemini-3-flash-preview",
    "Gemini-2.5-Flash": "google/gemini-2.5-flash",
    "Gemini-2.5-Flash-Lite": "google/gemini-2.5-flash-lite",
    "Qwen3-Max": "qwen/qwen3-max",
}

REASONING_MODELS = {
    "qwen/qwen3-8b",
    "openai/gpt-5.4",
    "openai/gpt-5.2",
    "openai/gpt-5.1",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.5",
    "anthropic/claude-opus-4.1",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-opus-4",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
}

# ======================================================
# PROMPTS
# ======================================================

MC_PROMPT_DIRECT = """You are answering a multiple-choice question. Please read the question carefully and select the correct answer from the given options.

Question: {question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Please provide ONLY the letter of your answer (A, B, C, or D). Do not include any other text.

Your answer:"""

MC_PROMPT_COT = """You are answering a multiple-choice question. Think step by step, then give your final answer.

Question: {question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Think step by step and end your response with "The answer is X" (where X is A, B, C, or D)."""

QA_PROMPT_DIRECT = """Answer the following question as concisely as possible. Give only the final answer without any explanation or reasoning.

Question: {question}

Answer:"""

QA_PROMPT_COT = """Answer the following question. Think step by step, then give your final answer.

Question: {question}

Think step by step. At the very end, write your final answer on its own line in this exact format:
FINAL ANSWER: <your answer>"""

# ======================================================
# API CALL
# ======================================================

semaphore: asyncio.Semaphore = None


def _needs_reasoning(model: str, mode: str) -> bool:
    return model in REASONING_MODELS and mode == "direct"


def _resolve_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


async def call_api(session: ClientSession, model: str, prompt: str, mode: str):
    max_retries = 8
    last_error = None
    is_reasoning = _needs_reasoning(model, mode)
    max_tokens = 8192 if is_reasoning else (2048 if mode == "cot" else 1024)

    async with semaphore:
        for attempt in range(max_retries):
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }
            if is_reasoning:
                payload["reasoning"] = {"enabled": True}

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
                "HTTP-Referer": "https://reasoning-bench-eval",
                "X-Title": "reasoning-bench-eval",
            }
            try:
                async with session.post(
                    API_URL, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as r:
                    if r.status != 200:
                        error_text = await r.text()
                        last_error = f"HTTP {r.status}: {error_text[:300]}"
                        if r.status == 429 or r.status >= 500:
                            delay = min(60, 2 ** (attempt + 1))
                            if attempt < max_retries - 1:
                                await asyncio.sleep(delay)
                            continue
                        return f"[ERROR: {last_error}]"
                    data = await r.json()
                    if "choices" in data and data["choices"]:
                        content = data["choices"][0].get("message", {}).get("content", "")
                        if content:
                            return _strip_think_tags(content.strip())
                        return "[ERROR: Empty content in choices]"
                    elif "error" in data:
                        last_error = f"API Error: {data['error'].get('message', str(data['error']))}"
                        err_code = data["error"].get("code", "") or data["error"].get("type", "")
                        if "rate_limit" in str(err_code) or "overloaded" in str(err_code):
                            delay = min(60, 2 ** (attempt + 1))
                            if attempt < max_retries - 1:
                                await asyncio.sleep(delay)
                            continue
                        return f"[ERROR: {last_error}]"
                    else:
                        return "[ERROR: Unexpected response format]"
            except asyncio.TimeoutError:
                last_error = "Request timeout"
                delay = min(60, 2 ** (attempt + 1))
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
            except Exception as e:
                last_error = f"Exception: {str(e)[:200]}"
                delay = min(60, 2 ** (attempt + 1))
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
        return f"[ERROR: All retries failed. Last error: {last_error}]" if last_error else "[ERROR: Unknown]"


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()

# ======================================================
# TEXT NORMALIZATION & EM / F1
# ======================================================


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
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


def extract_final_answer(resp: str) -> str:
    for line in reversed(resp.strip().split("\n")):
        line_s = line.strip()
        for prefix in [
            "FINAL ANSWER:", "Final Answer:", "final answer:",
            "The final answer is", "The answer is",
            "**FINAL ANSWER:**", "**Final Answer:**",
        ]:
            if prefix.lower() in line_s.lower():
                idx = line_s.lower().index(prefix.lower()) + len(prefix)
                answer = line_s[idx:].strip().strip("*").strip()
                if answer:
                    return answer
    last_line = resp.strip().split("\n")[-1].strip()
    if len(last_line) < 100:
        return last_line
    return resp

# ======================================================
# ANSWER PARSING
# ======================================================


def parse_answer_direct(response: str, options: dict) -> str | None:
    if not response:
        return None
    response_upper = response.upper().strip()
    for label in "ABCD":
        if response_upper.startswith(label):
            if len(response_upper) == 1 or response_upper[1] in ":.) \n\t":
                return label
    for label in "ABCD":
        patterns = [
            rf"^{label}[:.)]\s*", rf"\s+{label}[:.)]\s*", rf"\({label}\)",
            rf"答案[是：:]\s*{label}", rf"选择[：:]\s*{label}",
        ]
        for pattern in patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return label
    response_lower = response.lower().strip()
    for label, option_text in options.items():
        option_lower = str(option_text).lower().strip()
        if option_lower in response_lower or response_lower in option_lower:
            if option_lower == response_lower or (len(option_lower) > 5 and option_lower in response_lower):
                return label
    match = re.search(r"\b([ABCD])\b", response_upper)
    if match:
        return match.group(1)
    return None


def parse_answer_cot(response: str, options: dict) -> str | None:
    if not response:
        return None
    last_lines = "\n".join(response.strip().split("\n")[-5:])
    m = re.search(r"[Tt]he answer is\s+([ABCD])", last_lines)
    if m:
        return m.group(1).upper()
    return parse_answer_direct(response, options)

# ======================================================
# SINGLE-MODEL EVALUATION
# ======================================================

BATCH_SIZE = 100


async def evaluate_model(model: str, mode: str, lines: list[str]):
    global semaphore
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    model_safe = model.replace("/", "_").replace(".", "_")
    suffix = "_cot_results.jsonl" if mode == "cot" else "_direct_results.jsonl"
    output_file = str(Path(ANALYSIS_DIR) / f"{model_safe}{suffix}")

    reasoning_tag = " [reasoning ON]" if _needs_reasoning(model, mode) else ""
    mode_tag = "COT" if mode == "cot" else "DIRECT"

    print(f"\n{'=' * 60}")
    print(f"Model: {model}  [{mode_tag}]{reasoning_tag}")
    print(f"Samples: {'all' if EVAL_LIMIT is None else EVAL_LIMIT}  Concurrency: {MAX_CONCURRENCY}")
    print(f"{'=' * 60}\n")

    mc_valid = 0
    mc_correct = 0
    mc_failed = 0
    qa_valid = 0
    qa_total_em = 0.0
    qa_total_f1 = 0.0
    total_output_len = 0
    output_count = 0
    results = []
    processed_ids = set()

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENCY)

    def recompute_stats_from_results():
        nonlocal mc_valid, mc_correct, mc_failed
        nonlocal qa_valid, qa_total_em, qa_total_f1
        nonlocal total_output_len, output_count, processed_ids

        mc_valid = 0
        mc_correct = 0
        mc_failed = 0
        qa_valid = 0
        qa_total_em = 0.0
        qa_total_f1 = 0.0
        total_output_len = 0
        output_count = 0
        processed_ids = set()

        for r in results:
            sample_id = str(r.get("id", ""))
            if sample_id:
                processed_ids.add(sample_id)

            is_correct = r.get("is_correct")
            if is_correct is None:
                mc_failed += 1
            else:
                mc_valid += 1
                if is_correct:
                    mc_correct += 1

            if r.get("qa_output") and not str(r.get("qa_output", "")).startswith("[ERROR:"):
                qa_valid += 1
                qa_total_em += float(r.get("em", 0.0))
                qa_total_f1 += float(r.get("f1", 0.0))

            mc_len = int(r.get("mc_output_len", 0) or 0)
            if mc_len > 0:
                total_output_len += mc_len
                output_count += 1

    def flush_results():
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if Path(output_file).exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        recompute_stats_from_results()

    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            parsed = []
            for line_idx, line in enumerate(lines):
                try:
                    data = json.loads(line)
                    sample_id = str(data.get("id", line_idx))
                    if sample_id in processed_ids:
                        continue
                    parsed.append((data, line_idx))
                except json.JSONDecodeError:
                    continue

            async def process_sample(sample_data, idx):
                nonlocal mc_valid, mc_correct, mc_failed
                nonlocal qa_valid, qa_total_em, qa_total_f1
                nonlocal total_output_len, output_count

                sample_id = sample_data.get("id", str(idx))
                multi_hop_question = sample_data.get("multi_hop_question", "")
                gold_answer = sample_data.get("multi_hop_answer", "")
                repaired_hops = sample_data.get("repaired_single_hop") or sample_data.get("single_hop", [])
                if not multi_hop_question or not repaired_hops:
                    mc_failed += 1
                    return

                last_hop = repaired_hops[-1]
                options = last_hop.get("multiple_choice_options", {})
                correct_label = last_hop.get("correct_answer_label", "")
                if not options or not correct_label:
                    mc_failed += 1
                    return

                mc_template = MC_PROMPT_COT if mode == "cot" else MC_PROMPT_DIRECT
                mc_prompt = mc_template.format(
                    question=multi_hop_question,
                    option_a=options.get("A", ""),
                    option_b=options.get("B", ""),
                    option_c=options.get("C", ""),
                    option_d=options.get("D", ""),
                )
                qa_template = QA_PROMPT_COT if mode == "cot" else QA_PROMPT_DIRECT
                qa_prompt = qa_template.format(question=multi_hop_question)

                mc_resp, qa_resp = await asyncio.gather(
                    call_api(session, model, mc_prompt, mode),
                    call_api(session, model, qa_prompt, mode),
                )

                # --- MC ---
                mc_ok = mc_resp and not mc_resp.startswith("[ERROR:")
                predicted = None
                is_correct = None
                if mc_ok:
                    total_output_len += len(mc_resp)
                    output_count += 1
                    parser = parse_answer_cot if mode == "cot" else parse_answer_direct
                    predicted = parser(mc_resp, options)
                    is_correct = (predicted == correct_label) if predicted else False
                    mc_valid += 1
                    if is_correct:
                        mc_correct += 1
                else:
                    mc_failed += 1

                # --- QA (EM / F1) ---
                em, f1 = 0.0, 0.0
                qa_ok = qa_resp and not qa_resp.startswith("[ERROR:")
                if qa_ok and gold_answer:
                    eval_text = extract_final_answer(qa_resp) if mode == "cot" else qa_resp
                    em = compute_exact_match(eval_text, gold_answer)
                    f1 = compute_f1(eval_text, gold_answer)
                    qa_valid += 1
                    qa_total_em += em
                    qa_total_f1 += f1

                results.append({
                    "id": sample_id, "model": model, "mode": mode,
                    "correct_label": correct_label, "predicted": predicted,
                    "is_correct": is_correct, "mc_output": mc_resp,
                    "qa_output": qa_resp, "em": em, "f1": f1,
                    "mc_output_len": len(mc_resp) if mc_ok else 0,
                    "qa_output_len": len(qa_resp) if qa_ok else 0,
                })
                processed_ids.add(str(sample_id))

            pbar = tqdm(total=len(lines), initial=len(processed_ids), desc=f"{model} [{mode_tag}]")
            for batch_start in range(0, len(parsed), BATCH_SIZE):
                batch = parsed[batch_start:batch_start + BATCH_SIZE]
                tasks = [process_sample(d, i) for d, i in batch]
                await asyncio.gather(*tasks)
                pbar.update(len(batch))
                flush_results()
            pbar.close()

    finally:
        flush_results()

        acc = mc_correct / mc_valid * 100 if mc_valid else 0
        avg_len = total_output_len / output_count if output_count else 0
        em_avg = qa_total_em / qa_valid * 100 if qa_valid else 0
        f1_avg = qa_total_f1 / qa_valid * 100 if qa_valid else 0

        print(f"\n{'─' * 60}")
        print(f"  Model: {model}  [{mode_tag}]")
        print(f"  MC  - valid: {mc_valid}  correct: {mc_correct}  failed: {mc_failed}  accuracy: {acc:.2f}%")
        print(f"  QA  - valid: {qa_valid}  EM: {em_avg:.2f}%  F1: {f1_avg:.2f}%")
        print(f"  Average output length (MC): {avg_len:.1f} chars")
        print(f"  Results: {output_file}")
        print(f"{'─' * 60}\n")

# ======================================================
# MAIN
# ======================================================


async def async_main():
    parser = argparse.ArgumentParser(
        description="Evaluate models through the OpenRouter API (direct / cot)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help='Model ID or "all"',
    )
    parser.add_argument(
        "--mode", type=str, choices=["direct", "cot"], required=True,
    )
    parser.add_argument("--input", type=str, default=INPUT_FILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()

    global EVAL_LIMIT, MAX_CONCURRENCY  # noqa: PLW0603
    if args.limit is not None:
        EVAL_LIMIT = args.limit
    if args.concurrency is not None:
        MAX_CONCURRENCY = args.concurrency

    if not API_KEY:
        print("Error: please set the OPENROUTER_API_KEY environment variable")
        return

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: file not found - {args.input}")
        return

    if EVAL_LIMIT is not None:
        lines = lines[:EVAL_LIMIT]
    print(f"Loaded {len(lines)} samples")

    models = ALL_MODELS if args.model == "all" else [_resolve_model_name(args.model)]
    for model in models:
        await evaluate_model(model, args.mode, lines)


if __name__ == "__main__":
    asyncio.run(async_main())
