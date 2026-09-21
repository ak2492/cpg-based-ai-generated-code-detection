"""External OOD evaluations — exact notebook logic per language.

Graphs and vocab come from the {language}_cpg_bundle.pt saved by main.py;
external datasets are loaded fresh (SemEval/HMCorp/GPTSniffer) exactly as
notebooks do. Dual-model comparison is preserved here (notebook Cells 7-9
evaluate both models on every external suite).

Sequential usage:
  python main.py --language python
  python train.py --language python && python train.py --language python --adversarial
  python external_eval.py --language python --suite all

SemEval-2026 Task 13:
  python: filter in ['python','py'], no wrapper, cost eval
  java:   filter == 'java', DummyWrapper class, no-cost eval (Cell 6)
  cpp:    filter in ['cpp','c++','cxx','cc'], dummy_wrapper, cost eval
HMCorp:
  python: python_dataset.jsonl, no wrapper, cost eval (Cell 9)
  java:   java_dataset.jsonl, DummyWrapper, cost eval (Cell 8)
  cpp:    not present in notebooks -> reported as unavailable
GPTSniffer (java Cell 9 only):
  0_* = AI(1), 1_* = Human(0), DummyWrapper, 3000/class cap, cost eval

Usage:
  python external_eval.py --language python --suite semeval_A
  python external_eval.py --language java --suite all
"""
import argparse
import gc
import glob
import json
import os
import random
import subprocess

import torch
from torch_geometric.loader import DataLoader

from language_configs import SEED, BPE_VOCAB_SIZE, DEFAULT_BATCH_SIZE, CLEAN_CHECKPOINT, ADV_CHECKPOINT
from model import AdvancedASTGraphEncoder, load_checkpoint
from attack_utils import (
    execute_model_eval,
    execute_model_eval_with_cost,
    print_detailed_metrics,
    print_detailed_metrics_with_cost,
    set_seed,
)
from graph_builder import process_split, apply_normalization
from pipeline import load_bundle


def _load_both(ctx, device, language):
    def _make():
        return AdvancedASTGraphEncoder(num_node_types=ctx.vocab_size,
                                       bpe_vocab_size=BPE_VOCAB_SIZE,
                                       pad_idx=ctx.pad_id).to(device)
    m_clean, m_adv = _make(), _make()
    load_checkpoint(CLEAN_CHECKPOINT[language], m_clean, device)
    load_checkpoint(ADV_CHECKPOINT[language], m_adv, device)
    return m_clean, m_adv


def run_external_semeval_python(bundle, subtask_name, is_multiclass=False, batch_size=None, threshold=0.50):
    from datasets import load_dataset
    from language_configs import MAX_SEMEVAL_SAMPLES_PER_CLASS
    ctx = bundle["ctx"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size or DEFAULT_BATCH_SIZE["python"]
    model_clean, model_adv = _load_both(ctx, device, "python")

    print("\n" + "=" * 85)
    print(f"EXTERNAL EVALUATION: SemEval-2026 Task 13 Python (Subtask {subtask_name})")
    print("=" * 85)

    raw_ds = load_dataset("DaniilOr/SemEval-2026-Task13", subtask_name, split="validation")
    filtered_ds = raw_ds.filter(lambda x: str(x.get('language', '')).strip().lower() in ['python', 'py'])

    human_samples, ai_samples = [], []
    for row in filtered_ds:
        code_text = row.get('code') or row.get('text') or ''
        trimmed = code_text.strip()
        if not trimmed:
            continue

        raw_label = int(row.get('label', 0))
        binary_label = 1 if (raw_label > 0 if is_multiclass else raw_label == 1) else 0

        if binary_label == 0:
            human_samples.append({'code': code_text, 'label': binary_label})
        else:
            ai_samples.append({'code': code_text, 'label': binary_label})

    n_h = min(len(human_samples), MAX_SEMEVAL_SAMPLES_PER_CLASS)
    n_a = min(len(ai_samples), MAX_SEMEVAL_SAMPLES_PER_CLASS)

    if n_h == 0 or n_a == 0:
        print(f"[!] Insufficient Python samples: {n_h} Human, {n_a} AI. Skipping.")
        return

    balanced = human_samples[:n_h] + ai_samples[:n_a]
    random.Random(SEED).shuffle(balanced)

    print(f"Isolated and balanced {len(balanced)} Python samples: {n_h} Human, {n_a} AI.")
    semeval_graphs = process_split(balanced, f"Parsing SemEval Subtask {subtask_name} (Python)", ctx)

    dropped_count = len(balanced) - len(semeval_graphs)
    print(f"[!] Python PARSE DROPS: Attempted {len(balanced)}, Parsed {len(semeval_graphs)}, Dropped {dropped_count}")

    if len(semeval_graphs) == 0:
        print("[!] No Python graphs parsed successfully.")
        return

    apply_normalization(semeval_graphs, ctx)
    loader = DataLoader(semeval_graphs, batch_size=batch_size, shuffle=False)

    res_m1 = execute_model_eval_with_cost(model_clean, loader, device, threshold=threshold)
    res_m2 = execute_model_eval_with_cost(model_adv, loader, device, threshold=threshold)

    print("\n" + "-" * 85)
    print(f"SEMEVAL SUBTASK {subtask_name} PYTHON RESULTS")
    print("-" * 85)
    print_detailed_metrics_with_cost("Model 1 (Clean Baseline)", res_m1)
    print("-" * 85)
    print_detailed_metrics_with_cost("Model 2 (Adversarial GNN)", res_m2)

    del raw_ds, filtered_ds, human_samples, ai_samples, balanced, semeval_graphs, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_external_semeval_java(bundle, subtask_name, is_multiclass=False, batch_size=None, threshold=0.50):
    from datasets import load_dataset
    from language_configs import MAX_SEMEVAL_SAMPLES_PER_CLASS
    ctx = bundle["ctx"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size or DEFAULT_BATCH_SIZE["java"]
    model_clean, model_adv = _load_both(ctx, device, "java")

    print("\n" + "=" * 85)
    print(f"EXTERNAL EVALUATION: SemEval-2026 Task 13 (Subtask {subtask_name})")
    print("=" * 85)

    raw_ds = load_dataset("DaniilOr/SemEval-2026-Task13", subtask_name, split="validation")
    filtered_ds = raw_ds.filter(lambda x: str(x.get('language', '')).strip().lower() == 'java')

    human_samples, ai_samples = [], []
    for row in filtered_ds:
        code_text = row.get('code') or row.get('text') or ''
        trimmed = code_text.strip()
        if not trimmed:
            continue

        # Wrap bare functions in a class structure for Tree-sitter
        if not ("class " in trimmed or "interface " in trimmed or "enum " in trimmed):
            code_text = f"public class DummyWrapper {{\n{code_text}\n}}"

        raw_label = int(row.get('label', 0))
        binary_label = 1 if (raw_label > 0 if is_multiclass else raw_label == 1) else 0

        if binary_label == 0:
            human_samples.append({'code': code_text, 'label': binary_label})
        else:
            ai_samples.append({'code': code_text, 'label': binary_label})

    n_h = min(len(human_samples), MAX_SEMEVAL_SAMPLES_PER_CLASS)
    n_a = min(len(ai_samples), MAX_SEMEVAL_SAMPLES_PER_CLASS)
    balanced = human_samples[:n_h] + ai_samples[:n_a]
    random.Random(SEED).shuffle(balanced)

    print(f"Isolated {len(balanced)} balanced Java samples ({n_h} Human, {n_a} AI).")
    semeval_graphs = process_split(balanced, f"Parsing SemEval Subtask {subtask_name}", ctx)
    apply_normalization(semeval_graphs, ctx)

    loader = DataLoader(semeval_graphs, batch_size=batch_size, shuffle=False)

    res_m1 = execute_model_eval(model_clean, loader, device, threshold=threshold)
    res_m2 = execute_model_eval(model_adv, loader, device, threshold=threshold)

    print("\n" + "-" * 85)
    print(f"SEMEVAL SUBTASK {subtask_name} RESULTS")
    print("-" * 85)
    print_detailed_metrics("Model 1 (Clean Baseline)", res_m1)
    print("-" * 85)
    print_detailed_metrics("Model 2 (Adversarial GNN)", res_m2)

    del raw_ds, filtered_ds, human_samples, ai_samples, balanced, semeval_graphs, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_external_semeval_cpp(bundle, subtask_name, is_multiclass=False, batch_size=None, threshold=0.50):
    from datasets import load_dataset
    from language_configs import MAX_SEMEVAL_SAMPLES_PER_CLASS
    ctx = bundle["ctx"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size or DEFAULT_BATCH_SIZE["cpp"]
    model_clean, model_adv = _load_both(ctx, device, "cpp")

    print("\n" + "=" * 85)
    print(f"EXTERNAL EVALUATION: SemEval-2026 Task 13 C++ (Subtask {subtask_name})")
    print("=" * 85)

    raw_ds = load_dataset("DaniilOr/SemEval-2026-Task13", subtask_name, split="validation")
    # Comprehensive case-insensitive filtering for C++ indicators
    filtered_ds = raw_ds.filter(lambda x: str(x.get('language', '')).strip().lower() in ['cpp', 'c++', 'cxx', 'cc'])

    human_samples, ai_samples = [], []
    for row in filtered_ds:
        code_text = row.get('code') or row.get('text') or ''
        trimmed = code_text.strip()
        if not trimmed:
            continue

        # Wrap free statements without function headers into a valid C++ wrapper
        has_func = any(k in trimmed for k in ['main(', 'void ', 'int ', 'double ', 'float ', 'bool ', 'char ', 'class ', 'struct ', 'template'])
        if not has_func:
            code_text = f"#include <iostream>\nusing namespace std;\nvoid dummy_wrapper() {{\n{code_text}\n}}"

        raw_label = int(row.get('label', 0))
        binary_label = 1 if (raw_label > 0 if is_multiclass else raw_label == 1) else 0

        if binary_label == 0:
            human_samples.append({'code': code_text, 'label': binary_label})
        else:
            ai_samples.append({'code': code_text, 'label': binary_label})

    n_h = min(len(human_samples), MAX_SEMEVAL_SAMPLES_PER_CLASS)
    n_a = min(len(ai_samples), MAX_SEMEVAL_SAMPLES_PER_CLASS)

    if n_h == 0 or n_a == 0:
        print(f"[!] Insufficient C++ samples extracted: {n_h} Human, {n_a} AI. Skipping.")
        return

    balanced = human_samples[:n_h] + ai_samples[:n_a]
    random.Random(SEED).shuffle(balanced)

    print(f"Isolated and balanced {len(balanced)} C++ samples: {n_h} Human, {n_a} AI.")
    semeval_graphs = process_split(balanced, f"Parsing SemEval Subtask {subtask_name} (C++)", ctx)

    dropped_count = len(balanced) - len(semeval_graphs)
    print(f"[!] C++ PARSE DROPS: Attempted {len(balanced)}, Parsed {len(semeval_graphs)}, Dropped {dropped_count}")

    if len(semeval_graphs) == 0:
        print("[!] No C++ graphs parsed successfully.")
        return

    apply_normalization(semeval_graphs, ctx)
    loader = DataLoader(semeval_graphs, batch_size=batch_size, shuffle=False)

    res_m1 = execute_model_eval_with_cost(model_clean, loader, device, threshold=threshold)
    res_m2 = execute_model_eval_with_cost(model_adv, loader, device, threshold=threshold)

    print("\n" + "-" * 85)
    print(f"SEMEVAL SUBTASK {subtask_name} C++ RESULTS")
    print("-" * 85)
    print_detailed_metrics_with_cost("Model 1 (Clean Baseline)", res_m1)
    print("-" * 85)
    print_detailed_metrics_with_cost("Model 2 (Adversarial GNN)", res_m2)

    del raw_ds, filtered_ds, human_samples, ai_samples, balanced, semeval_graphs, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_hmcorp_python(bundle, batch_size=None, threshold=0.50):
    from huggingface_hub import hf_hub_download
    ctx = bundle["ctx"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size or DEFAULT_BATCH_SIZE["python"]
    model_clean, model_adv = _load_both(ctx, device, "python")

    print("\n" + "=" * 85)
    print("EXTERNAL OOD EVALUATION: HMCorp Dataset (Python)")
    print("=" * 85)

    try:
        file_path = hf_hub_download(
            repo_id="OSS-forge/HumanVsAICode",
            filename="python_dataset.jsonl",
            repo_type="dataset"
        )
    except Exception as e:
        print(f"[!] Failed to download HMCorp Python dataset: {e}")
        return

    human_samples, ai_samples = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(human_samples) >= 2000 and len(ai_samples) >= 2000:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            h_code = str(row.get('human_code', '')).strip()
            ai_code = str(row.get('chatgpt_code', '')).strip()

            for code_text, label in [(h_code, 0), (ai_code, 1)]:
                if not code_text or len(code_text) < 10:
                    continue

                if label == 0 and len(human_samples) < 2000:
                    human_samples.append({'code': code_text, 'label': label})
                elif label == 1 and len(ai_samples) < 2000:
                    ai_samples.append({'code': code_text, 'label': label})

    balanced_samples = human_samples + ai_samples
    random.Random(42).shuffle(balanced_samples)

    print(f"Isolated and balanced {len(balanced_samples)} HMCorp Python samples ({len(human_samples)} Human, {len(ai_samples)} AI).")
    hmcorp_graphs = process_split(balanced_samples, "Parsing HMCorp Python", ctx)

    if len(hmcorp_graphs) == 0:
        print("[!] No Python graphs parsed successfully.")
        return

    apply_normalization(hmcorp_graphs, ctx)
    loader = DataLoader(hmcorp_graphs, batch_size=batch_size, shuffle=False)

    print("\nEvaluating Model 1: Clean Baseline Python GNN")
    res_m1 = execute_model_eval_with_cost(model_clean, loader, device, threshold=threshold)
    print_detailed_metrics_with_cost("Model 1 (Clean Baseline)", res_m1)

    print("\nEvaluating Model 2: Adversarial Python GNN")
    res_m2 = execute_model_eval_with_cost(model_adv, loader, device, threshold=threshold)
    print_detailed_metrics_with_cost("Model 2 (Adversarial GNN)", res_m2)

    del human_samples, ai_samples, balanced_samples, hmcorp_graphs, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_hmcorp_java(bundle, batch_size=None, threshold=0.50):
    from huggingface_hub import hf_hub_download
    ctx = bundle["ctx"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size or DEFAULT_BATCH_SIZE["java"]
    model_clean, model_adv = _load_both(ctx, device, "java")

    print("\n" + "=" * 85)
    print("EXTERNAL OOD EVALUATION: HMCorp Dataset (Java)")
    print("=" * 85)

    try:
        file_path = hf_hub_download(
            repo_id="OSS-forge/HumanVsAICode",
            filename="java_dataset.jsonl",
            repo_type="dataset"
        )
    except Exception as e:
        print(f"[!] Failed to download HMCorp Java dataset: {e}")
        return

    human_samples, ai_samples = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(human_samples) >= 2000 and len(ai_samples) >= 2000:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            h_code = str(row.get('human_code', '')).strip()
            ai_code = str(row.get('chatgpt_code', '')).strip()

            for code_text, label in [(h_code, 0), (ai_code, 1)]:
                if not code_text or len(code_text) < 10:
                    continue

                if not ("class " in code_text or "interface " in code_text or "enum " in code_text):
                    code_text = f"public class DummyWrapper {{\n{code_text}\n}}"

                if label == 0 and len(human_samples) < 2000:
                    human_samples.append({'code': code_text, 'label': label})
                elif label == 1 and len(ai_samples) < 2000:
                    ai_samples.append({'code': code_text, 'label': label})

    balanced_samples = human_samples + ai_samples
    random.Random(42).shuffle(balanced_samples)

    print(f"Isolated and balanced {len(balanced_samples)} HMCorp Java samples ({len(human_samples)} Human, {len(ai_samples)} AI).")
    hmcorp_graphs = process_split(balanced_samples, "Parsing HMCorp Java", ctx)

    if len(hmcorp_graphs) == 0:
        print("[!] No Java graphs parsed successfully.")
        return

    apply_normalization(hmcorp_graphs, ctx)
    loader = DataLoader(hmcorp_graphs, batch_size=batch_size, shuffle=False)

    print("\nEvaluating Model 1: Clean Baseline Java GNN")
    res_m1 = execute_model_eval_with_cost(model_clean, loader, device, threshold=threshold)
    print_detailed_metrics_with_cost("Model 1 (Clean Baseline)", res_m1)

    print("\nEvaluating Model 2: Adversarial Java GNN")
    res_m2 = execute_model_eval_with_cost(model_adv, loader, device, threshold=threshold)
    print_detailed_metrics_with_cost("Model 2 (Adversarial GNN)", res_m2)

    del human_samples, ai_samples, balanced_samples, hmcorp_graphs, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_gptsniffer(bundle, batch_size=None, threshold=0.50):
    ctx = bundle["ctx"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size or DEFAULT_BATCH_SIZE["java"]
    model_clean, model_adv = _load_both(ctx, device, "java")

    print("\n" + "=" * 85)
    print("EXTERNAL 2023-ERA EVALUATION: GPTSniffer Dataset (Java - Dual Model)")
    print("=" * 85)

    if not os.path.exists("GPTSniffer"):
        print("Cloning GPTSniffer repository...")
        subprocess.run(["git", "clone", "https://huggingface.co/datasets/mahirlabibdihan/GPTSniffer"], check=True)

    human_samples = []
    ai_samples = []

    files = glob.glob("GPTSniffer/test/*.java") + glob.glob("GPTSniffer/train/*.java")

    for filepath in files:
        filename = os.path.basename(filepath)

        # Corrected label mapping: 0_ = AI (ChatGPT), 1_ = Human
        if filename.startswith("0_"):
            binary_label = 1
        elif filename.startswith("1_"):
            binary_label = 0
        else:
            continue

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code_text = f.read().strip()

        if code_text and not ("class " in code_text or "interface " in code_text or "enum " in code_text):
            code_text = f"public class DummyWrapper {{\n{code_text}\n}}"

        if code_text:
            if binary_label == 0:
                human_samples.append({'code': code_text, 'label': binary_label})
            else:
                ai_samples.append({'code': code_text, 'label': binary_label})

    n_samples = min(len(human_samples), len(ai_samples), 3000)
    balanced_samples = human_samples[:n_samples] + ai_samples[:n_samples]
    random.Random(SEED).shuffle(balanced_samples)

    print(f"Isolated and balanced {len(balanced_samples)} GPTSniffer samples: {n_samples} Human, {n_samples} AI.")

    sniffer_graphs = process_split(balanced_samples, "Parsing GPTSniffer (2023-era Java)", ctx)

    if len(sniffer_graphs) == 0:
        print("[!] No GPTSniffer graphs successfully parsed.")
        return

    apply_normalization(sniffer_graphs, ctx)
    loader = DataLoader(sniffer_graphs, batch_size=batch_size, shuffle=False)

    print("\n--- Evaluating Model 1: Clean Baseline Java GNN ---")
    res_m1 = execute_model_eval_with_cost(model_clean, loader, device, threshold=threshold)
    print_detailed_metrics_with_cost("Model 1 (Clean Baseline)", res_m1)

    print("\n--- Evaluating Model 2: Adversarial Java GNN ---")
    res_m2 = execute_model_eval_with_cost(model_adv, loader, device, threshold=threshold)
    print_detailed_metrics_with_cost("Model 2 (Adversarial GNN)", res_m2)

    del human_samples, ai_samples, balanced_samples, sniffer_graphs, loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--suite", type=str, default="all",
                        choices=["semeval_A", "semeval_B", "hmcorp", "gptsniffer", "all"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--base_seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.base_seed)
    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE[args.language]
    bundle = load_bundle(args.language)

    def _run_semeval(subtask, multiclass):
        if args.language == "python":
            run_external_semeval_python(bundle, subtask, multiclass, args.batch_size, args.threshold)
        elif args.language == "java":
            run_external_semeval_java(bundle, subtask, multiclass, args.batch_size, args.threshold)
        else:
            run_external_semeval_cpp(bundle, subtask, multiclass, args.batch_size, args.threshold)

    if args.suite in ("semeval_A", "all"):
        _run_semeval("A", False)
    if args.suite in ("semeval_B", "all"):
        _run_semeval("B", True)
    if args.suite in ("hmcorp", "all"):
        if args.language == "python":
            evaluate_hmcorp_python(bundle, args.batch_size, args.threshold)
        elif args.language == "java":
            evaluate_hmcorp_java(bundle, args.batch_size, args.threshold)
        else:
            print("[!] HMCorp OOD is not defined for C++ in the notebooks; skipping.")
    if args.suite in ("gptsniffer", "all"):
        if args.language == "java":
            evaluate_gptsniffer(bundle, args.batch_size, args.threshold)
        elif args.suite == "gptsniffer":
            print("[!] GPTSniffer is Java-only in the notebooks; skipping.")
    if args.language == "python" and args.suite == "all":
        print("\n[✓] All Python external benchmarks finished.")
    elif args.language == "cpp" and args.suite == "all":
        print("\n[✓] All C++ external benchmarks finished.")
    elif args.language == "java" and args.suite == "all":
        print("\n[✓] All comparative pipelines completed successfully.")
