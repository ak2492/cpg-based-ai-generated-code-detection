"""9-suite robustness benchmark — mirrors Cell 5/6 of each notebook.

Evaluates EXACTLY ONE model per run (hybrid style): default runs all 9
suites against the clean checkpoint; --adversarial runs them against the
adv checkpoint. Suite transforms and eval math are notebook-exact; all three
languages report cost metrics (latency/throughput/peak RAM).
Graphs come from the {language}_cpg_bundle.pt saved by main.py; raw test
rows reload cheaply (no graph building here).

Usage:
  python attack_evaluation.py --language python
  python attack_evaluation.py --language java --adversarial
"""
import argparse
import gc

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from language_configs import DEFAULT_BATCH_SIZE, CLEAN_CHECKPOINT, ADV_CHECKPOINT
from model import build_encoder_from_checkpoint
from attack_utils import (
    ATTACK_SUITES,
    attack_graph_cache_path,
    build_or_load_graphs,
    generate_attack_samples,
    execute_model_eval_with_cost,
    print_detailed_metrics_with_cost,
    set_seed,
)
from graph_builder import process_split, apply_normalization
from pipeline import load_bundle, load_test_raw_rows


def _load_model(ctx, device, language, adversarial=False):
    ckpt_file = (ADV_CHECKPOINT if adversarial else CLEAN_CHECKPOINT)[language]
    model, _ = build_encoder_from_checkpoint(ctx, ckpt_file, device)
    return model


def _parse_desc(language, suite_name):
    if language == "python":
        return f"Parsing Python {suite_name}"
    elif language == "cpp":
        return f"Parsing C++ {suite_name}"
    else:
        return f"Parsing {suite_name}"


def _model_tag(language, adversarial):
    if adversarial:
        return "Model 2 (Adversarial GNN)"
    return "Model 1 (Clean Baseline)"


def run_attack_benchmark(language="python", batch_size=None, threshold=0.50,
                         base_seed=42, adversarial=False, target="machine",
                         graphs_cache_dir=None, rebuild_cache=False):
    """Paper Sec 4.7 benchmark: machine-only by default (target='machine').

    graphs_cache_dir enables the parse-once .pt cache (None = parse every
    call, exactly like before); rebuild_cache forces re-parsing.
    """
    set_seed(base_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE[language]

    bundle = load_bundle(language)
    ctx = bundle["ctx"]
    test_graphs = bundle["test_graphs"]
    test_raw = load_test_raw_rows(language)
    parser = bundle["parser"]

    model = _load_model(ctx, device, language, adversarial=adversarial)
    tag = _model_tag(language, adversarial)

    comparison_records = []

    def _cache_path(attack_key, mode):
        # I return None here when caching is off so suites take their
        # original uncached path byte-for-byte.
        if graphs_cache_dir is None:
            return None
        return attack_graph_cache_path(language, attack_key, mode, target,
                                       base_seed, None, graphs_cache_dir)

    for suite_name, attack_key, mode in ATTACK_SUITES:
        if attack_key == "clean":
            eval_graphs = test_graphs
            attack_data = None
        else:
            attack_data = generate_attack_samples(test_raw, attack_key, mode, language, parser,
                                                  base_seed=base_seed, target=target)
            eval_graphs = build_or_load_graphs(
                _cache_path(attack_key, mode), ctx,
                lambda: process_split(attack_data, _parse_desc(language, suite_name), ctx),
                rebuild_cache)

        eval_loader = DataLoader(eval_graphs, batch_size=batch_size, shuffle=False)

        res = execute_model_eval_with_cost(model, eval_loader, device, threshold=threshold)
        comparison_records.append({
            'Scenario': suite_name,
            'Acc': res['Acc'], 'Prec': res['Prec'], 'Rec': res['Rec'],
            'F1': res['F1'], 'ROC': res['ROC'], 'FPR': res['FPR'],
            'Lat': res['Latency_ms'], 'RAM': res['PeakRAM_MB'],
            # I record throughput here because the 5-seed runner needs the
            # same cost columns for every row.
            'Thr': res['Throughput'], 'N': res['N'],
        })

        print("\n" + "=" * 85)
        print(f"BENCHMARK: {suite_name.upper()} (N = {res['N']}) [{tag}]")
        print("=" * 85)
        print_detailed_metrics_with_cost(tag, res)

        if attack_key != "clean":
            del attack_data, eval_graphs
        del eval_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_comp = pd.DataFrame(comparison_records)
    title = {"python": "PYTHON ROBUSTNESS SUMMARY",
             "cpp": "C++ ROBUSTNESS SUMMARY",
             "java": "JAVA ROBUSTNESS SUMMARY"}.get(language, "ROBUSTNESS SUMMARY MATRIX")
    print("\n" + "=" * 85 + f"\n{title} [{tag}]\n" + "=" * 85)
    print(df_comp.to_string(index=False))
    return df_comp


def run_single_attack(language="python", attack_type="auth", mode="enhanced", batch_size=None,
                      threshold=0.50, base_seed=42,
                      target="machine", adversarial=False):
    """Single-layer evaluation of exactly one model.

    Paper Sec 4.7 protocol: machine-only by default (target='machine');
    target='all' attacks every sample. BASIC mode is paper-identical
    (most difficult); graphs are rebuilt from the single-transform code.
    """
    set_seed(base_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE[language]

    bundle = load_bundle(language)
    ctx = bundle["ctx"]
    test_graphs = bundle["test_graphs"]
    test_raw = load_test_raw_rows(language)
    parser = bundle["parser"]

    model = _load_model(ctx, device, language, adversarial=adversarial)

    if attack_type == "clean":
        eval_graphs = test_graphs
    else:
        attack_data = generate_attack_samples(test_raw, attack_type, mode, language, parser,
                                              base_seed=base_seed, target=target)
        eval_graphs = process_split(attack_data, _parse_desc(language, f"{attack_type}-{mode}"), ctx)
        apply_normalization(eval_graphs, ctx)

    loader = DataLoader(eval_graphs, batch_size=batch_size, shuffle=False)
    res = execute_model_eval_with_cost(model, loader, device, threshold=threshold)
    print_detailed_metrics_with_cost(f"{attack_type}-{mode} [{_model_tag(language, adversarial)}]", res)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--adversarial", action="store_true",
                        help="Benchmark the adversarially trained checkpoint instead of clean")
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE[args.language]
    run_attack_benchmark(language=args.language, batch_size=args.batch_size,
                         threshold=args.threshold, base_seed=args.base_seed,
                         adversarial=args.adversarial)
