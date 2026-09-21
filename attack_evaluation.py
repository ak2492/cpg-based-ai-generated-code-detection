"""9-suite robustness benchmark — mirrors Cell 5/6 of each notebook.

Python/C++ use execute_model_eval_with_cost (with latency); Java Cell 5 uses
execute_model_eval (no cost). This module preserves that per-language
difference while sharing the 9-suite loop. Single-attack scripts
(attack_authorship/statistical/semantic/full) reuse `run_single_attack()`.

Usage:
  python attack_evaluation.py --language python
  python attack_evaluation.py --language java --limit 200
"""
import argparse
import gc

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from language_configs import BPE_VOCAB_SIZE, DEFAULT_BATCH_SIZE, CLEAN_CHECKPOINT, ADV_CHECKPOINT
from model import AdvancedASTGraphEncoder, load_checkpoint
from attack_utils import (
    ATTACK_SUITES,
    generate_attack_samples,
    execute_model_eval,
    execute_model_eval_with_cost,
    print_detailed_metrics,
    print_detailed_metrics_with_cost,
    set_seed,
)
from graph_builder import process_split, apply_normalization
from pipeline import prepare_graphs


def _load_both_models(ctx, device, language, adversarial_only=False):
    def _make():
        return AdvancedASTGraphEncoder(
            num_node_types=ctx.vocab_size,
            bpe_vocab_size=BPE_VOCAB_SIZE,
            pad_idx=ctx.pad_id,
        ).to(device)
    model_clean, model_adv = _make(), _make()
    load_checkpoint(CLEAN_CHECKPOINT[language], model_clean, device)
    load_checkpoint(ADV_CHECKPOINT[language], model_adv, device)
    return model_clean, model_adv


def _parse_desc(language, suite_name):
    if language == "python":
        return f"Parsing Python {suite_name}"
    elif language == "cpp":
        return f"Parsing C++ {suite_name}"
    else:
        return f"Parsing {suite_name}"


def run_attack_benchmark(language="python", batch_size=None, threshold=0.50,
                         trial_samples=None, limit=None, base_seed=42):
    set_seed(base_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE[language]

    bundle = prepare_graphs(language, trial_samples=trial_samples, limit=limit)
    ctx = bundle["ctx"]
    test_graphs = bundle["test_graphs"]
    test_raw = bundle["test_raw"]
    parser = bundle["parser"]

    model_clean, model_adv = _load_both_models(ctx, device, language)

    use_cost = language in ("python", "cpp")
    comparison_records = []

    for suite_name, attack_key, mode in ATTACK_SUITES:
        if attack_key == "clean":
            eval_graphs = test_graphs
        else:
            attack_data = generate_attack_samples(test_raw, attack_key, mode, language, parser)
            eval_graphs = process_split(attack_data, _parse_desc(language, suite_name), ctx)
            apply_normalization(eval_graphs, ctx)

        eval_loader = DataLoader(eval_graphs, batch_size=batch_size, shuffle=False)

        if use_cost:
            res_clean = execute_model_eval_with_cost(model_clean, eval_loader, device, threshold=threshold)
            res_adv = execute_model_eval_with_cost(model_adv, eval_loader, device, threshold=threshold)
        else:
            res_clean = execute_model_eval(model_clean, eval_loader, device, threshold=threshold)
            res_adv = execute_model_eval(model_adv, eval_loader, device, threshold=threshold)

        if use_cost:
            comparison_records.append({
                'Scenario': suite_name,
                'M1_Acc': res_clean['Acc'], 'M1_Prec': res_clean['Prec'], 'M1_Rec': res_clean['Rec'],
                'M1_F1': res_clean['F1'], 'M1_ROC': res_clean['ROC'], 'M1_FPR': res_clean['FPR'],
                'M1_Lat': res_clean['Latency_ms'],
                'M2_Acc': res_adv['Acc'], 'M2_Prec': res_adv['Prec'], 'M2_Rec': res_adv['Rec'],
                'M2_F1': res_adv['F1'], 'M2_ROC': res_adv['ROC'], 'M2_FPR': res_adv['FPR'],
                'M2_Lat': res_adv['Latency_ms'],
            })
        else:
            comparison_records.append({
                'Suite': suite_name,
                'M1_Acc': res_clean['Acc'], 'M1_Prec': res_clean['Prec'], 'M1_Rec': res_clean['Rec'],
                'M1_F1': res_clean['F1'], 'M1_ROC': res_clean['ROC'], 'M1_FPR': res_clean['FPR'],
                'M2_Acc': res_adv['Acc'], 'M2_Prec': res_adv['Prec'], 'M2_Rec': res_adv['Rec'],
                'M2_F1': res_adv['F1'], 'M2_ROC': res_adv['ROC'], 'M2_FPR': res_adv['FPR'],
            })

        print("\n" + "=" * 85)
        print(f"BENCHMARK: {suite_name.upper()} (N = {res_clean['N']})")
        print("=" * 85)
        if use_cost:
            print_detailed_metrics_with_cost("Model 1 (Clean Baseline)", res_clean)
            print("-" * 85)
            print_detailed_metrics_with_cost("Model 2 (Adversarial GNN)", res_adv)
        else:
            print_detailed_metrics("Model 1 (Clean Baseline)", res_clean)
            print("-" * 85)
            print_detailed_metrics("Model 2 (Adversarial GNN)", res_adv)

        if attack_key != "clean":
            del attack_data, eval_graphs
        del eval_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_comp = pd.DataFrame(comparison_records)
    title = {"python": "HEAD-TO-HEAD PYTHON COMPARISON SUMMARY",
             "cpp": "HEAD-TO-HEAD C++ COMPARISON SUMMARY"}.get(language, "HEAD-TO-HEAD SUMMARY MATRIX")
    print("\n" + "=" * 85 + f"\n{title}\n" + "=" * 85)
    print(df_comp.to_string(index=False))
    return df_comp


def run_single_attack(language="python", attack_type="auth", mode="enhanced", batch_size=None,
                      threshold=0.50, trial_samples=None, limit=None, base_seed=42,
                      target="machine", adversarial=False):
    """Single-layer CLI used by attack_authorship/statistical/semantic/full.

    Notebook rule is preserved exactly: basic attacks ALL samples,
    enhanced attacks machine-only. `target` is accepted for hybrid-folder
    CLI compatibility but does not alter notebook outputs.
    """
    set_seed(base_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE[language]

    bundle = prepare_graphs(language, trial_samples=trial_samples, limit=limit)
    ctx = bundle["ctx"]
    test_graphs = bundle["test_graphs"]
    test_raw = bundle["test_raw"]
    parser = bundle["parser"]

    from model import AdvancedASTGraphEncoder, load_checkpoint
    model = AdvancedASTGraphEncoder(num_node_types=ctx.vocab_size,
                                    bpe_vocab_size=BPE_VOCAB_SIZE,
                                    pad_idx=ctx.pad_id).to(device)
    ckpt_file = (ADV_CHECKPOINT if adversarial else CLEAN_CHECKPOINT)[language]
    load_checkpoint(ckpt_file, model, device)

    if attack_type == "clean":
        eval_graphs = test_graphs
    else:
        attack_data = generate_attack_samples(test_raw, attack_type, mode, language, parser)
        eval_graphs = process_split(attack_data, _parse_desc(language, f"{attack_type}-{mode}"), ctx)
        apply_normalization(eval_graphs, ctx)

    loader = DataLoader(eval_graphs, batch_size=batch_size, shuffle=False)
    res = execute_model_eval_with_cost(model, loader, device, threshold=threshold)
    print_detailed_metrics_with_cost(f"{attack_type}-{mode}", res)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--trial-samples", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE[args.language]
    run_attack_benchmark(language=args.language, batch_size=args.batch_size,
                         threshold=args.threshold, trial_samples=args.trial_samples,
                         limit=args.limit, base_seed=args.base_seed)
