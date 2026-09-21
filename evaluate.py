"""Evaluation — clean-test eval mirroring the notebook benchmark's clean suite.

Rebuilds graphs exactly as notebooks do (vocab + normalization locked on
clean), loads the matching checkpoint, and reports cost-aware metrics.

Usage (mirrors hybrid folder):
  python evaluate.py --language python
  python evaluate.py --language cpp --adversarial
"""
import argparse
import gc

import torch
from torch_geometric.loader import DataLoader

from language_configs import BPE_VOCAB_SIZE, DEFAULT_BATCH_SIZE, CLEAN_CHECKPOINT, ADV_CHECKPOINT
from model import AdvancedASTGraphEncoder, load_checkpoint
from attack_utils import execute_model_eval_with_cost, print_detailed_metrics_with_cost
from pipeline import prepare_graphs


def evaluate_model(language="python", batch_size=None, adversarial=False, threshold=0.50,
                   trial_samples=None, limit=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE[language]
    print(f"Evaluating {language.upper()} CPG on {device}...")

    bundle = prepare_graphs(language, trial_samples=trial_samples, limit=limit)
    ctx = bundle["ctx"]
    test_graphs = bundle["test_graphs"]

    model = AdvancedASTGraphEncoder(
        num_node_types=ctx.vocab_size,
        bpe_vocab_size=BPE_VOCAB_SIZE,
        pad_idx=ctx.pad_id,
    ).to(device)
    ckpt_file = (ADV_CHECKPOINT if adversarial else CLEAN_CHECKPOINT)[language]
    load_checkpoint(ckpt_file, model, device)
    model.eval()

    loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    res = execute_model_eval_with_cost(model, loader, device, threshold=threshold)
    print("\n--- FINAL TEST METRICS ---")
    print_detailed_metrics_with_cost(f"{language.upper()} {'Adv' if adversarial else 'Clean'}", res)

    del loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--trial-samples", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE[args.language]
    evaluate_model(language=args.language, batch_size=args.batch_size,
                   adversarial=args.adversarial, threshold=args.threshold,
                   trial_samples=args.trial_samples, limit=args.limit)
