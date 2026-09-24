"""Evaluation (test step) — clean-test eval mirroring the notebook benchmark's clean suite.

Tests EXACTLY ONE model per run (hybrid style): default evaluates the clean
checkpoint; --adversarial evaluates the adv checkpoint. Graphs always come
from the {language}_cpg_bundle.pt saved by main.py; nothing is rebuilt here.

Sequential usage:
  python main.py --language python [--adversarial]
  python train.py --language python [--adversarial]
  python evaluate.py --language python [--adversarial]
"""
import argparse
import gc

import torch
from torch_geometric.loader import DataLoader

from language_configs import DEFAULT_BATCH_SIZE, CLEAN_CHECKPOINT, ADV_CHECKPOINT
from model import build_encoder_from_checkpoint
from attack_utils import execute_model_eval_with_cost, print_detailed_metrics_with_cost
from pipeline import load_bundle, seed_everything


def evaluate_model(language="python", batch_size=None, adversarial=False, threshold=0.50, seed=42):
    # I seed here for completeness; inference itself is deterministic
    # (shuffle=False), so varying seed must not change these scores.
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE[language]
    print(f"Evaluating {language.upper()} CPG on {device}...")

    bundle = load_bundle(language)
    ctx = bundle["ctx"]
    test_graphs = bundle["test_graphs"]

    ckpt_file = (ADV_CHECKPOINT if adversarial else CLEAN_CHECKPOINT)[language]
    model, _ = build_encoder_from_checkpoint(ctx, ckpt_file, device)
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
    parser.add_argument("--adversarial", action="store_true",
                        help="Evaluate the adversarially trained checkpoint instead of clean")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed (inference is deterministic; kept for 5-seed protocol uniformity)")
    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE[args.language]
    evaluate_model(language=args.language, batch_size=args.batch_size,
                   adversarial=args.adversarial, threshold=args.threshold, seed=args.seed)
