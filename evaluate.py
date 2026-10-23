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
from model import (ablation_checkpoint_path, build_encoder_from_checkpoint, canonical_ablation,
                   checkpoint_ablation, checkpoint_threshold, DEFAULT_ABLATION, checkpoint_path_for_config)
from attack_utils import execute_model_eval_with_cost, print_detailed_metrics_with_cost
from pipeline import load_bundle_for_ablation, seed_everything


def evaluate_model(language="python", batch_size=None, adversarial=False, threshold=None, seed=42, ablation=DEFAULT_ABLATION):
    # I seed here for completeness; inference itself is deterministic
    # (shuffle=False), so varying seed must not change these scores.
    seed_everything(seed)
    ablation = canonical_ablation(ablation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE[language]
    print(f"Evaluating {language.upper()} CPG on {device}...")

    bundle = load_bundle_for_ablation(language, ablation)
    ctx = bundle["ctx"]
    test_graphs = bundle["test_graphs"]

    ckpt_file = checkpoint_path_for_config(language, ablation=ablation, adversarial=adversarial)
    ckpt_abl = checkpoint_ablation(ckpt_file, device)
    if ckpt_abl != ablation:
        raise ValueError(
            f"Ablation mismatch: --ablation {ablation} but checkpoint {ckpt_file} "
            f"holds a different config ({ckpt_abl}). Graphs and encoder would silently disagree.")
    if threshold is None:
        # Val-calibrated cut saved by train.py; old checkpoints -> 0.50.
        threshold = checkpoint_threshold(ckpt_file, device)
        print(f"[i] Decision threshold from checkpoint: {threshold:.4f}")
    model, _ = build_encoder_from_checkpoint(ctx, ckpt_file, device)
    model.eval()

    loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    res = execute_model_eval_with_cost(model, loader, device, threshold=threshold)
    print("\n--- FINAL TEST METRICS ---")
    tag = f"{language.upper()} Abl-{ablation}" if ablation != DEFAULT_ABLATION else f"{language.upper()} {'Adv' if adversarial else 'Clean'}"
    print_detailed_metrics_with_cost(tag, res)

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
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (default: the checkpoint's val-calibrated value)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed (inference is deterministic; kept for 5-seed protocol uniformity)")
    parser.add_argument("--ablation", type=str, default=DEFAULT_ABLATION,
                        help="Study config to evaluate: a single id or '+'-joined combo "
                             "(uses its namespaced checkpoint + derived graphs)")
    args = parser.parse_args()

    try:
        args.ablation = canonical_ablation(args.ablation)
    except ValueError as e:
        parser.error(str(e))

    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE[args.language]
    evaluate_model(language=args.language, batch_size=args.batch_size,
                   adversarial=args.adversarial, threshold=args.threshold, seed=args.seed,
                   ablation=args.ablation)
