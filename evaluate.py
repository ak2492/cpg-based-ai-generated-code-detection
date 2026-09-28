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
from model import ablation_checkpoint_path, build_encoder_from_checkpoint, canonical_ablation, checkpoint_ablation
from attack_utils import execute_model_eval_with_cost, print_detailed_metrics_with_cost
from pipeline import load_bundle_for_ablation, seed_everything


def evaluate_model(language="python", batch_size=None, adversarial=False, threshold=0.50, seed=42, ablation="full"):
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

    if ablation != "full":
        if adversarial:
            raise ValueError("Ablations run clean-only; --adversarial is only valid with --ablation full.")
        ckpt_file = ablation_checkpoint_path(language, ablation, adversarial)
    else:
        ckpt_file = (ADV_CHECKPOINT if adversarial else CLEAN_CHECKPOINT)[language]
    if checkpoint_ablation(ckpt_file, device) != ablation:
        raise ValueError(
            f"Ablation mismatch: --ablation {ablation} but checkpoint {ckpt_file} "
            f"holds a different config. Graphs and encoder would silently disagree.")
    model, _ = build_encoder_from_checkpoint(ctx, ckpt_file, device)
    model.eval()

    loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    res = execute_model_eval_with_cost(model, loader, device, threshold=threshold)
    print("\n--- FINAL TEST METRICS ---")
    tag = f"{language.upper()} Abl-{ablation}" if ablation != "full" else f"{language.upper()} {'Adv' if adversarial else 'Clean'}"
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
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed (inference is deterministic; kept for 5-seed protocol uniformity)")
    parser.add_argument("--ablation", type=str, default="full",
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
