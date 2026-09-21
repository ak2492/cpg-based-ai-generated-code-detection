"""End-to-end entry point — mirrors the full notebook top-to-bottom.

Builds graphs (Cell 1) then trains Model 1 clean (Cell 3) and Model 2
adversarial (Cell 4), exactly as notebooks do, while exposing the same
CLI style as AI_Generated_Hybrid_Code_Detection/main.py.

Usage:
  python main.py --language python
  python main.py --language java --epochs 45
  python main.py --language cpp --trial-samples 500 --limit 200
"""
import argparse

import torch

from language_configs import DEFAULT_BATCH_SIZE
from pipeline import prepare_graphs
from train import train_single_model


def run_pipeline(language="python", batch_size=None, epochs=45, patience=10,
                 trial_samples=None, limit=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE[language]

    bundle = prepare_graphs(language, trial_samples=trial_samples, limit=limit)
    ctx = bundle["ctx"]

    model_clean, _, _ = train_single_model(
        bundle["train_clean_graphs"], bundle["val_graphs"], ctx, device,
        batch_size, epochs=epochs, patience=patience,
        adversarial=False, language=language)

    model_adv, _, _ = train_single_model(
        bundle["train_adv_graphs"], bundle["val_graphs"], ctx, device,
        batch_size, epochs=epochs, patience=patience,
        adversarial=True, language=language)

    return {"ctx": ctx, "model_clean": model_clean, "model_adv": model_adv, **bundle}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--trial-samples", type=int, default=None,
                        help="Mirror notebook TRIAL_SAMPLES (shuffle+select raw splits)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap balanced splits for quick debugging (None = notebook exact)")
    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE[args.language]
    run_pipeline(language=args.language, batch_size=args.batch_size, epochs=args.epochs,
                 patience=args.patience, trial_samples=args.trial_samples, limit=args.limit)
