"""Extraction entry point — Cell 1 of the notebooks, hybrid-style.

Builds CPGs (vocab + graphs + clean-locked normalization) and persists them
to {language}_cpg_bundle.pt. Trains NOTHING; training lives only in train.py.

Sequential usage:
  python main.py --language python                 # clean/val/test graphs
  python main.py --language python --adversarial   # additionally builds adv pool + adv graphs
  python train.py --language python [--adversarial]
  python evaluate.py --language python [--adversarial]
"""
import argparse

from pipeline import prepare_graphs, save_bundle


def run_extraction(language="python", trial_samples=None, limit=None, adversarial=False):
    bundle = prepare_graphs(language, trial_samples=trial_samples, limit=limit,
                            adversarial=adversarial)
    path = save_bundle(bundle)
    n_clean = len(bundle["train_clean_graphs"])
    n_adv = len(bundle["train_adv_graphs"]) if bundle.get("train_adv_graphs") is not None else 0
    print(f"Extraction complete: clean={n_clean} adv={n_adv} "
          f"val={len(bundle['val_graphs'])} test={len(bundle['test_graphs'])} -> {path}")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--adversarial", action="store_true",
                        help="Also synthesize the adversarial pool and build adv graphs")
    parser.add_argument("--trial-samples", type=int, default=None,
                        help="Mirror notebook TRIAL_SAMPLES (shuffle+select raw splits)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap balanced splits for quick debugging (None = notebook exact)")
    args = parser.parse_args()

    run_extraction(language=args.language, trial_samples=args.trial_samples,
                   limit=args.limit, adversarial=args.adversarial)
