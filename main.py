"""Extraction entry point — Cell 1 of the notebooks, hybrid-style.

Builds CPGs (vocab + graphs + clean-locked normalization) and persists them
to {language}_cpg_bundle.pt. Trains NOTHING; training lives only in train.py.

Sequential usage:
  python main.py --language python                 # clean/val/test graphs only
  python main.py --language python --adversarial   # adv pool + adv graphs ONLY
                                                   # (requires the clean bundle first;
                                                   #  vocab/stats are reused untouched)
  python train.py --language python [--adversarial]
  python evaluate.py --language python [--adversarial]
"""
import argparse

from pipeline import prepare_graphs, save_bundle, build_adv_only


def run_extraction(language="python", trial_samples=None, limit=None, adversarial=False, seed=42):
    if adversarial:
        return build_adv_only(language, trial_samples=trial_samples, limit=limit, seed=seed)
    bundle = prepare_graphs(language, trial_samples=trial_samples, limit=limit,
                            adversarial=False, seed=seed)
    path = save_bundle(bundle, trial_samples=trial_samples, limit=limit)
    n_clean = len(bundle["train_clean_graphs"])
    print(f"Extraction complete: clean={n_clean} "
          f"val={len(bundle['val_graphs'])} test={len(bundle['test_graphs'])} -> {path}")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--adversarial", action="store_true",
                        help="Build ONLY the adv pool + adv graphs into the existing clean bundle")
    parser.add_argument("--trial-samples", type=int, default=None,
                        help="Mirror notebook TRIAL_SAMPLES (shuffle+select raw splits)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap balanced splits for quick debugging (None = notebook exact)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed (model init/shuffle/dropout); data splits stay locked for comparability")
    args = parser.parse_args()

    run_extraction(language=args.language, trial_samples=args.trial_samples,
                   limit=args.limit, adversarial=args.adversarial, seed=args.seed)
