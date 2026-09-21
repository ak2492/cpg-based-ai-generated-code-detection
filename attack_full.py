"""Full (combined) attack — mirrors notebook full suite (auth+sem+stat).

Usage:
  python attack_full.py --language java --mode basic --target all
  python attack_full.py --language python --mode enhanced
"""
import argparse

from language_configs import DEFAULT_BATCH_SIZE
from attack_evaluation import run_single_attack


def main():
    ap = argparse.ArgumentParser(description="Full Attack Evaluation")
    ap.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    ap.add_argument("--mode", type=str, default="basic", choices=["basic", "enhanced"])
    ap.add_argument("--target", type=str, default="machine", choices=["machine", "all"])
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--base_seed", type=int, default=42)
    ap.add_argument("--adversarial", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.50)
    args = ap.parse_args()
    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE[args.language]
    run_single_attack(language=args.language, attack_type="full", mode=args.mode,
                      batch_size=args.batch_size, threshold=args.threshold,
                      base_seed=args.base_seed, target=args.target,
                      adversarial=args.adversarial)


if __name__ == "__main__":
    main()
