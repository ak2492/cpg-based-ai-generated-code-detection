"""Bayesian hyperparameter search (Optuna TPE) over train.py's flag space.

Reads graphs from {language}_cpg_bundle.pt (run main.py first), optimizes
best validation F1 exactly as the notebooks select checkpoints, and leaves
the winner in the canonical checkpoint + a reusable train.py command.

Usage:
  pip install -r requirements.txt      # includes optuna
  python main.py --language java --limit 3000
  python bayes_search.py --language java --n_trials 25 --epochs 30 --space core
  python bayes_search.py --language java --list-space   # inspect space, run nothing

BUNDLE REQUIREMENT — READ THIS OR THE SEARCH WILL TIME OUT:
  bayes_search.py reuses whatever graphs are in {language}_cpg_bundle.pt.
  NEVER run it on a full bundle (a single full C++ trial costs 65+ minutes;
  25 trials will not finish before Kaggle kills the session). Always build
  a lightweight proxy bundle first, search on it, then refit the winner on
  the full bundle with the printed reproduce command. Proxy sizes:

    python main.py --language python --limit 5000
    python main.py --language java   --limit 3000
    python main.py --language cpp    --limit 4000

  then e.g. `python bayes_search.py --language cpp --space core
  --epochs 25 --n_trials 25`, take the printed "Reproduce with" command,
  rebuild the full bundle (`python main.py --language cpp`), refit with
  full epochs, and test once (`python evaluate.py --language cpp`).

Resume: re-run the same command; completed trials are picked up from the
SQLite study + CSV and their checkpoints reused, not retrained.
"""

import argparse
import csv
import gc
import json
import os
import shutil
import sys
import time

import torch

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    sys.exit("ERROR: optuna is required for Bayesian search: pip install optuna")

from train import train_model
from language_configs import DEFAULT_BATCH_SIZE, CLEAN_CHECKPOINT, ADV_CHECKPOINT
from pipeline import load_bundle


# Proxy sizes used with `main.py --limit` before searching (see docstring).
PROXY_LIMITS = {"python": 5000, "java": 3000, "cpp": 4000}
# Above this clean-train graph count the search prints a slow-bundle warning.
LARGE_BUNDLE_WARN = 8000


# ---------------------------------------------------------------------------
# Search spaces (values centered on notebook defaults)
# ---------------------------------------------------------------------------

def _batch_choices(language):
    d = DEFAULT_BATCH_SIZE[language]
    return sorted({max(8, d // 2), d, min(256, d * 2)})


def build_space(trial, language, preset):
    """Suggest one configuration. Returns kwargs for train_model()."""
    kw = {}
    # -- core preset: the 6 dims that move val-F1 most per unit cost --
    kw["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 128, 256, 384])
    kw["num_layers"] = trial.suggest_int("num_layers", 2, 5)
    kw["lr"] = trial.suggest_float("lr", 1e-5, 2e-3, log=True)
    kw["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    kw["batch_size"] = trial.suggest_categorical("batch_size", _batch_choices(language))
    kw["dropout_gnn"] = trial.suggest_float("dropout_gnn", 0.0, 0.4)
    if preset == "core":
        return kw
    # -- full preset: everything else output-affecting --
    kw["type_dim"] = trial.suggest_categorical("type_dim", [32, 64, 96])
    kw["subword_dim"] = trial.suggest_categorical("subword_dim", [64, 128, 192])
    kw["pool_hidden"] = trial.suggest_categorical("pool_hidden", [64, 128, 256])
    kw["film_hidden"] = trial.suggest_categorical("film_hidden", [64, 128, 256])
    kw["cls_hidden1"] = trial.suggest_categorical("cls_hidden1", [128, 256, 512])
    kw["cls_hidden2"] = trial.suggest_categorical("cls_hidden2", [32, 64, 128])
    kw["dropout_cls1"] = trial.suggest_float("dropout_cls1", 0.1, 0.5)
    kw["dropout_cls2"] = trial.suggest_float("dropout_cls2", 0.0, 0.4)
    kw["patience"] = trial.suggest_categorical("patience", [5, 10, 15])
    kw["accum_steps"] = trial.suggest_categorical("accum_steps", [1, 2, 4])
    kw["eta_min"] = trial.suggest_float("eta_min", 1e-7, 1e-5, log=True)
    kw["smooth_pos"] = trial.suggest_categorical("smooth_pos", [0.95, 0.975, 0.99])
    kw["grad_clip"] = trial.suggest_categorical("grad_clip", [0.5, 1.0, 2.0])
    kw["mask_rate"] = trial.suggest_float("mask_rate", 0.05, 0.3)
    return kw


def space_snapshot(language, preset):
    """Static view of the space for --list-space (no Optuna run)."""
    snap = {
        "hidden_dim": [64, 128, 256, 384], "num_layers": [2, 5],
        "lr": "[1e-5, 2e-3] log", "weight_decay": "[1e-5, 1e-2] log",
        "batch_size": _batch_choices(language), "dropout_gnn": "[0.0, 0.4]",
    }
    if preset == "full":
        snap.update({
            "type_dim": [32, 64, 96], "subword_dim": [64, 128, 192],
            "pool_hidden": [64, 128, 256],
            "film_hidden": [64, 128, 256], "cls_hidden1": [128, 256, 512],
            "cls_hidden2": [32, 64, 128], "dropout_cls1": "[0.1, 0.5]",
            "dropout_cls2": "[0.0, 0.4]", "patience": [5, 10, 15],
            "accum_steps": [1, 2, 4], "eta_min": "[1e-7, 1e-5] log",
            "smooth_pos": [0.95, 0.975, 0.99], "grad_clip": [0.5, 1.0, 2.0],
            "mask_rate": "[0.05, 0.3]",
        })
    return snap


# ---------------------------------------------------------------------------
# Objective: one train_model call = one trial, scored on best val-F1
# ---------------------------------------------------------------------------

class Objective:
    def __init__(self, args, completed, csv_path):
        self.args = args
        self.completed = completed  # trial numbers already logged (resume)
        self.csv_path = csv_path
        self.fixed_ckpt = (ADV_CHECKPOINT if args.adversarial
                           else CLEAN_CHECKPOINT)[args.language]

    def trial_ckpt(self, n):
        stem, ext = os.path.splitext(self.fixed_ckpt)
        return f"{stem}_bo{n:03d}{ext}"

    def log_trial(self, n, val_f1, val_roc, params, seconds):
        new_file = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["trial", "val_f1", "val_roc", "seconds",
                            "checkpoint"] + sorted(params))
            w.writerow([n, f"{val_f1:.4f}", f"{val_roc:.4f}", f"{seconds:.1f}",
                        self.trial_ckpt(n)] +
                       [params[k] for k in sorted(params)])

    def __call__(self, trial):
        params = build_space(trial, self.args.language, self.args.space)
        if "smooth_pos" in params:
            smooth_neg = round(1.0 - params["smooth_pos"], 3)
        else:
            smooth_neg = 0.025
        t0 = time.perf_counter()
        try:
            model, best_f1, best_roc = train_model(
                language=self.args.language, epochs=self.args.epochs,
                adversarial=self.args.adversarial,
                tmax=self.args.epochs,  # notebook convention: T_max = epochs
                smooth_neg=smooth_neg, threshold=0.50, seed=self.args.seed, **params)
            # Destroy the model reference FIRST: train_model already saved
            # the .pth to disk, and everything below (copy, cache flush)
            # is useless while the graph is still anchored to a local.
            # (`del` + collect + empty_cache BEFORE the copy, not after.)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:  # isolate failures (e.g. OOM on big widths)
            print(f"[trial {trial.number}] FAILED ({type(e).__name__}: {e}); "
                  f"continuing search.", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise  # Optuna records trial as FAIL; sampler moves on
        seconds = time.perf_counter() - t0

        # train_model wrote the canonical name: copy aside before next trial
        shutil.copyfile(self.fixed_ckpt, self.trial_ckpt(trial.number))

        if trial.number not in self.completed:
            self.log_trial(trial.number, best_f1, best_roc, params, seconds)
            self.completed.add(trial.number)
        print(f"[trial {trial.number}] val_F1={best_f1:.4f} "
              f"val_ROC={best_roc:.4f} ({seconds/60:.1f} min)", flush=True)
        return best_f1  # direction="maximize"


def winner_command(args, params):
    parts = [f"python train.py --language {args.language}"]
    if args.adversarial:
        parts.append("--adversarial")
    if args.epochs != 45:
        parts.append(f"--epochs {args.epochs}")
    # I emit --seed and the derived --smooth_neg here because refitting
    # without them silently falls back to defaults and will not reproduce
    # the winning trial.
    parts.append(f"--seed {args.seed}")
    if "smooth_pos" in params:
        parts.append(f"--smooth_neg {round(1.0 - params['smooth_pos'], 3)}")
    for k in sorted(params):
        v = params[k]
        if k == "batch_size" and v == DEFAULT_BATCH_SIZE[args.language]:
            continue
        parts.append(f"--{k} {v}")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Bayesian (TPE) search over train.py flags")
    ap.add_argument("--language", default="python", choices=["python", "java", "cpp"])
    ap.add_argument("--adversarial", action="store_true")
    ap.add_argument("--space", default="core", choices=["core", "full"])
    ap.add_argument("--n_trials", type=int, default=25)
    ap.add_argument("--timeout", type=int, default=None,
                    help="Optuna wall-clock budget in seconds (None = trial count only)")
    ap.add_argument("--epochs", type=int, default=30,
                    help="Epoch cap per trial (refit winner with full 45 after)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--study", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--list-space", action="store_true")
    args = ap.parse_args()

    if args.list_space:
        print(json.dumps(space_snapshot(args.language, args.space), indent=2))
        return

    tag = "adv" if args.adversarial else "clean"
    study_name = args.study or f"{args.language}_{tag}_bo_{args.space}"
    csv_path = args.csv or f"{study_name}.csv"

    # Preflight: bundle must exist (graphs are never rebuilt here). Prints
    # scale so a full-bundle accident is obvious in the first seconds.
    try:
        _pre = load_bundle(args.language)
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}")
    _sizes = {k: len(_pre[k]) for k in
              ("train_clean_graphs", "val_graphs", "test_graphs")}
    _sizes["train_adv_graphs"] = (len(_pre["train_adv_graphs"])
                                  if _pre.get("train_adv_graphs") is not None
                                  else 0)
    print(f"Bundle graphs: clean={_sizes['train_clean_graphs']} "
          f"adv={_sizes['train_adv_graphs']} val={_sizes['val_graphs']} "
          f"test={_sizes['test_graphs']}")
    if _sizes["train_clean_graphs"] > LARGE_BUNDLE_WARN:
        print(f"WARNING: large bundle — Bayesian search will be very slow. "
              f"Rebuild a proxy first, e.g. "
              f"python main.py --language {args.language} "
              f"--limit {PROXY_LIMITS[args.language]}")
    if args.adversarial:
        from pipeline import require_adv_graphs
        try:
            require_adv_graphs(_pre, args.language)
        except RuntimeError as e:
            sys.exit(f"ERROR: {e}")
    del _pre
    gc.collect()

    completed = set()
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    completed.add(int(row["trial"]))
                except (KeyError, ValueError):
                    pass
        print(f"Resuming: {len(completed)} trials already logged in {csv_path}")

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{study_name}.db",
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
    )
    objective = Objective(args, completed, csv_path)
    # catch=(Exception,): a failed trial (e.g. OOM at a big width) records as
    # FAIL and the search continues. Without it Optuna re-raises and one bad
    # trial kills the whole study. KeyboardInterrupt still aborts (BaseException).
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   gc_after_trial=True, catch=(Exception,))

    best = study.best_trial
    print("\n" + "=" * 70)
    print(f"BEST trial {best.number}: val_F1={best.value:.4f}")
    for k in sorted(best.params):
        print(f"  {k} = {best.params[k]}")

    # Leave the winner where evaluate.py/attacks expect it + reproducibility kit
    fixed_ckpt = (ADV_CHECKPOINT if args.adversarial
                  else CLEAN_CHECKPOINT)[args.language]
    shutil.copyfile(objective.trial_ckpt(best.number), fixed_ckpt)
    with open(f"{study_name}_best.json", "w", encoding="utf-8") as f:
        json.dump({"trial": best.number, "val_f1": best.value,
                   "params": best.params,
                   "reproduce": winner_command(args, best.params)}, f, indent=2)
    print(f"Winner checkpoint -> {fixed_ckpt}")
    print(f"Params JSON       -> {study_name}_best.json")
    print("Reproduce with:")
    print("  " + winner_command(args, best.params))
    print("Then test once: "
          f"python evaluate.py --language {args.language}"
          f"{' --adversarial' if args.adversarial else ''}")


if __name__ == "__main__":
    main()
