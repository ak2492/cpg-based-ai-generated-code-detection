"""Five-seed protocol runner — train + clean eval + 9-suite attacks + external OOD.

Runs the full benchmark for each training seed over IDENTICAL data bundles
(main.py output) and identical attacked sets (attack --base_seed fixed at 42),
then reports every seed's values plus mean +/- std. No manual math.

Sequential usage (bundles first, once per language):
  python main.py --language python
  python run_five_seeds.py --language python
  python run_five_seeds.py --language all --seeds 42-46 --adversarial

Per-seed checkpoints are copied to *_seed{s}.pth BEFORE the next seed
overwrites the canonical file, and copied back before every eval step, so
existing single-seed CLIs work unmodified.
"""
import argparse
import gc
import os
import shutil
import subprocess
import sys

import pandas as pd
import torch

from language_configs import CLEAN_CHECKPOINT, ADV_CHECKPOINT
from train import train_model
from evaluate import evaluate_model
from attack_evaluation import run_attack_benchmark
from pipeline import load_bundle
import external_eval as ext

LANGUAGES = ["python", "java", "cpp"]
METRIC_COLS = ["Acc", "Prec", "Rec", "F1", "ROC", "FPR",
               "Latency_ms", "Throughput", "PeakRAM_MB", "N"]
COLUMNS = (["Language", "Variant", "Seed", "Scenario", "TrainValF1", "TrainValROC"]
           + METRIC_COLS)


def parse_seeds(spec):
    """Accept '42,43,44,45,46', '42-46', or mixes like '42-44,46'."""
    seeds = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    # I dedupe here because I want --resume reruns to stay idempotent.
    return sorted(set(seeds))


def _ckpt_files(language, adversarial):
    ckpt = (ADV_CHECKPOINT if adversarial else CLEAN_CHECKPOINT)[language]
    base, suffix = os.path.splitext(ckpt)
    return ckpt, f"{base}_seed{{s}}{suffix}"


def _external_suites(language):
    # I mirror external_eval.py's per-language availability exactly:
    # HMCorp is python/java only, GPTSniffer is java-only.
    suites = ["semeval_A", "semeval_B"]
    if language in ("python", "java"):
        suites.append("hmcorp")
    if language == "java":
        suites.append("gptsniffer")
    return suites


def _scenario_name(suite):
    return {"semeval_A": "Ext SemEval-A", "semeval_B": "Ext SemEval-B",
            "hmcorp": "Ext HMCorp", "gptsniffer": "Ext GPTSniffer"}.get(suite, suite)


def run_language(language, seeds, adversarial=False, epochs=45, batch_size=None,
                 threshold=0.50, base_seed=42, skip_external=False,
                 out_csv=None, resume=False):
    variant = "adv" if adversarial else "clean"
    tag = "ADV" if adversarial else "CLEAN"
    ckpt_canon, ckpt_tpl = _ckpt_files(language, adversarial)
    rows = []

    prev_all = None
    if out_csv and resume and os.path.exists(out_csv):
        prev_all = pd.read_csv(out_csv)
        prev = prev_all[(prev_all["Language"] != language) | (prev_all["Variant"] != variant)]
        rows.extend(prev.to_dict("records"))
        print(f"[resume] kept {len(prev)} existing rows for other configs.")

    for seed in seeds:
        print("\n" + "=" * 85)
        print(f"SEED {seed} [{language.upper()} {tag}]")
        print("=" * 85)
        seed_ckpt = ckpt_tpl.format(s=seed)

        if resume and os.path.exists(seed_ckpt):
            print(f"[resume] reusing existing {seed_ckpt}, skipping training.")
            best_f1 = best_roc = float("nan")
            if prev_all is not None:
                hit = prev_all[(prev_all["Language"] == language)
                               & (prev_all["Variant"] == variant)
                               & (prev_all["Seed"] == seed)]
                if len(hit):
                    best_f1 = float(hit["TrainValF1"].iloc[0])
                    best_roc = float(hit["TrainValROC"].iloc[0])
        else:
            _, best_f1, best_roc = train_model(
                language=language, epochs=epochs, batch_size=batch_size,
                adversarial=adversarial, seed=seed)
            shutil.copyfile(ckpt_canon, seed_ckpt)
            print(f"Checkpoint archived -> {seed_ckpt}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        def _row(scenario, res):
            if res is None:
                return
            rows.append({
                "Language": language, "Variant": variant, "Seed": seed,
                "Scenario": scenario, "TrainValF1": best_f1, "TrainValROC": best_roc,
                "Acc": res["Acc"], "Prec": res["Prec"], "Rec": res["Rec"],
                "F1": res["F1"], "ROC": res["ROC"], "FPR": res["FPR"],
                "Latency_ms": res["Latency_ms"], "Throughput": res["Throughput"],
                "PeakRAM_MB": res.get("PeakRAM_MB", 0.0), "N": res["N"],
            })

        # I restore the per-seed checkpoint before every eval step because
        # all eval CLIs load the canonical filename.
        shutil.copyfile(seed_ckpt, ckpt_canon)
        res = evaluate_model(language=language, batch_size=batch_size,
                             adversarial=adversarial, threshold=threshold,
                             seed=seed)
        _row("Clean Test", res)

        shutil.copyfile(seed_ckpt, ckpt_canon)
        df = run_attack_benchmark(language=language, batch_size=batch_size,
                                  threshold=threshold, base_seed=base_seed,
                                  adversarial=adversarial)
        for _, r in df.iterrows():
            if r["Scenario"] == "Clean Test":
                continue  # already covered by evaluate_model above
            _row(r["Scenario"], {"Acc": r["Acc"], "Prec": r["Prec"], "Rec": r["Rec"],
                                 "F1": r["F1"], "ROC": r["ROC"], "FPR": r["FPR"],
                                 "Latency_ms": r["Lat"], "Throughput": r["Thr"],
                                 "PeakRAM_MB": r["RAM"], "N": r["N"]})

        if not skip_external:
            shutil.copyfile(seed_ckpt, ckpt_canon)
            bundle = load_bundle(language)
            suites = [s for s in _external_suites(language)]
            for suite in suites:
                if suite.startswith("semeval"):
                    subtask, multi = ("A", False) if suite == "semeval_A" else ("B", True)
                    fn = {"python": ext.run_external_semeval_python,
                          "java": ext.run_external_semeval_java,
                          "cpp": ext.run_external_semeval_cpp}[language]
                    out = fn(bundle, subtask, multi, batch_size, threshold, variant)
                elif suite == "hmcorp":
                    fn = ext.evaluate_hmcorp_python if language == "python" else ext.evaluate_hmcorp_java
                    out = fn(bundle, batch_size, threshold, variant)
                else:
                    out = ext.evaluate_gptsniffer(bundle, batch_size, threshold, variant)
                res = (out or {}).get(variant)
                _row(_scenario_name(suite), res)

        if out_csv:
            pd.DataFrame(rows, columns=COLUMNS).to_csv(out_csv, index=False)

    df_all = pd.DataFrame(rows, columns=COLUMNS)
    if out_csv:
        df_all.to_csv(out_csv, index=False)
    return df_all


def summarize(df_all):
    cur = df_all
    print("\n" + "=" * 100)
    print("PER-SEED RESULTS")
    print("=" * 100)
    print(cur.to_string(index=False))
    summary = cur.groupby(["Language", "Variant", "Scenario"], as_index=False).agg(
        {c: ["mean", "std"] for c in METRIC_COLS + ["TrainValF1", "TrainValROC"]})
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.values]
    print("\n" + "=" * 100)
    print("SUMMARY: MEAN +/- STD (sample std, ddof=1)")
    print("=" * 100)
    show = summary.copy()
    for c in METRIC_COLS:
        show[c] = (summary[f"{c}_mean"].map("{:.4f}".format) + " +/- "
                   + summary[f"{c}_std"].map("{:.4f}".format))
    print(show[["Language", "Variant", "Scenario"] + METRIC_COLS].to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPG 5-seed protocol runner")
    parser.add_argument("--language", type=str, default="python",
                        choices=["python", "java", "cpp", "all"])
    parser.add_argument("--seeds", type=str, default="42-46",
                        help="Seed list, e.g. '42-46' or '7,123,999' (any ints allowed)")
    parser.add_argument("--adversarial", action="store_true",
                        help="Run the adversarial variant instead of clean (default: clean)")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Fixed attack-sampling seed (keep 42 so attacked sets match across training seeds)")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse existing *_seed{s}.pth files and keep other configs' CSV rows")
    parser.add_argument("--upload", action="store_true",
                        help="Upload per-seed .pth files to Hugging Face at the end")
    parser.add_argument("--upload-repo", type=str, default=None)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    print(f"Seeds: {seeds}")
    langs = LANGUAGES if args.language == "all" else [args.language]
    variant = "adv" if args.adversarial else "clean"

    for lang in langs:
        out_csv = f"five_seed_{lang}_{variant}.csv"
        df_all = run_language(lang, seeds, adversarial=args.adversarial,
                              epochs=args.epochs, batch_size=args.batch_size,
                              threshold=args.threshold, base_seed=args.base_seed,
                              skip_external=args.skip_external,
                              out_csv=out_csv, resume=args.resume)
        summary = summarize(df_all[df_all["Language"] == lang])
        summary.to_csv(f"five_seed_{lang}_{variant}_summary.csv", index=False)

        if args.upload:
            repo = args.upload_repo or f"ak2492/cpg_based_detection-models_{lang}"
            script = os.path.join("tools", "hf_upload", "upload_seed_models.py")
            subprocess.run([sys.executable, script, "--model-dir", ".",
                            "--repo", repo, "--pattern", "*_seed*.pth"], check=True)
