"""5-seed benchmark runner with extract-once attack graphs cache.

Trains 5 random-seed instances (seeds 42-46) per language on clean or adversarial
data, saves per-seed checkpoint archives alongside the canonical file, evaluates
each model across 27 attack scenarios (Sec 4.7 benchmark) + held-out external
benchmarks (SemEval-2024 Task 8 Subtask A/B, HMCorp, GPTSniffer), and writes
full metric rows + a mean +/- std summary CSV.

Sequential usage:
  python run_five_seeds.py --language python
  python run_five_seeds.py --language java --adversarial
  python run_five_seeds.py --language all --seeds 42-46
"""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd
import torch

from language_configs import CLEAN_CHECKPOINT, ADV_CHECKPOINT
from model import checkpoint_threshold, DEFAULT_ABLATION
from train import train_model
from evaluate import evaluate_model
from attack_evaluation import run_attack_benchmark
from pipeline import load_bundle, ensure_data
import external_eval as ext

MODEL_NAME = "cpg"
LANGUAGES = ["python", "java", "cpp"]
METRIC_COLS = ["Acc", "Prec", "Rec", "F1", "ROC", "FPR",
               "Tokens", "Latency_ms", "Throughput", "FilesPerSec",
               "MsPerToken", "PeakRAM_MB", "N"]
COLUMNS = (["Language", "Variant", "Seed", "Scenario", "Threshold", "TrainValF1", "TrainValROC"]
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


def _write_timing(out_csv, language, variant, seeds, df, total_wall_s,
                  training_s, eval_s, data_s, clean_eval_s,
                  attack_eval_s, external_eval_s):
    """Persist run-level deployment accounting next to the metrics CSV.

    `non_train_s` follows the requested all-tests formula: total wall time
    (preprocessing + every clean/attack/external evaluation + model loading)
    minus training time, divided by every file/token that was scored.
    """
    if not out_csv:
        return None
    n = pd.to_numeric(df.get("N", pd.Series(dtype=float)), errors="coerce").fillna(0)
    tokens = pd.to_numeric(df.get("Tokens", pd.Series(dtype=float)),
                           errors="coerce").fillna(0)
    total_file_evals = int(n.sum())
    total_tokens = int(tokens.sum())
    non_train_s = max(0.0, float(total_wall_s) - float(training_s))

    clean = df[df["Scenario"] == "Clean Test"] if "Scenario" in df else df.iloc[0:0]
    clean_n = int(pd.to_numeric(clean.get("N", pd.Series(dtype=float)),
                                errors="coerce").fillna(0).sum())
    clean_tokens = int(pd.to_numeric(clean.get("Tokens", pd.Series(dtype=float)),
                                     errors="coerce").fillna(0).sum())
    unique_test_files = int(pd.to_numeric(clean.get("N", pd.Series(dtype=float)),
                                          errors="coerce").dropna().iloc[-1]) \
        if len(clean) and clean["N"].notna().any() else 0

    payload = {
        "model": MODEL_NAME,
        "language": language,
        "variant": variant,
        "seeds": list(seeds),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_wall_s": round(float(total_wall_s), 6),
        "preprocessing_s": round(float(data_s), 6),
        "training_s": round(float(training_s), 6),
        "eval_s": round(float(eval_s), 6),
        "clean_eval_s": round(float(clean_eval_s), 6),
        "attack_eval_s": round(float(attack_eval_s), 6),
        "external_eval_s": round(float(external_eval_s), 6),
        "non_train_s": round(non_train_s, 6),
        "unique_test_files": unique_test_files,
        "total_file_evaluations": total_file_evals,
        "total_tokens": total_tokens,
        "amortized_ms_per_file": round(1000.0 * non_train_s / total_file_evals, 6)
        if total_file_evals else None,
        "amortized_ms_per_token": round(1000.0 * non_train_s / total_tokens, 6)
        if total_tokens else None,
        "clean_test": {
            "file_evaluations": clean_n,
            "tokens": clean_tokens,
            "eval_s": round(float(clean_eval_s), 6),
            "ms_per_file": round(1000.0 * float(clean_eval_s) / clean_n, 6)
            if clean_n else None,
            "ms_per_token": round(1000.0 * float(clean_eval_s) / clean_tokens, 6)
            if clean_tokens else None,
            "files_per_sec": round(clean_n / float(clean_eval_s), 6)
            if clean_eval_s > 0 else None,
            "tokens_per_sec": round(clean_tokens / float(clean_eval_s), 6)
            if clean_eval_s > 0 else None,
        },
        "note": ("Clean Test rows score precomputed graphs; raw-code end-to-end "
                 "deployment numbers come from deployment_benchmark.py."),
    }
    path = os.path.splitext(out_csv)[0] + "_timing.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[i] Wrote run-level deployment timing -> {path}")
    return path


def run_language(language, seeds, adversarial=False, epochs=30, batch_size=None,
                 threshold=None, base_seed=42, skip_external=False,
                 out_csv=None, resume=False, graphs_cache_dir=".",
                 rebuild_cache=False, rebuild_bundle=False,
                 ablation=DEFAULT_ABLATION):
    t_run_start = time.perf_counter()
    data_s = training_s = eval_s = 0.0
    clean_eval_s = attack_eval_s = external_eval_s = 0.0

    # Ensure CPG bundle exists; auto-extract via main.py if missing
    t_phase = time.perf_counter()
    ensure_data(language, adversarial=adversarial, rebuild=rebuild_bundle)
    data_s += time.perf_counter() - t_phase
    if rebuild_bundle:
        # A rebuilt bundle invalidates every derived graph cache. Do this
        # explicitly instead of keying cache names on the bundle mtime.
        rebuild_cache = True
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
        # I start each seed from a clean memory state because 5 seeds of
        # alloc/free cycles fragment VRAM; the reset also keeps peak stats honest.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
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
            t_phase = time.perf_counter()
            trained, best_f1, best_roc = train_model(
                language=language, epochs=epochs, batch_size=batch_size,
                adversarial=adversarial, seed=seed, ablation=ablation)
            training_s += time.perf_counter() - t_phase
            shutil.copyfile(ckpt_canon, seed_ckpt)
            print(f"Checkpoint archived -> {seed_ckpt}")
            # I delete the trained model here because the next seed allocates
            # a fresh model + optimizer; keeping this alive would double VRAM.
            del trained
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        def _row(scenario, res):
            if res is None:
                return
            rows.append({
                "Language": language, "Variant": variant, "Seed": seed,
                "Scenario": scenario, "Threshold": seed_threshold,
                "TrainValF1": best_f1, "TrainValROC": best_roc,
                "Acc": res["Acc"], "Prec": res["Prec"], "Rec": res["Rec"],
                "F1": res["F1"], "ROC": res["ROC"], "FPR": res["FPR"],
                "Tokens": res.get("Tokens", 0),
                "Latency_ms": res["Latency_ms"], "Throughput": res["Throughput"],
                "FilesPerSec": res.get("FilesPerSec", 0.0),
                "MsPerToken": res.get("MsPerToken", 0.0),
                "PeakRAM_MB": res.get("PeakRAM_MB", 0.0), "N": res["N"],
            })

        # I restore the per-seed checkpoint before every eval step because
        # all eval CLIs load the canonical filename.
        shutil.copyfile(seed_ckpt, ckpt_canon)
        # One val-calibrated threshold per seed: read the value train.py
        # stored, so clean/attack/external evals all use the same cut and
        # the 5-seed spread reflects model variance, not threshold drift.
        seed_threshold = threshold if threshold is not None else checkpoint_threshold(ckpt_canon)
        print(f"[i] Seed {seed} decision threshold: {seed_threshold:.4f}"
              + (" (checkpoint-calibrated)" if threshold is None else " (forced via --threshold)"))
        t_phase = time.perf_counter()
        res = evaluate_model(language=language, batch_size=batch_size,
                             adversarial=adversarial, threshold=seed_threshold,
                             seed=seed, ablation=ablation)
        dt = time.perf_counter() - t_phase
        clean_eval_s += dt
        eval_s += dt
        _row("Clean Test", res)
        del res

        shutil.copyfile(seed_ckpt, ckpt_canon)
        t_phase = time.perf_counter()
        df = run_attack_benchmark(language=language, batch_size=batch_size,
                                  threshold=seed_threshold, base_seed=base_seed,
                                  adversarial=adversarial,
                                  graphs_cache_dir=graphs_cache_dir,
                                  rebuild_cache=rebuild_cache)
        for _, r in df.iterrows():
            if r["Scenario"] == "Clean Test":
                continue  # already covered by evaluate_model above
            _row(r["Scenario"], {"Acc": r["Acc"], "Prec": r["Prec"], "Rec": r["Rec"],
                                 "F1": r["F1"], "ROC": r["ROC"], "FPR": r["FPR"],
                                 "Tokens": r.get("Tokens", 0),
                                 "Latency_ms": r.get("Latency_ms", 0.0),
                                 "Throughput": r.get("Throughput", 0.0),
                                 "FilesPerSec": r.get("FilesPerSec", 0.0),
                                 "MsPerToken": r.get("MsPerToken", 0.0),
                                 "PeakRAM_MB": r.get("PeakRAM_MB", 0.0),
                                 "N": r.get("N", 0)})
        dt = time.perf_counter() - t_phase
        attack_eval_s += dt
        eval_s += dt

        if not skip_external:
            shutil.copyfile(seed_ckpt, ckpt_canon)
            t_phase = time.perf_counter()
            bundle = load_bundle(language)
            for suite in _external_suites(language):
                if suite.startswith("semeval_"):
                    subtask = suite.split("_")[1].upper()
                    multi = (subtask == "B")
                    fn = {"python": ext.run_external_semeval_python,
                          "java": ext.run_external_semeval_java,
                          "cpp": ext.run_external_semeval_cpp}[language]
                    out = fn(bundle, subtask, multi, batch_size, seed_threshold, variant,
                             graphs_cache_dir, rebuild_cache)
                elif suite == "hmcorp":
                    fn = ext.evaluate_hmcorp_python if language == "python" else ext.evaluate_hmcorp_java
                    out = fn(bundle, batch_size, seed_threshold, variant,
                             graphs_cache_dir, rebuild_cache)
                else:
                    out = ext.evaluate_gptsniffer(bundle, batch_size, seed_threshold, variant,
                                                  graphs_cache_dir, rebuild_cache)
                res = (out or {}).get(variant)
                _row(_scenario_name(suite), res)
                del out, res
            del bundle
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            dt = time.perf_counter() - t_phase
            external_eval_s += dt
            eval_s += dt

        # Incremental save after each seed finishes so a killed session leaves
        # a complete CSV of every seed run up to that point.
        df_seed = pd.DataFrame(rows, columns=COLUMNS)
        if out_csv:
            df_seed.to_csv(out_csv, index=False)
            print(f"[{language.upper()}] incremental rows saved to {out_csv} "
                  f"({len(df_seed)} rows through seed {seed})")

    df_all = pd.DataFrame(rows, columns=COLUMNS)
    _write_timing(out_csv, language, variant, seeds, df_all,
                  time.perf_counter() - t_run_start, training_s, eval_s,
                  data_s, clean_eval_s, attack_eval_s, external_eval_s)
    return df_all


def summarize(df):
    """Compute mean +/- std across seeds for each scenario."""
    if df.empty:
        return pd.DataFrame()
    records = []
    # I group by (Language, Variant, Scenario) so multi-config CSVs (e.g. from
    # --language all) summarize cleanly without cross-language cross-contamination.
    keys = ["Language", "Variant", "Scenario"]
    for (lang, var, scen), group in df.groupby(keys, sort=False):
        row = {"Language": lang, "Variant": var, "Scenario": scen,
               "Seeds": len(group["Seed"].unique())}
        for col in METRIC_COLS:
            vals = group[col].dropna()
            if len(vals) == 0:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_std"] = float("nan")
            else:
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        records.append(row)
    summary = pd.DataFrame(records)
    # Order scenarios logically: Clean Test first, then attacks, then external
    def _rank(scen):
        if scen == "Clean Test":
            return (0, scen)
        if scen.startswith("Ext "):
            return (2, scen)
        return (1, scen)
    summary["_rank"] = summary["Scenario"].apply(_rank)
    summary = summary.sort_values(by=["Language", "Variant", "_rank"]).drop(columns=["_rank"])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPG 5-seed protocol runner")
    parser.add_argument("--language", type=str, default="python",
                        choices=["python", "java", "cpp", "all"])
    parser.add_argument("--seeds", type=str, default="42-46",
                        help="Seed list, e.g. '42-46' or '7,123,999' (any ints allowed)")
    parser.add_argument("--adversarial", action="store_true",
                        help="Run the adversarial variant instead of clean (default: clean)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold for every eval step (default: each seed's "
                             "checkpoint-calibrated value from training)")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Fixed attack-sampling seed (keep 42 so attacked sets match across training seeds)")
    parser.add_argument("--ablation", type=str, default=DEFAULT_ABLATION,
                        help="Model architecture ablation (default: L4_mp2 'mean-pool+no-film')")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--no-feature-cache", action="store_true",
                        help="Disable the extract-once graph cache (parse every seed)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force re-parsing even when cache files exist")
    parser.add_argument("--rebuild-bundle", action="store_true",
                        help="Force re-extracting the CPG bundle via main.py "
                             "(also rebuilds the derived graph caches)")
    parser.add_argument("--cache-dir", type=str, default=".",
                        help="Directory for extract-once cache files")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse existing *_seed{s}.pth files and keep other configs' CSV rows")
    parser.add_argument("--upload", action="store_true",
                        help="Upload per-seed .pth files to Hugging Face at the end")
    parser.add_argument("--upload-repo", type=str, default=None,
                        help="Required with --upload: full repo id 'owner/name' (no default; "
                             "missing repo/token/files skips upload gracefully)")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    print(f"Seeds: {seeds}")
    langs = LANGUAGES if args.language == "all" else [args.language]
    variant = "adv" if args.adversarial else "clean"

    # I pass None (not ".") when caching is off so library code takes its
    # original uncached path byte-for-byte.
    graphs_cache_dir = None if args.no_feature_cache else args.cache_dir
    for lang in langs:
        out_csv = f"five_seed_{lang}_{variant}.csv"
        df_all = run_language(lang, seeds, adversarial=args.adversarial,
                              epochs=args.epochs, batch_size=args.batch_size,
                              threshold=args.threshold, base_seed=args.base_seed,
                              skip_external=args.skip_external,
                              out_csv=out_csv, resume=args.resume,
                              graphs_cache_dir=graphs_cache_dir,
                              rebuild_cache=args.rebuild_cache,
                              rebuild_bundle=args.rebuild_bundle,
                              ablation=args.ablation)
        summary = summarize(df_all[df_all["Language"] == lang])
        summary.to_csv(f"five_seed_{lang}_{variant}_summary.csv", index=False)

        if args.upload:
            # I require an explicit repo and never default to a personal
            # account; anything missing is a warned skip, never an error.
            if not args.upload_repo:
                print("[!] --upload needs --upload-repo <owner/name>; skipping upload.")
                continue
            script = os.path.join("tools", "hf_upload", "upload_seed_models.py")
            subprocess.run([sys.executable, script, "--model-dir", ".",
                            "--repo", args.upload_repo,
                            "--pattern", "*_seed*.pth"], check=True)
