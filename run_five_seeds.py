"""Five-seed protocol runner — train + clean eval + 9-suite attacks + external OOD.

Runs the full benchmark for each training seed over IDENTICAL data bundles
(main.py output) and identical attacked sets (attack --base_seed fixed at 42),
then reports every seed's values plus mean +/- std. No manual math.

Sequential usage (bundles first, once per language):
  python main.py --language python
  python run_five_seeds.py --language python
  python run_five_seeds.py --language all --seeds 42-46 --adversarial

Eval-only attack table from already-trained per-seed checkpoints:
  python run_five_seeds.py --language all --stage attack \
      --seeds 42-46 --threshold-mode calibrated

Decision thresholds default to the clean-validation calibrated cut used by
the CPG and hybrid five-seed tables; pass --threshold-mode fixed for 0.50.

Per-seed checkpoints are copied to *_seed{s}.pth BEFORE the next seed
overwrites the canonical file, and copied back before every eval step, so
existing single-seed CLIs work unmodified.
"""
import argparse
import gc
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd
import torch

from language_configs import CLEAN_CHECKPOINT, ADV_CHECKPOINT
from train import train_model
from evaluate import evaluate_model
from attack_evaluation import run_attack_benchmark
from pipeline import load_bundle
from threshold import compute_seed_threshold, test_prior
import external_eval as ext

MODEL_NAME = "legacy"
LANGUAGES = ["python", "java", "cpp"]
METRIC_COLS = ["Acc", "Prec", "Rec", "F1", "ROC", "FPR",
               "Tokens", "Latency_ms", "Throughput", "FilesPerSec",
               "MsPerToken", "PeakRAM_MB", "N"]
COLUMNS = (["Language", "Variant", "Seed", "Scenario", "Threshold", "ThresholdMode",
            "TrainValF1", "TrainValROC"]
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


def purge_attack_cache(cache_dir, language):
    """Delete this language's attack/external graph caches.

    The cache key embeds the bundle mtime, but a cache written from a
    different split can still be picked up when the bundle is rebuilt with a
    similar timestamp. Purging once before a comparable run removes that
    silent failure mode; later seeds still reuse the caches the first seed
    writes (base_seed is fixed at 42).
    """
    if cache_dir is None:
        return []
    removed = []
    for pattern in (f"{language}_graphs_*.pt", f"{language}_extgraphs_*.pt"):
        for path in glob.glob(os.path.join(cache_dir, pattern)):
            try:
                os.remove(path)
                removed.append(path)
            except OSError as exc:
                print(f"[!] Could not remove cache {path}: {exc}")
    return removed


def _resolve_threshold(language, adversarial, checkpoint, forced, mode,
                       eval_prior, batch_size):
    """Return the cut for one seed.

    `forced` (--threshold) always wins. In calibrated mode the cut is derived
    here from clean validation with the CPG rule, because legacy checkpoints
    only store the fixed 0.50 passed to train.py. Fixed mode keeps 0.50.
    """
    if forced is not None:
        return float(forced)
    if mode == "calibrated":
        return compute_seed_threshold(language, adversarial=adversarial,
                                      checkpoint=checkpoint, fallback=0.50,
                                      prior=eval_prior, batch_size=batch_size)
    return 0.50


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
                 rebuild_cache=False, threshold_mode="calibrated",
                 stage="full", target="machine", purge_cache=False):
    """Run one language.

    stage="full"  -> train (or reuse) each seed, then clean eval + attacks + external.
    stage="attack" -> no training, no external: reload the archived per-seed
                      checkpoints and run only the 9-suite attack benchmark.
                      This is the eval-side stage that makes legacy rows
                      directly comparable to the CPG/hybrid tables.
    """
    if threshold_mode not in ("fixed", "calibrated"):
        raise ValueError(f"Unknown threshold_mode {threshold_mode!r}")
    if stage not in ("full", "attack"):
        raise ValueError(f"Unknown stage {stage!r}")

    t_run_start = time.perf_counter()
    data_s = training_s = eval_s = 0.0
    clean_eval_s = attack_eval_s = external_eval_s = 0.0

    variant = "adv" if adversarial else "clean"
    tag = "ADV" if adversarial else "CLEAN"
    ckpt_canon, ckpt_tpl = _ckpt_files(language, adversarial)
    rows = []

    if purge_cache:
        removed = purge_attack_cache(graphs_cache_dir, language)
        print(f"[i] Purged {len(removed)} attack/external graph cache file(s) "
              f"for {language}.")

    # The calibration prior is the clean test-set prevalence: one number per
    # language, independent of the training seed. A forced --threshold does
    # not need it, so skip the extra bundle load in that case.
    eval_prior = (test_prior(language)
                  if (threshold_mode == "calibrated" and threshold is None)
                  else None)
    if eval_prior is not None:
        print(f"[i] Eval prior for {language}: {eval_prior:.4f}")

    prev_all = None
    if out_csv and resume and os.path.exists(out_csv):
        prev_all = pd.read_csv(out_csv)
        prev = prev_all[(prev_all["Language"] != language) | (prev_all["Variant"] != variant)]
        rows.extend(prev.to_dict("records"))
        print(f"[resume] kept {len(prev)} existing rows for other configs.")
    elif stage == "attack":
        # Recover TrainValF1/ROC from the full five-seed table when it exists,
        # so the attack CSV carries the same metadata columns as CPG/hybrid.
        meta_path = f"five_seed_{language}_{variant}.csv"
        if os.path.exists(meta_path):
            prev_all = pd.read_csv(meta_path)
            print(f"[attack] reusing TrainVal metadata from {meta_path}.")

    for seed in seeds:
        # I start each seed from a clean memory state because 5 seeds of
        # alloc/free cycles fragment VRAM; the reset also keeps peak stats honest.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        print("\n" + "=" * 85)
        print(f"SEED {seed} [{language.upper()} {tag}] stage={stage}")
        print("=" * 85)
        seed_ckpt = ckpt_tpl.format(s=seed)

        if stage == "attack":
            # The attack stage consumes checkpoints only. It never trains and
            # never falls back to the canonical file, because that file may
            # belong to a different seed after an interrupted run.
            if not os.path.exists(seed_ckpt):
                raise FileNotFoundError(
                    f"[attack] missing archived checkpoint {seed_ckpt}. "
                    f"Run `python run_five_seeds.py --language {language} "
                    f"--seeds {seed}` first (or restore the *_seed*.pth files).")
            print(f"[attack] reusing {seed_ckpt}; training skipped.")
            best_f1 = best_roc = float("nan")
            if prev_all is not None:
                hit = prev_all[(prev_all["Language"] == language)
                               & (prev_all["Variant"] == variant)
                               & (prev_all["Seed"] == seed)]
                if len(hit):
                    if "TrainValF1" in hit:
                        best_f1 = float(hit["TrainValF1"].iloc[0])
                    if "TrainValROC" in hit:
                        best_roc = float(hit["TrainValROC"].iloc[0])
            shutil.copyfile(seed_ckpt, ckpt_canon)
        elif resume and os.path.exists(seed_ckpt):
            print(f"[resume] reusing existing {seed_ckpt}, skipping training.")
            best_f1 = best_roc = float("nan")
            if prev_all is not None:
                hit = prev_all[(prev_all["Language"] == language)
                               & (prev_all["Variant"] == variant)
                               & (prev_all["Seed"] == seed)]
                if len(hit):
                    if "TrainValF1" in hit:
                        best_f1 = float(hit["TrainValF1"].iloc[0])
                    if "TrainValROC" in hit:
                        best_roc = float(hit["TrainValROC"].iloc[0])
        else:
            t_phase = time.perf_counter()
            trained, best_f1, best_roc = train_model(
                language=language, epochs=epochs, batch_size=batch_size,
                adversarial=adversarial, seed=seed)
            training_s += time.perf_counter() - t_phase
            shutil.copyfile(ckpt_canon, seed_ckpt)
            print(f"Checkpoint archived -> {seed_ckpt}")
            # I delete the trained model here because the next seed allocates
            # a fresh model + optimizer; keeping this alive would double VRAM.
            del trained
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # I restore the per-seed checkpoint before every eval step because
        # all eval/attack CLIs load the canonical filename.
        shutil.copyfile(seed_ckpt, ckpt_canon)

        seed_threshold = _resolve_threshold(
            language, adversarial, ckpt_canon, threshold, threshold_mode,
            eval_prior, batch_size)
        how = ("forced" if threshold is not None
               else ("clean-val calibrated" if threshold_mode == "calibrated" else "fixed"))
        print(f"[i] Seed {seed} threshold {seed_threshold:.4f} ({how})")

        def _row(scenario, res):
            if res is None:
                return
            rows.append({
                "Language": language, "Variant": variant, "Seed": seed,
                "Scenario": scenario, "Threshold": seed_threshold,
                "ThresholdMode": threshold_mode,
                "TrainValF1": best_f1, "TrainValROC": best_roc,
                "Acc": res["Acc"], "Prec": res["Prec"], "Rec": res["Rec"],
                "F1": res["F1"], "ROC": res["ROC"], "FPR": res["FPR"],
                "Tokens": res.get("Tokens", 0),
                "Latency_ms": res["Latency_ms"], "Throughput": res["Throughput"],
                "FilesPerSec": res.get("FilesPerSec", 0.0),
                "MsPerToken": res.get("MsPerToken", 0.0),
                "PeakRAM_MB": res.get("PeakRAM_MB", 0.0), "N": res["N"],
            })

        if stage == "attack":
            t_phase = time.perf_counter()
            df = run_attack_benchmark(language=language, batch_size=batch_size,
                                      threshold=seed_threshold, base_seed=base_seed,
                                      adversarial=adversarial, target=target,
                                      graphs_cache_dir=graphs_cache_dir,
                                      rebuild_cache=rebuild_cache)
            for _, r in df.iterrows():
                # Keep the benchmark's own Clean Test row: the attack stage
                # does not run evaluate_model separately.
                _row(r["Scenario"], {"Acc": r["Acc"], "Prec": r["Prec"],
                                     "Rec": r["Rec"], "F1": r["F1"],
                                     "ROC": r["ROC"], "FPR": r["FPR"],
                                     "Tokens": r.get("Tokens", 0),
                                     "Latency_ms": r["Latency_ms"],
                                     "Throughput": r["Throughput"],
                                     "FilesPerSec": r.get("FilesPerSec", 0.0),
                                     "MsPerToken": r.get("MsPerToken", 0.0),
                                     "PeakRAM_MB": r["PeakRAM_MB"], "N": r["N"]})
            del df
            dt = time.perf_counter() - t_phase
            attack_eval_s += dt
            eval_s += dt
        else:
            t_phase = time.perf_counter()
            res = evaluate_model(language=language, batch_size=batch_size,
                                 adversarial=adversarial, threshold=seed_threshold,
                                 seed=seed)
            _row("Clean Test", res)
            del res
            dt = time.perf_counter() - t_phase
            clean_eval_s += dt
            eval_s += dt

            shutil.copyfile(seed_ckpt, ckpt_canon)
            t_phase = time.perf_counter()
            df = run_attack_benchmark(language=language, batch_size=batch_size,
                                      threshold=seed_threshold, base_seed=base_seed,
                                      adversarial=adversarial, target=target,
                                      graphs_cache_dir=graphs_cache_dir,
                                      rebuild_cache=rebuild_cache)
            for _, r in df.iterrows():
                if r["Scenario"] == "Clean Test":
                    continue  # already covered by evaluate_model above
                _row(r["Scenario"], {"Acc": r["Acc"], "Prec": r["Prec"], "Rec": r["Rec"],
                                     "F1": r["F1"], "ROC": r["ROC"], "FPR": r["FPR"],
                                     "Tokens": r.get("Tokens", 0),
                                     "Latency_ms": r["Latency_ms"],
                                     "Throughput": r["Throughput"],
                                     "FilesPerSec": r.get("FilesPerSec", 0.0),
                                     "MsPerToken": r.get("MsPerToken", 0.0),
                                     "PeakRAM_MB": r["PeakRAM_MB"], "N": r["N"]})
            del df
            dt = time.perf_counter() - t_phase
            attack_eval_s += dt
            eval_s += dt

            if not skip_external:
                shutil.copyfile(seed_ckpt, ckpt_canon)
                t_phase = time.perf_counter()
                bundle = load_bundle(language)
                suites = [s for s in _external_suites(language)]
                for suite in suites:
                    if suite.startswith("semeval"):
                        subtask, multi = ("A", False) if suite == "semeval_A" else ("B", True)
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
                # I drop the bundle here because it holds all graph lists; the
                # next seed reloads it fresh and I want no CPU-RAM overlap.
                del bundle
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                dt = time.perf_counter() - t_phase
                external_eval_s += dt
                eval_s += dt

        if out_csv:
            pd.DataFrame(rows, columns=COLUMNS).to_csv(out_csv, index=False)

    df_all = pd.DataFrame(rows, columns=COLUMNS)
    if out_csv:
        df_all.to_csv(out_csv, index=False)
    _write_timing(out_csv, language, variant, seeds, df_all,
                  time.perf_counter() - t_run_start, training_s, eval_s,
                  data_s, clean_eval_s, attack_eval_s, external_eval_s)
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--stage", type=str, default="full", choices=["full", "attack"],
                        help="'full' = train/eval/attacks/external; 'attack' = eval-side only, "
                             "reload archived *_seed*.pth checkpoints and run the 9-suite benchmark")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Force one decision cut for every seed (default: calibrated per seed)")
    parser.add_argument("--threshold-mode", type=str, default="calibrated",
                        choices=["fixed", "calibrated"],
                        help="'calibrated' (default) = clean-validation cut with the CPG rule; "
                             "'fixed' = 0.50")
    parser.add_argument("--target", type=str, default="machine", choices=["machine", "all"],
                        help="Attack target rows (default machine-only, matching the paper)")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Fixed attack-sampling seed (keep 42 so attacked sets match across training seeds)")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--no-feature-cache", action="store_true",
                        help="Disable the extract-once graph cache (parse every seed)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force re-parsing even when cache files exist")
    parser.add_argument("--purge-cache", action="store_true",
                        help="Delete this language's *_graphs_*.pt / *_extgraphs_*.pt caches first")
    parser.add_argument("--keep-cache", action="store_true",
                        help="Attack stage only: keep existing caches instead of purging them")
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
    # Attack-only runs are the cross-model comparison path, so a stale graph
    # cache is the one silent contaminant I purge by default. --keep-cache
    # opts out; the full stage only purges when asked.
    purge_cache = bool(args.purge_cache) or (args.stage == "attack" and not args.keep_cache)

    for lang in langs:
        if args.stage == "attack":
            out_csv = f"legacy_{lang}_attack_{args.threshold_mode}.csv"
            summary_csv = f"legacy_{lang}_attack_{args.threshold_mode}_summary.csv"
        else:
            out_csv = f"five_seed_{lang}_{variant}.csv"
            summary_csv = f"five_seed_{lang}_{variant}_summary.csv"
        df_all = run_language(lang, seeds, adversarial=args.adversarial,
                              epochs=args.epochs, batch_size=args.batch_size,
                              threshold=args.threshold, base_seed=args.base_seed,
                              skip_external=args.skip_external,
                              out_csv=out_csv, resume=args.resume,
                              graphs_cache_dir=graphs_cache_dir,
                              rebuild_cache=args.rebuild_cache,
                              threshold_mode=args.threshold_mode,
                              stage=args.stage, target=args.target,
                              purge_cache=purge_cache)
        summary = summarize(df_all[df_all["Language"] == lang])
        summary.to_csv(summary_csv, index=False)

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
