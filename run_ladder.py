"""Lightweight component ladder — bottom-up model shrinking study.

The full semantic model is ~7.2M trainable parameters (4 gated RGCN layers
carry ~4.7M of that) plus a 768-d frozen CodeT5+ feature vector per sample.
The base paper's ablation shows semantic features alone nearly match the
full detector, and the legacy leave-one-out study (2026-09/10) showed some
graph components are nearly free to remove (attention pooling, mildly: JK,
virtual node, gate). This runner walks a fixed *bottom-up* ladder instead:
start from the smallest meaningful stack (type embedding + pooling + CodeT5+
+ classifier, zero message passing, no branches) and append exactly one
component per rung, so each rung's delta is the component's marginal value
at that point. One `full` rung at the end anchors the curve to the protocol
model.

Every rung trains clean-only with the exact 5-seed-protocol knobs (30
epochs, lr 5e-5, batch size per language, patience 30, noise-free sched),
evaluates the held-out test split twice — at the checkpoint's val-calibrated
threshold (headline) and at a flat 0.50 (strict paper-comparable row) — and
writes one row per (rung, seed) to the output CSV, incrementally, so a
killed session still leaves usable artifacts. The bundle is loaded once and
shared by every rung (no re-parsing, no re-loading).

Usage:
  python -u run_ladder.py --language java                    # all rungs, seed 42
  python -u run_ladder.py --language java --rungs L0_semtype,L3_mp1
  python -u run_ladder.py --language cpp --seeds 42,43 --epochs 30
"""
import argparse
import csv
import gc
import time
import shutil

import torch
from torch_geometric.loader import DataLoader

from language_configs import CLEAN_CHECKPOINT, DEFAULT_BATCH_SIZE
from model import ablation_checkpoint_path, canonical_ablation, checkpoint_threshold
from pipeline import load_bundle_for_ablation, seed_everything
from train import train_single_model
from attack_utils import execute_model_eval_with_cost, print_detailed_metrics_with_cost

# name, ablation id, num_layers, description (what this rung adds vs the one above)
LADDER = (
    ("L0_semtype", "no-mp+no-subword+no-struct+mean-pool+no-film", 4,
     "base: type-emb + mean/max pool + CodeT5+ + classifier (0 MP layers, no branches)"),
    ("L1_struct", "no-mp+no-subword+mean-pool+no-film", 4,
     "+ 28-d per-node stylometry (struct features)"),
    ("L2_subword", "no-mp+mean-pool+no-film", 4,
     "+ 128-d BPE subword branch"),
    ("L3_mp1", "mean-pool+no-film", 1,
     "+ 1 gated RGCN layer (JK vacuous at 1 layer)"),
    ("L4_mp2", "mean-pool+no-film", 2,
     "+ 2nd gated RGCN layer, 2-scale JK concat"),
    ("L4b_nojk", "no-jk+mean-pool+no-film", 2,
     "2 layers, last-layer readout only (isolates JK vs L4_mp2)"),
    ("L5_film", "mean-pool", 2,
     "+ FiLM macro-stat gate"),
    ("L6_attn", "full", 2,
     "+ attention pooling (full component set at 2 layers)"),
    ("L7_full4", "full", 4,
     "reference: 4-layer protocol model"),
)

FIELDS = (
    "language", "rung", "seed", "ablation", "num_layers", "description",
    "params", "params_m", "best_val_f1", "best_val_roc", "threshold",
    "train_min", "mean_epoch_s", "peak_vram_mb",
    "acc", "prec", "rec", "f1", "roc", "fpr",
    "acc_at_050", "prec_at_050", "rec_at_050", "f1_at_050", "fpr_at_050",
    "latency_ms", "throughput", "peak_ram_mb", "n_test", "ckpt_copy",
)


def _row(args, name, desc, ablation, layers, seed, params, best_f1, best_roc,
         thr, train_min, peak_vram, res, res50, ckpt_copy):
    canon = canonical_ablation(ablation)
    return {
        "language": args.language, "rung": name, "seed": seed,
        "ablation": canon, "num_layers": layers, "description": desc,
        "params": params, "params_m": round(params / 1e6, 3),
        "best_val_f1": round(best_f1, 6), "best_val_roc": round(best_roc, 6),
        "threshold": round(thr, 6),
        "train_min": round(train_min, 2),
        "mean_epoch_s": round(train_min * 60.0 / max(1, args.epochs), 2),
        "peak_vram_mb": round(peak_vram, 1),
        "acc": round(res["Acc"], 6), "prec": round(res["Prec"], 6),
        "rec": round(res["Rec"], 6), "f1": round(res["F1"], 6),
        "roc": round(res["ROC"], 6), "fpr": round(res["FPR"], 6),
        "acc_at_050": round(res50["Acc"], 6), "prec_at_050": round(res50["Prec"], 6),
        "rec_at_050": round(res50["Rec"], 6), "f1_at_050": round(res50["F1"], 6),
        "fpr_at_050": round(res50["FPR"], 6),
        "latency_ms": round(res["Latency_ms"], 4),
        "throughput": round(res["Throughput"], 3),
        "peak_ram_mb": round(res.get("PeakRAM_MB", 0.0), 2),
        "n_test": res["N"], "ckpt_copy": ckpt_copy,
    }


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_table(rows):
    header = (f"{'rung':<10} {'layers':>6} {'params_M':>9} {'thr':>7} "
              f"{'valF1':>7} {'F1':>7} {'ROC':>7} {'F1@0.50':>8} {'ep_s':>6}")
    print("\n" + "=" * len(header))
    print("LADDER SUMMARY (deployed threshold = val-calibrated per rung)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['rung']:<10} {r['num_layers']:>6} {r['params_m']:>9.3f} "
              f"{r['threshold']:>7.4f} {r['best_val_f1']:>7.4f} {r['f1']:>7.4f} "
              f"{r['roc']:>7.4f} {r['f1_at_050']:>8.4f} {r['mean_epoch_s']:>6.1f}")
    print("=" * len(header), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Lightweight component ladder runner")
    parser.add_argument("--language", type=str, default="java", choices=["python", "java", "cpp"])
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated seeds, e.g. 42 or 42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--rungs", type=str, default="all",
                        help="'all' or comma-separated rung names from the LADDER list")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path (default: ladder_<language>_summary.csv)")
    parser.add_argument("--test-every", type=int, default=5,
                        help="Periodic test interval during training (0 disables)")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    wanted = None if args.rungs.strip().lower() == "all" else {
        r.strip() for r in args.rungs.split(",") if r.strip()}
    rungs = [r for r in LADDER if wanted is None or r[0] in wanted]
    if not rungs:
        raise SystemExit(f"No rungs selected; available: {[r[0] for r in LADDER]}")
    out_path = args.out or f"ladder_{args.language}_summary.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE[args.language]
    print(f"[i] Ladder: language={args.language} seeds={seeds} epochs={args.epochs} "
          f"batch={batch_size} device={device}", flush=True)
    for name, ablation, layers, desc in rungs:
        print(f"    - {name:<10} ablation={canonical_ablation(ablation):<45} "
              f"layers={layers}  {desc}", flush=True)

    bundle = load_bundle_for_ablation(args.language, "full")
    ctx = bundle["ctx"]
    train_g = bundle["train_clean_graphs"]
    val_g = bundle["val_graphs"]
    test_g = bundle["test_graphs"]
    has_sem = bool(getattr(test_g[0], "sem", None) is not None) if test_g else False
    print(f"[i] Bundle: train={len(train_g)} val={len(val_g)} test={len(test_g)} "
          f"semantic={'yes (768-d)' if has_sem else 'no'}", flush=True)
    test_loader = DataLoader(test_g, batch_size=batch_size, shuffle=False)

    rows = []
    _write_csv(out_path, rows)
    for name, ablation, layers, desc in rungs:
        for seed in seeds:
            print("\n" + "=" * 100)
            print(f"RUNG {name} | ablation={canonical_ablation(ablation)} | "
                  f"num_layers={layers} | seed={seed}", flush=True)
            print(f"     {desc}", flush=True)
            print("=" * 100, flush=True)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            seed_everything(seed)
            t0 = time.perf_counter()
            model, best_f1, best_roc = train_single_model(
                train_g, val_g, ctx, device, batch_size,
                epochs=args.epochs, patience=args.epochs, adversarial=False,
                language=args.language, ablation=ablation, num_layers=layers,
                test_graphs=test_g, test_every=args.test_every)
            train_min = (time.perf_counter() - t0) / 60.0
            params = sum(p.numel() for p in model.parameters())
            peak_vram = (torch.cuda.max_memory_allocated() / (1024 * 1024)
                         if torch.cuda.is_available() else 0.0)

            canon = canonical_ablation(ablation)
            ckpt_src = (ablation_checkpoint_path(args.language, canon)
                        if canon != "full" else CLEAN_CHECKPOINT[args.language])
            thr = checkpoint_threshold(ckpt_src, device="cpu", default=0.50)
            ckpt_copy = f"ladder_{args.language}_{name}_s{seed}.pth"
            shutil.copyfile(ckpt_src, ckpt_copy)

            res = execute_model_eval_with_cost(model, test_loader, device, threshold=thr)
            res50 = execute_model_eval_with_cost(model, test_loader, device, threshold=0.50)
            print(f"\n[{name} seed {seed}] params={params:,} ({params/1e6:.3f}M) | "
                  f"train {train_min:.1f} min | deployed thr {thr:.4f}", flush=True)
            print_detailed_metrics_with_cost(f"{args.language.upper()} ladder {name} @thr",
                                             res)
            print_detailed_metrics_with_cost(f"{args.language.upper()} ladder {name} @0.50",
                                             res50)

            rows.append(_row(args, name, desc, ablation, layers, seed, params,
                             best_f1, best_roc, thr, train_min, peak_vram,
                             res, res50, ckpt_copy))
            _write_csv(out_path, rows)
            print(f"[i] {out_path} updated ({len(rows)} row(s)); "
                  f"checkpoint kept as {ckpt_copy}", flush=True)

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _print_table(rows)
    print(f"\n[i] DONE. Artifacts: {out_path} + ladder_{args.language}_*.pth", flush=True)


if __name__ == "__main__":
    main()
