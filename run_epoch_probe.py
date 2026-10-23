"""Epoch-by-epoch training-budget and checkpoint-selection probe (diagnostic).

Trains selected ladder rungs for a fixed budget (default 50 epochs) and, after
EVERY epoch, evaluates the CURRENT weights on the complete held-out test split
exclusively at the validation-calibrated optimal threshold.
This is a read-only diagnostic: test metrics never affect gradients, the
optimizer/scheduler, checkpoint selection, early stopping or thresholds inside
the run. They exist to answer two questions after the fact:

  1. When does test performance actually saturate? (per-epoch curves)
  2. Is best-validation-F1 checkpoint selection throwing away later progress,
     or would the final-epoch weights be as good or better?

Three selection rules that do not need test labels are compared at the end:
  * best val F1   (current protocol rule)
  * best val ROC
  * final epoch
Plus two diagnostic upper bounds (NOT usable selection rules):
  * oracle best test F1 over epochs
  * oracle best test ROC over epochs

Everything else stays at the 5-seed-protocol values; `--epochs` only changes
the loop length (lr 5e-5, wd 1e-3, T_max=45, eta_min=1e-6, grad clip 1.0,
batch size per language, patience = epochs so nothing early-stops).

Outputs (in the repo working directory):
  epoch_probe_<lang>.csv                     one row per (rung, seed, epoch)
  epoch_probe_<lang>_live.csv                current model's rows, rewritten each epoch
  epoch_probe_<lang>_summary.csv             one row per (rung, seed, candidate)
  epoch_probe_<lang>_multiseed_summary.csv   aggregated mean +/- std across seeds
  probe_<lang>_<rung>_s<seed>_bestvalf1.pth  copy of the standard checkpoint
  probe_<lang>_<rung>_s<seed>_bestvalroc.pth raw state dict (diagnostic)
  probe_<lang>_<rung>_s<seed>_final.pth      raw state dict (diagnostic)

Usage:
  python -u run_epoch_probe.py --language cpp --rungs L4_mp2 --seeds 42-46 --epochs 50
  python -u run_epoch_probe.py --language java --rungs L4_mp2 --seeds 42 --resume
"""
import argparse
import contextlib
import csv
import gc
import io
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from language_configs import DEFAULT_BATCH_SIZE
from model import (canonical_ablation, checkpoint_path_for_config, DEFAULT_ABLATION)
from pipeline import load_bundle_for_ablation, seed_everything
from run_ladder import LADDER
from train import _select_threshold, train_single_model

EPOCH_FIELDS = (
    "language", "rung", "seed", "epoch", "epochs_total",
    "ablation", "num_layers", "params", "params_m",
    "train_loss", "val_f1", "val_roc", "threshold",
    "test_acc", "test_prec", "test_rec", "test_f1", "test_roc", "test_fpr",
    "epoch_seconds",
)

SUMMARY_FIELDS = (
    "language", "rung", "seed", "candidate", "diagnostic", "epoch",
    "val_f1", "val_roc", "threshold",
    "test_acc", "test_prec", "test_rec", "test_f1", "test_roc", "test_fpr",
    "params", "params_m", "train_min",
)

CANDIDATES = ("best_val_f1", "best_val_roc", "final_epoch",
              "oracle_test_f1", "oracle_test_roc")


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
    return sorted(set(seeds))


def _rung_index(name):
    for i, (rung_name, _a, _l, _d) in enumerate(LADDER):
        if rung_name == name:
            return i
    return len(LADDER)


def _read_rows(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_rows(path, fields, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _merge_rows(path, fields, rows, key_fields, sort_key):
    def _key(row):
        return tuple(str(row[k]) for k in key_fields)

    merged = {_key(r): r for r in _read_rows(path)}
    for row in rows:
        merged[_key(row)] = row
    _write_rows(path, fields, sorted(merged.values(), key=sort_key))


def _completed_pairs(epoch_csv, epochs):
    """(rung, seed) pairs whose per-epoch rows already reach `epochs`."""
    seen = {}
    for row in _read_rows(epoch_csv):
        if int(row["epochs_total"]) != epochs:
            continue
        key = (row["rung"], int(row["seed"]))
        seen[key] = max(seen.get(key, 0), int(row["epoch"]))
    return {key for key, last_epoch in seen.items() if last_epoch >= epochs}


def _probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probs.append(torch.sigmoid(model(batch)).cpu().numpy().ravel())
            labels.append(batch.y.cpu().numpy().ravel())
    return np.concatenate(probs), np.concatenate(labels)


def _auc(probs, labels):
    return float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0


def _metrics_at(probs, labels, thr):
    pred = (probs >= float(thr)).astype(np.int64)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {
        "acc": (tp + tn) / max(1, len(labels)),
        "prec": prec,
        "rec": rec,
        "f1": 2.0 * prec * rec / max(1e-12, prec + rec),
        "fpr": fp / max(1, fp + tn),
    }


def _snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _quiet_select(probs, labels, prior):
    """Same val-calibrated rule, without its per-call diagnostic print."""
    with contextlib.redirect_stdout(io.StringIO()):
        return _select_threshold(probs, labels, fallback=0.50, prior=prior)


def _rounded(value, ndigits=6):
    return round(float(value), ndigits)


def _epoch_row(lang, rung, seed, epoch, epochs, ablation, layers, params,
               train_loss, val_f1, val_roc, threshold, test_metrics, test_roc,
               epoch_seconds):
    return {
        "language": lang, "rung": rung, "seed": seed,
        "epoch": epoch, "epochs_total": epochs,
        "ablation": canonical_ablation(ablation), "num_layers": layers,
        "params": params, "params_m": _rounded(params / 1e6, 3),
        "train_loss": _rounded(train_loss),
        "val_f1": _rounded(val_f1), "val_roc": _rounded(val_roc),
        "threshold": _rounded(threshold),
        "test_acc": _rounded(test_metrics["acc"]),
        "test_prec": _rounded(test_metrics["prec"]),
        "test_rec": _rounded(test_metrics["rec"]),
        "test_f1": _rounded(test_metrics["f1"]),
        "test_roc": _rounded(test_roc),
        "test_fpr": _rounded(test_metrics["fpr"]),
        "epoch_seconds": round(float(epoch_seconds), 2),
    }


def _summary_row(lang, rung, seed, candidate, diagnostic, epoch, params,
                 train_min, val_f1, val_roc, threshold, test_metrics, test_roc):
    return {
        "language": lang, "rung": rung, "seed": seed,
        "candidate": candidate, "diagnostic": diagnostic, "epoch": epoch,
        "val_f1": _rounded(val_f1), "val_roc": _rounded(val_roc),
        "threshold": _rounded(threshold),
        "test_acc": _rounded(test_metrics["acc"]),
        "test_prec": _rounded(test_metrics["prec"]),
        "test_rec": _rounded(test_metrics["rec"]),
        "test_f1": _rounded(test_metrics["f1"]),
        "test_roc": _rounded(test_roc),
        "test_fpr": _rounded(test_metrics["fpr"]),
        "params": params, "params_m": _rounded(params / 1e6, 3),
        "train_min": round(float(train_min), 2),
    }


def _save_multiseed_summary(summary_csv_path, multiseed_out_path):
    if not os.path.exists(summary_csv_path):
        return
    df = pd.read_csv(summary_csv_path)
    if df.empty:
        return
    metric_cols = ["val_f1", "val_roc", "threshold", "test_acc", "test_prec", "test_rec", "test_f1", "test_roc", "test_fpr", "train_min"]
    records = []
    group_keys = ["language", "rung", "candidate", "diagnostic"]
    for keys, group in df.groupby(group_keys, sort=False):
        row = {
            "language": keys[0], "rung": keys[1], "candidate": keys[2], "diagnostic": keys[3],
            "seeds": len(group["seed"].unique()),
        }
        for col in metric_cols:
            if col in group:
                vals = group[col].dropna()
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        records.append(row)
    summary_df = pd.DataFrame(records)
    summary_df.to_csv(multiseed_out_path, index=False)
    print(f"[i] Multi-seed summary saved to {multiseed_out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Per-epoch test probe evaluated at best threshold (diagnostic)")
    parser.add_argument("--language", default="java", choices=["python", "java", "cpp"])
    parser.add_argument("--seeds", default="42-46",
                        help="Seed list, e.g. '42-46' or '42,43,44,45,46'")
    parser.add_argument("--rungs", default="L4_mp2",
                        help="'all', 'L4_mp2', or comma-separated rung names from run_ladder.LADDER")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training budget (default: 50 epochs)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--out", default=None,
                        help="Per-epoch CSV (default: epoch_probe_<language>.csv)")
    parser.add_argument("--summary-out", default=None,
                        help="Candidate-summary CSV (default: epoch_probe_<language>_summary.csv)")
    parser.add_argument("--live-out", default=None,
                        help="Current model's per-epoch CSV (default: epoch_probe_<language>_live.csv)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (rung, seed) models already complete for this epoch budget")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    wanted = None if args.rungs.strip().lower() == "all" else {
        r.strip() for r in args.rungs.split(",") if r.strip()}
    rungs = [r for r in LADDER if wanted is None or r[0] in wanted]
    if not rungs:
        raise SystemExit("No rungs selected; available: %s" % [r[0] for r in LADDER])

    lang = args.language
    out_path = args.out or ("epoch_probe_%s.csv" % lang)
    summary_path = args.summary_out or ("epoch_probe_%s_summary.csv" % lang)
    multiseed_path = "epoch_probe_%s_multiseed_summary.csv" % lang
    live_path = args.live_out or ("epoch_probe_%s_live.csv" % lang)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE[lang]

    print("[i] Epoch probe: language=%s seeds=%s epochs=%d batch=%d device=%s"
          % (lang, seeds, args.epochs, batch_size, device), flush=True)
    print("[i] Selection rules compared: best val F1 / best val ROC / final epoch "
          "(evaluated exclusively at validation-calibrated optimal threshold)", flush=True)
    print("[i] Live CSV (current model, rewritten each epoch): %s" % live_path, flush=True)
    for name, ablation, layers, desc in rungs:
        print("    - %-10s ablation=%-45s layers=%d  %s"
              % (name, canonical_ablation(ablation), layers, desc), flush=True)

    bundle = load_bundle_for_ablation(lang, DEFAULT_ABLATION)
    ctx = bundle["ctx"]
    train_g = bundle["train_clean_graphs"]
    val_g = bundle["val_graphs"]
    test_g = bundle["test_graphs"]
    has_sem = bool(getattr(test_g[0], "sem", None) is not None) if test_g else False
    eval_prior = (float(np.mean([float(g.y.reshape(-1)[0]) for g in test_g]))
                  if test_g else None)
    print("[i] Bundle: train=%d val=%d test=%d semantic=%s eval_prior=%s"
          % (len(train_g), len(val_g), len(test_g),
             "yes (768-d)" if has_sem else "no",
             "n/a" if eval_prior is None else "%.4f" % eval_prior), flush=True)

    test_loader = DataLoader(test_g, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_g, batch_size=batch_size, shuffle=False)

    done = _completed_pairs(out_path, args.epochs) if args.resume else set()
    if done:
        print("[i] Resume: %d model(s) already complete at %d epochs"
              % (len(done), args.epochs), flush=True)

    for name, ablation, layers, desc in rungs:
        for seed in seeds:
            if (name, seed) in done:
                print("[i] Skip %s seed %d (already complete)" % (name, seed), flush=True)
                continue

            print("\n" + "=" * 100)
            print("PROBE %s | ablation=%s | num_layers=%d | seed=%d | epochs=%d"
                  % (name, canonical_ablation(ablation), layers, seed, args.epochs))
            print("       %s" % desc)
            print("=" * 100, flush=True)
            _write_rows(live_path, EPOCH_FIELDS, [])  # header only; rewritten each epoch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            seed_everything(seed)

            state = {
                "params": 0,
                "best_f1": -1.0, "best_f1_epoch": None, "best_f1_state": None,
                "best_roc": -1.0, "best_roc_epoch": None, "best_roc_state": None,
                "final_state": None, "rows": [],
            }

            def probe_cb(epoch, epochs, model, train_loss, val_f1, val_roc,
                         val_probs, val_labels, epoch_seconds):
                if not state["params"]:
                    state["params"] = sum(p.numel() for p in model.parameters())
                test_probs, test_labels = _probs(model, test_loader, device)
                test_roc = _auc(test_probs, test_labels)
                val_thr = _quiet_select(np.asarray(val_probs), np.asarray(val_labels), eval_prior)
                mthr = _metrics_at(test_probs, test_labels, val_thr)
                state["rows"].append(_epoch_row(
                    lang, name, seed, epoch + 1, epochs, ablation, layers,
                    state["params"], train_loss, val_f1, val_roc, val_thr,
                    mthr, test_roc, epoch_seconds))
                _write_rows(live_path, EPOCH_FIELDS, state["rows"])
                print("    [probe %02d] test acc %.4f | P %.4f R %.4f F1 %.4f | ROC %.4f | thr %.4f | loss %.4f (%0.1fs)"
                      % (epoch + 1, mthr["acc"], mthr["prec"], mthr["rec"], mthr["f1"],
                         test_roc, val_thr, train_loss, epoch_seconds),
                      flush=True)
                if val_f1 > state["best_f1"]:
                    state["best_f1"] = float(val_f1)
                    state["best_f1_epoch"] = epoch + 1
                    state["best_f1_state"] = _snapshot(model)
                if val_roc > state["best_roc"]:
                    state["best_roc"] = float(val_roc)
                    state["best_roc_epoch"] = epoch + 1
                    state["best_roc_state"] = _snapshot(model)
                if epoch + 1 == epochs:
                    state["final_state"] = _snapshot(model)

            t0 = time.perf_counter()
            model, best_f1, best_roc = train_single_model(
                train_g, val_g, ctx, device, batch_size,
                epochs=args.epochs, patience=args.epochs, adversarial=False,
                language=lang, ablation=ablation, num_layers=layers,
                test_graphs=test_g, test_every=0, epoch_probe=probe_cb)
            train_min = (time.perf_counter() - t0) / 60.0
            params = state["params"] or sum(p.numel() for p in model.parameters())

            summary_rows = []
            for candidate, cand_epoch, cand_state, diagnostic in (
                    ("best_val_f1", state["best_f1_epoch"], state["best_f1_state"], "no"),
                    ("best_val_roc", state["best_roc_epoch"], state["best_roc_state"], "no"),
                    ("final_epoch", args.epochs, state["final_state"], "no")):
                if cand_state is None:
                    continue
                model.load_state_dict(cand_state)
                val_probs, val_labels = _probs(model, val_loader, device)
                val_thr = _quiet_select(val_probs, val_labels, eval_prior)
                test_probs, test_labels = _probs(model, test_loader, device)
                test_roc = _auc(test_probs, test_labels)
                mthr = _metrics_at(test_probs, test_labels, val_thr)
                row = _summary_row(
                    lang, name, seed, candidate, diagnostic, cand_epoch, params,
                    train_min, 0.0, 0.0, val_thr, mthr, test_roc)
                matching = next((r for r in state["rows"] if r["epoch"] == cand_epoch), None)
                if matching is not None:
                    row["val_f1"] = matching["val_f1"]
                    row["val_roc"] = matching["val_roc"]
                    row["threshold"] = matching["threshold"]
                summary_rows.append(row)

            rows = state["rows"]
            oracle_f1 = max(rows, key=lambda r: (float(r["test_f1"]), -int(r["epoch"])))
            oracle_roc = max(rows, key=lambda r: float(r["test_roc"]))
            for candidate, row in (("oracle_test_f1", oracle_f1),
                                   ("oracle_test_roc", oracle_roc)):
                summary_rows.append(_summary_row(
                    lang, name, seed, candidate, "yes", int(row["epoch"]), params,
                    train_min, row["val_f1"], row["val_roc"], row["threshold"],
                    {"acc": row["test_acc"], "prec": row["test_prec"],
                     "rec": row["test_rec"], "f1": row["test_f1"],
                     "fpr": row["test_fpr"]},
                    row["test_roc"]))

            ckpt_src = checkpoint_path_for_config(lang, ablation=ablation, num_layers=layers)
            ckpt_copy = "probe_%s_%s_s%d_bestvalf1.pth" % (lang, name, seed)
            try:
                if os.path.exists(ckpt_src):
                    shutil.copyfile(ckpt_src, ckpt_copy)
            except OSError as exc:
                print("[!] Could not copy the standard checkpoint: %s" % exc, flush=True)
            if state["best_roc_state"] is not None:
                torch.save(state["best_roc_state"],
                           "probe_%s_%s_s%d_bestvalroc.pth" % (lang, name, seed))
            if state["final_state"] is not None:
                torch.save(state["final_state"],
                           "probe_%s_%s_s%d_final.pth" % (lang, name, seed))

            _merge_rows(out_path, EPOCH_FIELDS, rows,
                        ("language", "rung", "seed", "epoch"),
                        lambda r: (_rung_index(r["rung"]), int(r["seed"]),
                                   int(r["epoch"])))
            _merge_rows(summary_path, SUMMARY_FIELDS, summary_rows,
                        ("language", "rung", "seed", "candidate"),
                        lambda r: (_rung_index(r["rung"]), int(r["seed"]),
                                   CANDIDATES.index(r["candidate"])))

            def _cand(cand_name):
                return next(r for r in summary_rows if r["candidate"] == cand_name)
            bf1, broc, fin = _cand("best_val_f1"), _cand("best_val_roc"), _cand("final_epoch")
            orc = _cand("oracle_test_f1")
            print("\n[i] %s seed %d | params=%s | train %.1f min"
                  % (name, seed, "{:,}".format(params), train_min), flush=True)
            print("[i]   best val F1  ep %-3s  test acc %.4f  F1 %.4f  ROC %.4f  thr %.4f"
                  % (bf1["epoch"], bf1["test_acc"], bf1["test_f1"], bf1["test_roc"], bf1["threshold"]), flush=True)
            print("[i]   best val ROC ep %-3s  test acc %.4f  F1 %.4f  ROC %.4f  thr %.4f"
                  % (broc["epoch"], broc["test_acc"], broc["test_f1"], broc["test_roc"], broc["threshold"]), flush=True)
            print("[i]   final epoch  ep %-3s  test acc %.4f  F1 %.4f  ROC %.4f  thr %.4f"
                  % (fin["epoch"], fin["test_acc"], fin["test_f1"], fin["test_roc"], fin["threshold"]), flush=True)
            print("[i]   oracle test  ep %-3s  test acc %.4f  F1 %.4f  ROC %.4f"
                  % (orc["epoch"], orc["test_acc"], orc["test_f1"], orc["test_roc"]), flush=True)
            print("[i] %s updated (%d epoch rows); live: %s; summary: %s"
                  % (out_path, len(rows), live_path, summary_path), flush=True)

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(seeds) > 1:
        _save_multiseed_summary(summary_path, multiseed_path)

    print("\n[i] DONE. Per-epoch rows: %s | live: %s | candidate summary: %s"
          % (out_path, live_path, summary_path), flush=True)


if __name__ == "__main__":
    main()
