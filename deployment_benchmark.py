"""Raw-code end-to-end deployment benchmark for the CPG detector.

evaluate.py scores graphs that were parsed and featurized offline. This script
instead starts from raw test-set source files and times the complete path:
tree-sitter parse, CPG construction, graph normalization, optional CodeT5+
semantic encoding, and model inference.

Reports both a request-scoped per-file view (batch size 1) and a batched
throughput view. The token denominator is the shared tree-sitter leaf-token
definition (token_metrics.py), so files/sec and tokens/sec are directly
comparable with the hybrid and legacy folders.

Example:
  python deployment_benchmark.py --language python --seed 42 --max-files 200
"""
import argparse
import csv
import gc
import json
import os
import random
import time

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from data_loader import load_code_data
from graph_builder import apply_normalization, build_optimized_ast_graph
from language_configs import ADV_CHECKPOINT, CLEAN_CHECKPOINT
from model import build_encoder_from_checkpoint, checkpoint_threshold
from pipeline import load_bundle
from semantic_features import embed_graphs, get_encoder


def _default_batch(language):
    return 128 if language == "python" else (32 if language == "java" else 16)


def _summary(values):
    if not values:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None, "total_s": 0.0}
    vals = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(vals.mean()),
        "median_ms": float(np.median(vals)),
        "p95_ms": float(np.percentile(vals, 95)),
        "total_s": float(vals.sum() / 1000.0),
    }


def _checkpoint_path(language, seed, adversarial):
    canonical = (ADV_CHECKPOINT if adversarial else CLEAN_CHECKPOINT)[language]
    base, ext = os.path.splitext(canonical)
    seeded = f"{base}_seed{seed}{ext}"
    path = seeded if os.path.exists(seeded) else canonical
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Train the model or restore the archived per-seed checkpoint first.")
    return path


def _build_graph(code, label, ctx, has_semantic, attach_semantic, device=None):
    graph = build_optimized_ast_graph(code, label, ctx)
    if graph is None:
        return None
    apply_normalization([graph], ctx)
    if has_semantic and attach_semantic:
        vec = get_encoder(device=device).encode([code], show_progress=False)[0]
        graph.sem = torch.from_numpy(np.ascontiguousarray(vec, dtype=np.float32)).unsqueeze(0)
    return graph


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _infer_one(model, graph, device):
    batch = Batch.from_data_list([graph]).to(device)
    with torch.no_grad():
        return model(batch)


def _infer_many(model, graphs, device, batch_size):
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out.extend(model(batch).detach().cpu().numpy().flatten().tolist())
    return out


def _token_count(graph, code, language):
    if getattr(graph, "n_tokens", None) is not None:
        return int(graph.n_tokens.reshape(-1)[0].item())
    from token_metrics import count_code_tokens
    return int(count_code_tokens(code, language))


def run_benchmark(language="python", seed=42, max_files=200, batch_size=None,
                  warmup=5, adversarial=False, threshold=None, output=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size or _default_batch(language)
    print(f"CPG raw-code deployment benchmark | {language.upper()} | {device}")

    codes, _labels = load_code_data(language=language, split="test")
    if max_files and max_files < len(codes):
        indices = sorted(random.Random(seed).sample(range(len(codes)), int(max_files)))
        codes = [codes[i] for i in indices]
    if not codes:
        raise RuntimeError(f"No test files loaded for {language}.")

    bundle = load_bundle(language)
    ctx = bundle["ctx"]
    has_semantic = bool(bundle.get("has_semantic"))
    ckpt = _checkpoint_path(language, seed, adversarial)
    if threshold is None:
        threshold = checkpoint_threshold(ckpt, device)
    model, _ = build_encoder_from_checkpoint(ctx, ckpt, device)
    model.eval()
    print(f"Bundle graphs: {len(bundle.get('test_graphs') or [])} | checkpoint: {ckpt} | "
          f"threshold: {float(threshold):.4f} | semantic: {has_semantic}")

    # Warmup uses the first files; their timings are discarded.
    for code in codes[:max(0, min(int(warmup), len(codes)))]:
        g = _build_graph(code, 0, ctx, has_semantic, attach_semantic=True, device=device)
        if g is not None:
            _infer_one(model, g, device)
    _sync(device)

    per_file_rows = []
    preprocess_ms, inference_ms, total_ms = [], [], []
    per_file_tokens = 0
    skipped = 0
    for idx, code in enumerate(codes):
        _sync(device)
        t0 = time.perf_counter()
        graph = _build_graph(code, 0, ctx, has_semantic, attach_semantic=True, device=device)
        _sync(device)
        t1 = time.perf_counter()
        if graph is None:
            skipped += 1
            continue
        _infer_one(model, graph, device)
        _sync(device)
        t2 = time.perf_counter()
        n_tokens = _token_count(graph, code, language)
        per_file_tokens += n_tokens
        pre_ms, inf_ms, tot_ms = (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t2 - t0) * 1000.0
        preprocess_ms.append(pre_ms)
        inference_ms.append(inf_ms)
        total_ms.append(tot_ms)
        per_file_rows.append({
            "index": idx, "tokens": n_tokens,
            "preprocess_ms": round(pre_ms, 6),
            "inference_ms": round(inf_ms, 6),
            "total_ms": round(tot_ms, 6),
        })
    scored_files = len(per_file_rows)
    per_file_total_s = sum(total_ms) / 1000.0

    # Batched view: graph construction still happens file-by-file, but the
    # CodeT5+ encoder and GNN operate in batches.
    _sync(device)
    t0 = time.perf_counter()
    graphs, texts = [], []
    for code in codes:
        graph = _build_graph(code, 0, ctx, has_semantic, attach_semantic=False, device=device)
        if graph is not None:
            graphs.append(graph)
            texts.append(code)
    if has_semantic and graphs:
        embed_graphs(graphs, texts, device=device, show_progress=False)
    _sync(device)
    t_pre = time.perf_counter() - t0
    t1 = time.perf_counter()
    if graphs:
        _infer_many(model, graphs, device, batch_size)
    _sync(device)
    t_inf = time.perf_counter() - t1
    batched_total_s = t_pre + t_inf
    batch_tokens = int(sum(_token_count(g, c, language) for g, c in zip(graphs, texts)))

    payload = {
        "model": "cpg",
        "language": language,
        "seed": int(seed),
        "adversarial": bool(adversarial),
        "device": str(device),
        "checkpoint": ckpt,
        "threshold": float(threshold),
        "has_semantic": has_semantic,
        "n_files": scored_files,
        "skipped_files": skipped,
        "total_tokens": per_file_tokens,
        "tokens_per_file": per_file_tokens / max(1, scored_files),
        "warmup_files": int(min(max(0, warmup), len(codes))),
        "per_file": {
            "batch_size": 1,
            "preprocess": _summary(preprocess_ms),
            "inference": _summary(inference_ms),
            "end_to_end": _summary(total_ms),
            "files_per_sec": scored_files / max(1e-9, per_file_total_s),
            "tokens_per_sec": per_file_tokens / max(1e-9, per_file_total_s),
            "ms_per_token": 1000.0 * per_file_total_s / per_file_tokens
            if per_file_tokens else None,
        },
        "batched": {
            "batch_size": int(batch_size),
            "n_files": len(graphs),
            "total_tokens": batch_tokens,
            "preprocess_s": float(t_pre),
            "inference_s": float(t_inf),
            "end_to_end_s": float(batched_total_s),
            "files_per_sec": len(graphs) / max(1e-9, batched_total_s),
            "tokens_per_sec": batch_tokens / max(1e-9, batched_total_s),
            "ms_per_file": 1000.0 * batched_total_s / max(1, len(graphs)),
            "ms_per_token": 1000.0 * batched_total_s / batch_tokens if batch_tokens else None,
        },
    }
    out_json = output or f"deployment_cpg_{language}_seed{seed}.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    csv_path = os.path.splitext(out_json)[0] + "_per_file.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["index", "tokens", "preprocess_ms",
                                                "inference_ms", "total_ms"])
        writer.writeheader()
        writer.writerows(per_file_rows)

    print("\nPer-file (batch=1):")
    print(f"  end-to-end mean/median/p95: {payload['per_file']['end_to_end']['mean_ms']:.2f} / "
          f"{payload['per_file']['end_to_end']['median_ms']:.2f} / "
          f"{payload['per_file']['end_to_end']['p95_ms']:.2f} ms")
    print(f"  files/sec: {payload['per_file']['files_per_sec']:.2f} | "
          f"tokens/sec: {payload['per_file']['tokens_per_sec']:.2f} | "
          f"ms/token: {payload['per_file']['ms_per_token']:.4f}")
    print("\nBatched end-to-end:")
    print(f"  files/sec: {payload['batched']['files_per_sec']:.2f} | "
          f"tokens/sec: {payload['batched']['tokens_per_sec']:.2f} | "
          f"ms/file: {payload['batched']['ms_per_file']:.2f} | "
          f"ms/token: {payload['batched']['ms_per_token']:.4f}")
    if skipped:
        print(f"[!] Skipped {skipped} files whose graph could not be built.")
    print(f"\nWrote {out_json} and {csv_path}")

    del model, graphs, bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPG raw-code deployment benchmark")
    parser.add_argument("--language", default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=200,
                        help="Deterministic test-set sample size (0 = all files)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_benchmark(language=args.language, seed=args.seed, max_files=args.max_files,
                  batch_size=args.batch_size, warmup=args.warmup,
                  adversarial=args.adversarial, threshold=args.threshold,
                  output=args.output)
