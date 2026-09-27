"""Shared data pipeline: load -> balance -> augment -> vocab -> graphs -> normalize.

Mirrors Cell 1 of each notebook exactly (prints, descs, test-balancing rule).
`main.py` calls `prepare_graphs()` once and persists the result with
`save_bundle()`; every downstream script (`train.py`, `evaluate.py`,
attack/external/audit) loads it with `load_bundle()` so training and
evaluation never rebuild CPGs. Sequential usage:

  python main.py --language python [--adversarial]
  python train.py --language python [--adversarial]
  python evaluate.py --language python [--adversarial]
"""
import os
import random

import numpy as np
import torch

from data_loader import balanced_subset, load_magecode_splits
from language_configs import SEED, get_parser
from graph_builder import (
    GraphContext,
    fit_vocab_and_tokenizer,
    process_split,
    fit_normalization,
    apply_normalization,
    CONTINUOUS_IDX,
)
from augment import generate_adversarial_augmentations


def seed_everything(seed=SEED):
    """Reproduce notebook Cell-1 RNG state. Call at the start of any process
    that consumes RNG (training shuffling/dropout/masking, augmentation)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _seed_everything():
    seed_everything()


SPLIT_DESC = {
    "python": ("Parsing Clean Baseline Python Graphs", "Parsing Augmented Python Train Graphs",
               "Parsing Validation Python Graphs", "Parsing Full Python Test Graphs"),
    "cpp": ("Parsing Clean Baseline C++ Graphs", "Parsing Augmented C++ Train Graphs",
            "Parsing Validation C++ Graphs", "Parsing Rebalanced C++ Test Graphs"),
    "java": ("Parsing Clean Baseline Train Graphs", "Parsing Augmented Train Graphs",
             "Parsing Validation Graphs", "Parsing Full Java Test Graphs"),
}

NORM_MSG = {
    "python": "[✓] Python Feature normalization locked using Clean Baseline statistics.",
    "cpp": "[✓] C++ Feature normalization locked using Clean Baseline statistics.",
    "java": "[✓] Feature normalization locked using Clean Baseline statistics.",
}

LOADED_MSG = {
    "python": "Loaded Python Splits: Train={t}, Val={v}, Test={te}",
    "java": "Loaded Java Splits: Train={t}, Val={v}, Test={te}",
    "cpp": "Loaded C++ Splits: Train={t}, Val={v}, Test={te}",
}


def prepare_graphs(language="python", trial_samples=None, limit=None, adversarial=False, seed=SEED):
    """Full Cell-1 pipeline. `limit` caps balanced splits for quick debugging (None = notebook exact).

    The adversarial pool (augmentation + adv graph parsing, ~40% of Cell-1
    cost) is built ONLY when `adversarial=True`, mirroring
    `main.py --adversarial`. Without the flag, `train_adv_*` entries are None
    and only clean/val/test splits are normalized (normalization is always
    fit on clean, so those outputs are bit-identical either way).

    I seed model RNG from `seed` here but keep data splits locked at SEED
    below, because I want 5-seed averages to compare identical rows with
    only init/shuffle/dropout varying.
    """
    seed_everything(seed)
    print(f"Loading full {language.capitalize()} dataset from Hugging Face..."
          if language != "cpp" else "Loading full C++ dataset from Hugging Face...")

    train_data, val_data, test_data = load_magecode_splits(language, trial_samples=trial_samples)
    print(LOADED_MSG[language].format(t=len(train_data), v=len(val_data), te=len(test_data)))

    parser, _ = get_parser(language)

    # Data splits stay locked at SEED even when seed varies (see docstring).
    train_clean_data = balanced_subset(train_data, SEED)
    val_eval_data = balanced_subset(val_data, SEED + 1)

    if limit is not None:
        train_clean_data = train_clean_data.select(range(min(limit, len(train_clean_data))))
        val_eval_data = val_eval_data.select(range(min(limit, len(val_eval_data))))

    if adversarial:
        train_adv_data = generate_adversarial_augmentations(train_clean_data, language, parser)
    else:
        train_adv_data = None

    type_to_id, vocab_size, bpe_tokenizer, pad_id, token_to_rank, max_rank = fit_vocab_and_tokenizer(
        train_clean_data, parser, language)
    ctx = GraphContext(language, parser, type_to_id, vocab_size, bpe_tokenizer, pad_id, token_to_rank, max_rank)

    d_clean, d_adv, d_val, d_test = SPLIT_DESC[language]
    train_clean_graphs = process_split(train_clean_data, d_clean, ctx)
    if adversarial:
        train_adv_graphs = process_split(train_adv_data, d_adv, ctx)
    else:
        train_adv_graphs = None
    val_graphs = process_split(val_eval_data, d_val, ctx)

    if language == "cpp":
        # Balanced Test set extraction to eliminate base-rate skew (C++ only)
        test_balanced_raw = balanced_subset(test_data, seed=42)
        if limit is not None:
            test_balanced_raw = test_balanced_raw.select(range(min(limit, len(test_balanced_raw))))
        test_graphs = process_split(test_balanced_raw, d_test, ctx)
    else:
        # Full test set, no balancing (python/java)
        test_balanced_raw = test_data
        if limit is not None:
            test_balanced_raw = test_balanced_raw.select(range(min(limit, len(test_balanced_raw))))
        test_graphs = process_split(test_balanced_raw, d_test, ctx)

    fit_normalization(train_clean_graphs, ctx)
    built_splits = [s for s in (train_clean_graphs, train_adv_graphs, val_graphs, test_graphs) if s is not None]
    for split in built_splits:
        apply_normalization(split, ctx)

    print(NORM_MSG[language])
    return {
        "ctx": ctx,
        "parser": parser,
        "train_clean_data": train_clean_data,
        "train_adv_data": train_adv_data,
        "val_eval_data": val_eval_data,
        "test_raw": test_balanced_raw,
        "test_full": test_data,
        "train_clean_graphs": train_clean_graphs,
        "train_adv_graphs": train_adv_graphs,
        "val_graphs": val_graphs,
        "test_graphs": test_graphs,
    }


# ---------------------------------------------------------------------------
# Bundle persistence: main.py saves once, every other script loads.
# Raw HF rows are NOT cached (cheap to reload, no parsing involved);
# use load_test_raw_rows() / load_audit_raw_rows() for those.
# ---------------------------------------------------------------------------

def bundle_path(language):
    return f"{language}_cpg_bundle.pt"


def save_bundle(bundle, trial_samples=None, limit=None, path=None):
    ctx = bundle["ctx"]
    language = ctx.language
    path = path or bundle_path(language)
    payload = {
        "language": language,
        "trial_samples": trial_samples,
        "limit": limit,
        "type_to_id": ctx.type_to_id,
        "vocab_size": ctx.vocab_size,
        "tokenizer_str": ctx.bpe_tokenizer.to_str(),
        "pad_id": ctx.pad_id,
        "token_to_rank": ctx.token_to_rank,
        "max_rank": ctx.max_rank,
        "means": ctx.means,
        "stds": ctx.stds,
        "g_means": ctx.g_means,
        "g_stds": ctx.g_stds,
        "has_adv": bundle.get("train_adv_graphs") is not None,
        "train_clean_graphs": bundle["train_clean_graphs"],
        "train_adv_graphs": bundle.get("train_adv_graphs"),
        "val_graphs": bundle["val_graphs"],
        "test_graphs": bundle["test_graphs"],
    }
    torch.save(payload, path)
    print(f"[+] CPG bundle saved at: {path} (adv graphs: {'yes' if payload['has_adv'] else 'no'})")
    return path


def load_bundle(language, path=None):
    from tokenizers import Tokenizer
    path = path or bundle_path(language)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Bundle not found: {path}. Run `python main.py --language {language}` first"
            " (add --adversarial if you need the adv pool).")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6 without the weights_only kwarg
        payload = torch.load(path, map_location="cpu")
    if payload.get("language", language) != language:
        raise ValueError(f"Bundle language mismatch: file holds '{payload.get('language')}', requested '{language}'.")
    # I check dims here because stale 34-d/16-d bundles fail later with
    # cryptic matmul/load_state_dict errors; rebuilding is the fix.
    _probe = (payload.get("train_clean_graphs") or payload.get("val_graphs") or payload.get("test_graphs") or [None])[0]
    if _probe is not None:
        try:
            _sd, _gd = int(_probe.x_struct.shape[1]), int(_probe.global_stats.shape[1])
        except Exception:
            _sd, _gd = -1, -1
        if _sd != 28 or _gd != 37:
            raise ValueError(
                f"Stale bundle dims in {path}: x_struct={_sd}, global_stats={_gd}; "
                f"expected 28/37 after the feature overhaul. Rerun `python main.py --language {language}` "
                f"(same trial_samples/limit) and retrain before eval/attacks.")

    parser, _ = get_parser(language)
    ctx = GraphContext(
        language,
        parser,
        payload["type_to_id"],
        payload["vocab_size"],
        Tokenizer.from_str(payload["tokenizer_str"]),
        payload["pad_id"],
        payload["token_to_rank"],
        payload["max_rank"],
    )
    ctx.means, ctx.stds, ctx.g_means, ctx.g_stds = (
        payload["means"], payload["stds"], payload["g_means"], payload["g_stds"])
    return {
        "ctx": ctx,
        "parser": parser,
        "trial_samples": payload.get("trial_samples"),
        "limit": payload.get("limit"),
        "has_adv": payload.get("has_adv", payload.get("train_adv_graphs") is not None),
        "train_clean_graphs": payload["train_clean_graphs"],
        "train_adv_graphs": payload.get("train_adv_graphs"),
        "val_graphs": payload["val_graphs"],
        "test_graphs": payload["test_graphs"],
    }


def require_adv_graphs(bundle, language):
    if bundle.get("train_adv_graphs") is None:
        raise RuntimeError(
            f"Adv graphs missing for '{language}'. Rerun `python main.py --language {language} --adversarial` first.")
    return bundle["train_adv_graphs"]


# ---------------------------------------------------------------------------
# Ablation graph variants: extract once, derive forever.
#
# main.py parses each language ONCE (standard bundle). The two graph-changing
# ablations derive their graphs in-memory from that bundle: no re-parsing,
# so test-set extraction (balanced cpp / full python-java) is never repeated.
# Normalization is refit per variant on the derived train_clean graphs and
# stored on a fresh ctx, so each variant is self-contained and deterministic.
# ---------------------------------------------------------------------------

def _denormalize_graphs(graphs, ctx):
    """Exactly invert apply_normalization (float roundtrip ~1 ulp, deterministic)."""
    out = []
    for g in graphs:
        g2 = g.clone()
        g2.x_struct[:, CONTINUOUS_IDX] = g2.x_struct[:, CONTINUOUS_IDX] * ctx.stds + ctx.means
        g2.global_stats = g2.global_stats * ctx.g_stds + ctx.g_means
        out.append(g2)
    return out


def _filter_graph_edges(graphs, keep_types):
    """Return new graphs keeping only edge types in keep_types (nodes untouched)."""
    from torch_geometric.data import Data
    keep = set(keep_types)
    out = []
    for g in graphs:
        mask = torch.tensor([int(t) in keep for t in g.edge_attr.tolist()], dtype=torch.bool)
        out.append(Data(
            x_type=g.x_type.clone(), x_struct=g.x_struct.clone(), x_subwords=g.x_subwords.clone(),
            edge_index=g.edge_index[:, mask].clone(), edge_attr=g.edge_attr[mask].clone(),
            y=g.y.clone(), num_nodes=int(g.num_nodes),
            virtual_idx=int(g.virtual_idx), global_stats=g.global_stats.clone(),
        ))
    return out


def _strip_virtual_node(graphs):
    """Return new graphs without the supernode (last node) and its type-8 edges."""
    from torch_geometric.data import Data
    out = []
    for g in graphs:
        n = int(g.num_nodes)
        v = int(g.virtual_idx)
        assert v == n - 1, f"Virtual node must be last (got virtual_idx={v}, num_nodes={n})"
        keep_edge = ((g.edge_index[0] != v) & (g.edge_index[1] != v) & (g.edge_attr != 8))
        assert bool(keep_edge.any()), "no-virtual variant left a graph with zero edges"
        out.append(Data(
            x_type=g.x_type[:-1].clone(), x_struct=g.x_struct[:-1].clone(),
            x_subwords=g.x_subwords[:-1].clone(),
            edge_index=g.edge_index[:, keep_edge].clone(), edge_attr=g.edge_attr[keep_edge].clone(),
            y=g.y.clone(), num_nodes=n - 1,
            virtual_idx=-1, global_stats=g.global_stats.clone(),
        ))
    return out


def derive_variant_bundle(bundle, variant):
    """Derive a graph-variant bundle from a loaded standard bundle.

    Returns a NEW bundle dict (fresh ctx, refit normalization); the input
    bundle is never mutated. variant="standard" returns the input unchanged.
    """
    from model import GRAPH_VARIANTS
    if variant not in GRAPH_VARIANTS:
        raise ValueError(f"Unknown graph variant '{variant}'. Choose from {list(GRAPH_VARIANTS)}.")
    if variant == "standard":
        return bundle
    ctx = bundle["ctx"]
    train = _denormalize_graphs(bundle["train_clean_graphs"], ctx)
    val = _denormalize_graphs(bundle["val_graphs"], ctx)
    test = _denormalize_graphs(bundle["test_graphs"], ctx)
    if variant == "syntax-only":
        from model import SYNTAX_ONLY_EDGE_TYPES
        train, val, test = (_filter_graph_edges(s, SYNTAX_ONLY_EDGE_TYPES) for s in (train, val, test))
    elif variant == "no-virtual":
        train, val, test = (_strip_virtual_node(s) for s in (train, val, test))
    new_ctx = GraphContext(ctx.language, ctx.parser, ctx.type_to_id, ctx.vocab_size,
                           ctx.bpe_tokenizer, ctx.pad_id, ctx.token_to_rank, ctx.max_rank)
    fit_normalization(train, new_ctx)
    apply_normalization(train, new_ctx)
    apply_normalization(val, new_ctx)
    apply_normalization(test, new_ctx)
    print(f"[✓] Graph variant '{variant}': derived train={len(train)} val={len(val)} "
          f"test={len(test)} from the cached standard bundle (no re-parsing).")
    return {
        "ctx": new_ctx,
        "parser": bundle["parser"],
        "trial_samples": bundle.get("trial_samples"),
        "limit": bundle.get("limit"),
        "has_adv": False,
        "train_clean_graphs": train,
        "train_adv_graphs": None,
        "val_graphs": val,
        "test_graphs": test,
    }


def load_bundle_for_ablation(language, ablation, path=None):
    """Load the standard bundle and derive the graphs an ablation id needs.

    Parsing/extraction happens exactly once per language (via main.py); all
    ten configs reuse it. The returned bundle is ready to train/evaluate.
    """
    from model import graph_variant_for_ablation
    bundle = load_bundle(language, path=path)
    return derive_variant_bundle(bundle, graph_variant_for_ablation(ablation))


def load_test_raw_rows(language):
    """Reload test raw rows (no graph building): full set except C++ balanced."""
    _, _, test_data = load_magecode_splits(language)
    if language == "cpp":
        # Balanced Test set extraction to eliminate base-rate skew
        return balanced_subset(test_data, seed=42)
    return test_data


def load_audit_raw_rows(language, trial_samples=None, limit=None, seed=SEED):
    """Reload raw rows for the leakage audit (no graph building).

    Mirrors the raw-data half of prepare_graphs, including the 20%+20%
    adversarial augmentation when present in the saved bundle.
    """
    from augment import generate_adversarial_augmentations
    seed_everything(seed)
    train_data, val_data, test_data = load_magecode_splits(language, trial_samples=trial_samples)
    train_clean_data = balanced_subset(train_data, SEED)
    val_eval_data = balanced_subset(val_data, SEED + 1)
    if limit is not None:
        train_clean_data = train_clean_data.select(range(min(limit, len(train_clean_data))))
        val_eval_data = val_eval_data.select(range(min(limit, len(val_eval_data))))
    parser, _ = get_parser(language)
    train_adv_data = generate_adversarial_augmentations(train_clean_data, language, parser)
    if language == "cpp":
        test_raw = balanced_subset(test_data, seed=42)
        if limit is not None:
            test_raw = test_raw.select(range(min(limit, len(test_raw))))
    else:
        test_raw = test_data
        if limit is not None:
            test_raw = test_raw.select(range(min(limit, len(test_raw))))
    return {
        "train_clean_data": train_clean_data,
        "train_adv_data": train_adv_data,
        "val_eval_data": val_eval_data,
        "test_raw": test_raw,
    }


def build_adv_only(language, trial_samples=None, limit=None, seed=SEED):
    """Build ONLY the adversarial pool + adv graphs, reusing a clean bundle.

    Requires `python main.py --language X` (same trial_samples/limit) to have
    run first: vocab, BPE tokenizer, and clean-locked normalization are reused
    untouched, so clean outputs stay bit-identical and adv outputs match the
    full-build path exactly. Only `train_adv_data` is parsed here.
    Data rows stay locked at SEED; `seed` only drives augmentation sampling.
    """
    from augment import generate_adversarial_augmentations
    seed_everything(seed)

    bundle = load_bundle(language)
    if (bundle.get("trial_samples"), bundle.get("limit")) != (trial_samples, limit):
        raise ValueError(
            f"Bundle params mismatch for '{language}': bundle was built with "
            f"trial_samples={bundle.get('trial_samples')}, limit={bundle.get('limit')} "
            f"but requested trial_samples={trial_samples}, limit={limit}. "
            f"Rerun `python main.py --language {language}` with matching params first.")
    ctx = bundle["ctx"]
    parser = bundle["parser"]

    train_data, _, _ = load_magecode_splits(language, trial_samples=trial_samples)
    train_clean_data = balanced_subset(train_data, SEED)
    if limit is not None:
        train_clean_data = train_clean_data.select(range(min(limit, len(train_clean_data))))

    train_adv_data = generate_adversarial_augmentations(train_clean_data, language, parser)
    _, d_adv, _, _ = SPLIT_DESC[language]
    train_adv_graphs = process_split(train_adv_data, d_adv, ctx)
    apply_normalization(train_adv_graphs, ctx)

    bundle["train_adv_graphs"] = train_adv_graphs
    bundle["has_adv"] = True
    path = save_bundle(bundle, trial_samples=trial_samples, limit=limit)
    print(f"Adv-only extraction complete: adv={len(train_adv_graphs)} -> {path}")
    return bundle
