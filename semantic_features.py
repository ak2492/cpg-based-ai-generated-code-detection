"""CodeT5+ semantic embeddings — the paper's dominant feature class.

The base paper's ablation (Table 6/7) shows a semantic-only CodeT5+ model
nearly matches the full detector, while the CPG model so far has zero
semantic input. This module precomputes ONE 768-d CLS embedding per sample
at CPG-bundle build time (same recipe as the hybrid implementation's
`semantic_extractor.py`: Salesforce/codet5p-220m, T5EncoderModel, CLS of
`last_hidden_state`, truncation at 512 tokens) and attaches it to each
PyG `Data` as `.sem` so the GNN classifier can fuse it with the graph
embedding.

Embeddings are computed inside `process_split` (the single
chokepoint for every parse path: main bundle, adv pool, attacks, external
suites), so every graph the model ever sees carries `.sem` — no separate
cache to keep in sync. Disable per process via `set_enabled(False)` or
the CPG_SEMANTIC=0 environment variable (e.g. `main.py --no-semantic`).
Inference is deterministic (eval mode, inference_mode -> no dropout) and
batched on GPU.
"""
import os

import numpy as np
import torch

MODEL_NAME = "Salesforce/codet5p-220m"
MAX_LEN = 512
EMB_DIM = 768

_ENABLED = None  # None -> read CPG_SEMANTIC lazily on first use


def semantic_enabled():
    """Whether process_split should attach CodeT5+ embeddings."""
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("CPG_SEMANTIC", "1").strip().lower() not in (
            "0", "false", "no", "off")
    return bool(_ENABLED)


def set_enabled(flag):
    """Explicit override (main.py --no-semantic); wins over the env var."""
    global _ENABLED
    _ENABLED = bool(flag)


class SemanticExtractor:
    """Batched CodeT5+ encoder; CLS last-hidden-state, exactly the hybrid recipe."""

    def __init__(self, device=None, model_name=MODEL_NAME, batch_size=64):
        from transformers import AutoTokenizer, T5EncoderModel
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as exc:  # noqa: BLE001 - version-specific tokenizer bugs
            # Some transformers releases reject this repo's legacy
            # tokenizer_config (special_tokens_map.json lists 100
            # `additional_special_tokens` as dicts). Loading the RoBERTa BPE
            # files directly yields identical ids (<s> CLS, same vocab); see
            # requirements.txt for the pinned, proven-good version.
            from transformers import RobertaTokenizer
            from huggingface_hub import hf_hub_download
            vocab = hf_hub_download(model_name, "vocab.json")
            merges = hf_hub_download(model_name, "merges.txt")
            self.tokenizer = RobertaTokenizer(vocab_file=vocab, merges_file=merges,
                                              errors="replace")
            print(f"[i] AutoTokenizer failed ({type(exc).__name__}: {exc}); "
                  f"loaded the RoBERTa BPE files directly instead.")
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device).eval()
        self.batch_size = batch_size

    @torch.inference_mode()
    def encode(self, texts, show_progress=False):
        """texts -> float32 array [N, 768]. Empty strings are encoded as a space
        so the tokenizer never yields zero-length sequences."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, EMB_DIM), dtype=np.float32)
        chunks = []
        starts = range(0, len(texts), self.batch_size)
        if show_progress:
            from tqdm import tqdm
            starts = tqdm(starts, desc="CodeT5+ embeddings")
        for start in starts:
            batch = [t if isinstance(t, str) and t.strip() else " "
                     for t in texts[start:start + self.batch_size]]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=MAX_LEN).to(self.device)
            out = self.model(**inputs)
            chunks.append(out.last_hidden_state[:, 0, :].float().cpu().numpy())
        return np.concatenate(chunks, axis=0)


_encoder = None


def get_encoder(device=None):
    """Process-wide singleton so repeated process_split calls reload nothing."""
    global _encoder
    if _encoder is None:
        _encoder = SemanticExtractor(device=device)
        print(f"[i] CodeT5+ semantic encoder loaded ({MODEL_NAME} on {_encoder.device}).")
    return _encoder


def release_encoder():
    """Free the encoder (GPU VRAM ~1 GB) when a process is done with it."""
    global _encoder
    if _encoder is not None:
        _encoder.model.to("cpu")
        _encoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def embed_graphs(graphs, texts, device=None, show_progress=True):
    """Attach `.sem` (float32 [1, 768]) to each graph from its raw source text.

    Stored as [1, 768] (like `global_stats`' [1, 37]) so PyG's default
    batch concatenation yields [B, 768] for the classifier, never a flat
    [B*768] vector. `texts` must line up 1:1 with `graphs` (process_split
    keeps them in lockstep by appending only when a graph was built).
    """
    if not graphs:
        return graphs
    if len(texts) != len(graphs):
        raise ValueError(f"semantic alignment error: {len(texts)} texts vs {len(graphs)} graphs")
    vecs = get_encoder(device=device).encode(texts, show_progress=show_progress)
    for g, vec in zip(graphs, vecs):
        g.sem = torch.from_numpy(np.ascontiguousarray(vec, dtype=np.float32)).unsqueeze(0)
    return graphs


def probe_semantic_dim(graphs):
    """768 when the graphs carry `.sem` (new bundles), else 0 (old bundles)."""
    for g in graphs or []:
        sem = getattr(g, "sem", None)
        if sem is not None:
            return int(sem.shape[-1])
    return 0
