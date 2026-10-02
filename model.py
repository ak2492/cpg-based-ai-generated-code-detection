import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax


class GatedGNNLayer(nn.Module):
    def __init__(self, hidden_dim, num_relations, dropout=0.15):
        super().__init__()
        self.conv = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=None)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        h = self.dropout(F.gelu(self.norm(self.conv(x, edge_index, edge_type))))
        g = torch.sigmoid(self.gate(torch.cat([x, h], dim=-1)))
        return g * h + (1.0 - g) * x


HPARAM_DEFAULTS = {
    'type_dim': 64, 'subword_dim': 128, 'hidden_dim': 256,
    'num_layers': 4, 'dropout_gnn': 0.15,
    'pool_hidden': 128, 'film_hidden': 128,
    'cls_hidden1': 256, 'cls_hidden2': 64,
    'dropout_cls1': 0.3, 'dropout_cls2': 0.2,
    'mask_rate': 0.15,
}


# Ablation configs for the paper study. Each id isolates exactly one
# architectural decision against `full` (plus the `no-mp` floor). I run all
# ten on each language, seed 42, clean training only, test-set eval only.
ABLATION_IDS = (
    "full",        # 0 reference upper bound
    "no-mp",       # 1 floor: 0 RGCN layers, projected features only
    "no-subword",  # 2 drop identifier/token-content branch
    "no-struct",   # 3 drop hand-engineered per-node stylometry
    "syntax-only", # 4 keep edge types {0,1,2,3,8}; drop data-flow/CFG/call/next-token
    "no-gate",     # 5 plain residual x+h instead of learned gate
    "no-jk",       # 6 last-layer readout instead of multi-scale fusion
    "no-virtual",  # 7 drop global supernode; pooled AST nodes only
    "mean-pool",   # 8 mean pooling instead of learned attention pooling
    "no-film",     # 9 no file-level macro-stat modulation
)

# Edge types kept by the syntax-only graph variant (parent<->child,
# siblings, node->virtual). num_relations stays 16 so RGCN capacity is
# untouched; absent relations simply never appear.
SYNTAX_ONLY_EDGE_TYPES = frozenset((0, 1, 2, 3, 8))

# Graph variants each ablation needs. Only syntax-only and no-virtual change
# graph construction; the other seven reuse the standard parsed graphs, so
# test-set extraction happens once per language and is never repeated.
GRAPH_VARIANT_FOR_ABLATION = {
    "syntax-only": "syntax-only",
    "no-virtual": "no-virtual",
}
GRAPH_VARIANTS = ("standard", "syntax-only", "no-virtual")


def parse_ablation(ablation):
    """Split a (possibly combined) ablation id into its single-factor parts.

    Combos join singles with '+', e.g. 'no-subword+no-struct'. Returns the
    parts in canonical ABLATION_IDS order so checkpoint filenames are stable
    no matter what order the user typed. Raises on unknown, duplicate, or
    degenerate combos.
    """
    parts = [p.strip() for p in str(ablation).split("+")]
    if any(not p for p in parts):
        raise ValueError(f"Malformed ablation id '{ablation}'. Use '+'-joined ids from {list(ABLATION_IDS)}.")
    for p in parts:
        if p not in ABLATION_IDS:
            raise ValueError(f"Unknown ablation '{p}' in '{ablation}'. Choose from {list(ABLATION_IDS)}.")
    if len(set(parts)) != len(parts):
        raise ValueError(f"Duplicate component in ablation '{ablation}'.")
    if "full" in parts and len(parts) > 1:
        raise ValueError(f"'full' cannot be combined: '{ablation}'.")
    if "no-mp" in parts and len(parts) > 1:
        # I reject these because the floor already zeroes the layer stack:
        # no-mp+no-jk (or +no-gate/...) is just no-mp wearing a disguise,
        # and running it would spend a full training on a duplicate.
        raise ValueError(f"'no-mp' cannot be combined: '{ablation}' is identical to 'no-mp'.")
    return tuple(p for p in ABLATION_IDS if p in parts)


def canonical_ablation(ablation):
    """Canonical (order-stable) string for an ablation id or combo."""
    return "+".join(parse_ablation(ablation))


def graph_variant_for_ablation(ablation):
    """Return the graph variant an ablation id trains/evaluates on."""
    variants = graph_variants_for_ablation(ablation)
    if len(variants) > 1:
        raise ValueError(
            f"Ablation '{ablation}' needs graph variants {list(variants)}; "
            f"use graph_variants_for_ablation() instead.")
    return variants[0] if variants else "standard"


def graph_variants_for_ablation(ablation):
    """Return the graph variants a (possibly combined) ablation needs, in
    derivation order (edge filter before supernode strip). Empty tuple means
    standard graphs."""
    parts = parse_ablation(ablation)
    ordered = [v for v in GRAPH_VARIANTS[1:] for p in parts
               if GRAPH_VARIANT_FOR_ABLATION.get(p) == v]
    return tuple(dict.fromkeys(ordered))


class AdvancedASTGraphEncoder(nn.Module):
    def __init__(self, num_node_types, bpe_vocab_size, type_dim=64, struct_dim=28, subword_dim=128, hidden_dim=256, num_relations=16, pad_idx=0, global_dim=37,
                 num_layers=4, dropout_gnn=0.15, pool_hidden=128, film_hidden=128,
                 cls_hidden1=256, cls_hidden2=64, dropout_cls1=0.3, dropout_cls2=0.2,
                 mask_rate=0.15, ablation="full"):
        super().__init__()
        parts = set(parse_ablation(ablation))
        self.ablation = canonical_ablation(ablation)
        self.ablation_parts = frozenset(parts)
        self.pad_idx = pad_idx
        self.mask_rate = mask_rate
        self.use_subword = "no-subword" not in parts
        self.use_struct = "no-struct" not in parts
        self.use_gate = "no-gate" not in parts
        self.use_virtual = "no-virtual" not in parts
        self.use_attn_pool = "mean-pool" not in parts
        self.use_film = "no-film" not in parts
        # I keep the full 4-layer stack for every config except the floor:
        # no-mp zeroes it (JK/virtual become vacuous by construction there).
        eff_layers = 0 if "no-mp" in parts else num_layers
        # I fuse scales exactly like the full model except no-jk (last layer
        # only) and no-mp (single-scale projected features, no message passing).
        self.jk_last_only = "no-jk" in parts
        jk_scales = 1 if ("no-mp" in parts or "no-jk" in parts) else eff_layers
        self.jk_dim = hidden_dim * jk_scales
        # I derive every downstream width from the flags above so full-model
        # dims (proj 220->256, pool_att 1052->128, film 37->3072, clf 3072)
        # fall out unchanged when ablation="full".
        proj_in = type_dim + (struct_dim if self.use_struct else 0) + (subword_dim if self.use_subword else 0)
        pool_in = self.jk_dim + (struct_dim if self.use_struct else 0)
        graph_dim = self.jk_dim * (3 if self.use_virtual else 2)
        self.type_emb = nn.Embedding(num_node_types, type_dim)
        if self.use_subword:
            self.subword_emb = nn.Embedding(bpe_vocab_size, subword_dim, padding_idx=pad_idx)
            self.subword_conv = nn.Conv1d(subword_dim, subword_dim, 3, padding=1)
            self.subword_attn = nn.Linear(subword_dim, 1)
        self.proj = nn.Sequential(nn.Linear(proj_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.layers = nn.ModuleList([GatedGNNLayer(hidden_dim, num_relations, dropout_gnn) for _ in range(eff_layers)])
        # NOTE: full-model graph_emb dim is hidden_dim * num_layers * 3
        # (JK-concat add-pool + max-pool + virtual). At notebook defaults
        # (256 x 4) this is exactly 3072, matching the notebooks bit-for-bit.
        # Ablations derive it as jk_dim * (3 with virtual | 2 without).
        if self.use_attn_pool:
            self.pool_att = nn.Sequential(nn.Linear(pool_in, pool_hidden), nn.GELU(), nn.Linear(pool_hidden, 1))
        else:
            # I delete the attention module outright so param counts stay
            # honest for the mean-pooling comparison.
            self.pool_att = None
        if self.use_film:
            self.film_gate = nn.Sequential(nn.Linear(global_dim, film_hidden), nn.LayerNorm(film_hidden), nn.GELU(), nn.Linear(film_hidden, graph_dim))
        else:
            self.film_gate = None
        self.classifier = nn.Sequential(nn.Linear(graph_dim, cls_hidden1), nn.LayerNorm(cls_hidden1), nn.GELU(), nn.Dropout(dropout_cls1), nn.Linear(cls_hidden1, cls_hidden2), nn.GELU(), nn.Dropout(dropout_cls2), nn.Linear(cls_hidden2, 1))

    def forward(self, data, enable_token_masking=False):
        edge_index, edge_type = data.edge_index, data.edge_attr
        x_subwords = data.x_subwords.clone()

        if self.training:
            pass  # Removed asymmetric edge dropout

            # Step 2: In-loop token masking activated only for Adversarial variant
            if enable_token_masking:
                subword_mask = torch.rand(x_subwords.size(0), device=x_subwords.device) < self.mask_rate
                x_subwords[subword_mask] = self.pad_idx

        proj_parts = [self.type_emb(data.x_type)]
        if self.use_struct:
            proj_parts.append(data.x_struct)
        if self.use_subword:
            sub_conv = F.gelu(self.subword_conv(self.subword_emb(x_subwords).permute(0, 2, 1))).permute(0, 2, 1)
            attn_logits = self.subword_attn(sub_conv).masked_fill(~(x_subwords != self.pad_idx).unsqueeze(-1), -1e9)
            agg_sub_emb = (sub_conv * torch.nan_to_num(torch.softmax(attn_logits, dim=1), nan=0.0)).sum(dim=1)
            proj_parts.append(agg_sub_emb)

        x = self.proj(torch.cat(proj_parts, dim=-1))
        if "no-mp" in self.ablation_parts:
            h_jk = x
        else:
            h_all = []
            for layer in self.layers:
                if self.use_gate:
                    x = layer(x, edge_index, edge_type)
                else:
                    # I ablate only the gate here: same conv+norm+dropout as
                    # GatedGNNLayer, plain residual instead of g*h+(1-g)*x.
                    # The unused gate Linear stays in the module on purpose:
                    # identical capacity means the comparison isolates the
                    # computation (gated blend vs addition), not model size.
                    h = layer.dropout(F.gelu(layer.norm(layer.conv(x, edge_index, edge_type))))
                    x = x + h
                h_all.append(x)
            h_jk = h_all[-1] if self.jk_last_only else torch.cat(h_all, dim=-1)

        if self.use_virtual:
            virtual_indices = torch.cumsum(torch.bincount(data.batch), dim=0) - 1
            ast_mask = torch.ones(h_jk.size(0), dtype=torch.bool, device=h_jk.device)
            ast_mask[virtual_indices] = False
            h_jk_ast, batch_ast = h_jk[ast_mask], data.batch[ast_mask]
        else:
            # I built no-virtual graphs without the supernode, so every node
            # is an AST node and pooling covers the whole batch directly.
            h_jk_ast, batch_ast = h_jk, data.batch

        if self.use_attn_pool:
            pool_parts = [h_jk_ast]
            if self.use_struct:
                pool_parts.append(data.x_struct[ast_mask] if self.use_virtual else data.x_struct)
            att_weights = softmax(self.pool_att(torch.cat(pool_parts, dim=-1)), batch_ast)
            pooled = global_add_pool(h_jk_ast * att_weights, batch_ast)
        else:
            pooled = global_mean_pool(h_jk_ast, batch_ast)

        if self.use_virtual:
            graph_emb = torch.cat([pooled, global_max_pool(h_jk_ast, batch_ast), h_jk[virtual_indices]], dim=-1)
        else:
            graph_emb = torch.cat([pooled, global_max_pool(h_jk_ast, batch_ast)], dim=-1)
        if self.use_film:
            graph_emb = graph_emb * torch.sigmoid(self.film_gate(data.global_stats))
        return self.classifier(graph_emb).view(-1)


def _checkpoint_dict(model, optimizer=None, epoch=0, val_f1=0.0, val_roc=0.0, threshold=0.50, means=None, stds=None, g_means=None, g_stds=None, hyperparams=None, ablation="full"):
    # Hybrid-style: weights + metadata only. Optimizer state is intentionally
    # omitted (~2/3 of file size) — no eval/attack path ever consumed it, and
    # exact-resume would additionally need scheduler + RNG states anyway.
    # `optimizer` is kept as an accepted (ignored) arg so existing
    # train.py call sites pass unchanged; old files containing the key
    # still load via load_checkpoint's tolerant guard below.
    return {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_f1': val_f1,
        'val_roc': val_roc,
        'optimal_threshold': threshold,
        'normalization_means': means,
        'normalization_stds': stds,
        'global_means': g_means,
        'global_stds': g_stds,
        'hyperparams': dict(hyperparams) if hyperparams else dict(HPARAM_DEFAULTS),
        # I persist the ablation id so eval rebuilds the matching encoder
        # variant automatically; old checkpoints predate the study and read
        # back as "full".
        'ablation': ablation,
    }


def _atomic_torch_save(payload, filepath):
    """Write payload atomically: save to sibling .tmp then os.replace.

    A kill mid-write leaves the previous canonical file intact instead of
    a truncated .pth. Tmp lives in the same directory so the rename is
    atomic on both POSIX and Windows.
    """
    tmp_path = f"{filepath}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, filepath)


def save_python_checkpoint(filepath, model, optimizer=None, epoch=0, val_f1=0.0, val_roc=0.0, threshold=0.50, means=None, stds=None, g_means=None, g_stds=None, hyperparams=None, ablation="full"):
    _atomic_torch_save(_checkpoint_dict(model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds, hyperparams, ablation), filepath)
    fsize_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"[+] Checkpoint preserved at: {filepath} ({fsize_mb:.2f} MB)")


def save_cpp_checkpoint(filepath, model, optimizer=None, epoch=0, val_f1=0.0, val_roc=0.0, threshold=0.50, means=None, stds=None, g_means=None, g_stds=None, hyperparams=None, ablation="full"):
    _atomic_torch_save(_checkpoint_dict(model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds, hyperparams, ablation), filepath)
    fsize_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"[+] Checkpoint saved at: {filepath} ({fsize_mb:.2f} MB)")


def save_checkpoint(filepath, model, optimizer=None, epoch=0, val_f1=0.0, val_roc=0.0, threshold=0.50, means=None, stds=None, g_means=None, g_stds=None, hyperparams=None, ablation="full"):
    _atomic_torch_save(_checkpoint_dict(model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds, hyperparams, ablation), filepath)
    print(f"[+] Checkpoint safely preserved at: {filepath}")


def save_checkpoint_for_language(language, filepath, model, optimizer=None, epoch=0, val_f1=0.0, val_roc=0.0, threshold=0.50, means=None, stds=None, g_means=None, g_stds=None, hyperparams=None, ablation="full"):
    if language == "python":
        return save_python_checkpoint(filepath, model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds, hyperparams, ablation)
    elif language == "cpp":
        return save_cpp_checkpoint(filepath, model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds, hyperparams, ablation)
    else:
        return save_checkpoint(filepath, model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds, hyperparams, ablation)


def ablation_checkpoint_path(language, ablation, adversarial=False):
    """Namespaced checkpoint file for an ablation run.

    Every non-full config (single or combined, canonically ordered) gets its
    own file so study runs can never clobber the main checkpoints.
    """
    canonical = canonical_ablation(ablation)
    tag = "adv" if adversarial else "clean"
    return f"model_{tag}_abl-{canonical}_{language}.pth"


def load_checkpoint(filepath, model, device, optimizer=None):
    try:
        ckpt = torch.load(filepath, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.6 without the weights_only kwarg
        ckpt = torch.load(filepath, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    # Backward compat: pre-drop files contain 'optimizer_state_dict';
    # new hybrid-style files omit it. Either loads for eval.
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt


def checkpoint_hparams(filepath, device="cpu"):
    """Read only the hyperparams dict from a checkpoint (defaults for old files)."""
    try:
        ckpt = torch.load(filepath, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(filepath, map_location=device)
    hp = dict(HPARAM_DEFAULTS)
    hp.update(ckpt.get('hyperparams') or {})
    return hp


def checkpoint_ablation(filepath, device="cpu"):
    """Read the ablation id stored in a checkpoint (old files predate the study: full)."""
    try:
        ckpt = torch.load(filepath, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(filepath, map_location=device)
    ablation = ckpt.get('ablation', 'full')
    return canonical_ablation(ablation)


def build_encoder(ctx, hparams=None, device="cpu", bpe_vocab_size=None, ablation="full"):
    """Rebuild the encoder for eval: checkpoint hyperparams win, else notebook defaults."""
    from language_configs import BPE_VOCAB_SIZE
    hp = dict(HPARAM_DEFAULTS)
    hp.update(hparams or {})
    model = AdvancedASTGraphEncoder(
        num_node_types=ctx.vocab_size,
        bpe_vocab_size=bpe_vocab_size or BPE_VOCAB_SIZE,
        pad_idx=ctx.pad_id,
        type_dim=hp['type_dim'], subword_dim=hp['subword_dim'],
        hidden_dim=hp['hidden_dim'], global_dim=37,
        num_layers=hp['num_layers'], dropout_gnn=hp['dropout_gnn'],
        pool_hidden=hp['pool_hidden'], film_hidden=hp['film_hidden'],
        cls_hidden1=hp['cls_hidden1'], cls_hidden2=hp['cls_hidden2'],
        dropout_cls1=hp['dropout_cls1'], dropout_cls2=hp['dropout_cls2'],
        mask_rate=hp['mask_rate'], ablation=ablation,
    ).to(device)
    return model, hp


def build_encoder_from_checkpoint(ctx, filepath, device="cpu"):
    ablation = checkpoint_ablation(filepath, device)
    hp = checkpoint_hparams(filepath, device)
    model, hp = build_encoder(ctx, hp, device, ablation=ablation)
    load_checkpoint(filepath, model, device)
    return model, hp
