"""GNN model — identical across all three notebooks, combined once.

GatedGNNLayer and AdvancedASTGraphEncoder bodies are byte-identical
(modulo a single comment line in the java notebook, preserved below).
Save helpers preserve each notebook's exact filename convention and
print wording; outputs (checkpoint dict keys) are identical.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_add_pool, global_max_pool
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


# Notebook-exact grid-searchable defaults. The constructor literals below mirror
# these values; keep both in sync. struct_dim (34-d features) and
# num_relations (16 edge types) stay locked — changing them breaks graph compat.
HPARAM_DEFAULTS = {
    'type_dim': 64, 'subword_dim': 128, 'hidden_dim': 256, 'global_dim': 16,
    'num_layers': 4, 'dropout_gnn': 0.15,
    'pool_hidden': 128, 'film_hidden': 128,
    'cls_hidden1': 256, 'cls_hidden2': 64,
    'dropout_cls1': 0.3, 'dropout_cls2': 0.2,
    'mask_rate': 0.15,
}


class AdvancedASTGraphEncoder(nn.Module):
    def __init__(self, num_node_types, bpe_vocab_size, type_dim=64, struct_dim=34, subword_dim=128, hidden_dim=256, num_relations=16, pad_idx=0, global_dim=16,
                 num_layers=4, dropout_gnn=0.15, pool_hidden=128, film_hidden=128,
                 cls_hidden1=256, cls_hidden2=64, dropout_cls1=0.3, dropout_cls2=0.2,
                 mask_rate=0.15):
        super().__init__()
        self.pad_idx = pad_idx
        self.mask_rate = mask_rate
        self.type_emb = nn.Embedding(num_node_types, type_dim)
        self.subword_emb = nn.Embedding(bpe_vocab_size, subword_dim, padding_idx=pad_idx)
        self.subword_conv = nn.Conv1d(subword_dim, subword_dim, 3, padding=1)
        self.subword_attn = nn.Linear(subword_dim, 1)
        self.proj = nn.Sequential(nn.Linear(type_dim + struct_dim + subword_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.layers = nn.ModuleList([GatedGNNLayer(hidden_dim, num_relations, dropout_gnn) for _ in range(num_layers)])
        # NOTE: graph_emb dim is hidden_dim * num_layers * 3
        # (JK-concat add-pool + max-pool + virtual). At notebook defaults
        # (256 x 4) this is exactly 3072, matching the notebooks bit-for-bit.
        self.pool_att = nn.Sequential(nn.Linear(hidden_dim * num_layers + struct_dim, pool_hidden), nn.GELU(), nn.Linear(pool_hidden, 1))
        self.film_gate = nn.Sequential(nn.Linear(global_dim, film_hidden), nn.LayerNorm(film_hidden), nn.GELU(), nn.Linear(film_hidden, hidden_dim * num_layers * 3))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * num_layers * 3, cls_hidden1), nn.LayerNorm(cls_hidden1), nn.GELU(), nn.Dropout(dropout_cls1), nn.Linear(cls_hidden1, cls_hidden2), nn.GELU(), nn.Dropout(dropout_cls2), nn.Linear(cls_hidden2, 1))

    def forward(self, data, enable_token_masking=False):
        edge_index, edge_type = data.edge_index, data.edge_attr
        x_subwords = data.x_subwords.clone()

        if self.training:
            pass  # Removed asymmetric edge dropout

            # Step 2: In-loop token masking activated only for Adversarial variant
            if enable_token_masking:
                subword_mask = torch.rand(x_subwords.size(0), device=x_subwords.device) < self.mask_rate
                x_subwords[subword_mask] = self.pad_idx

        sub_conv = F.gelu(self.subword_conv(self.subword_emb(x_subwords).permute(0, 2, 1))).permute(0, 2, 1)
        attn_logits = self.subword_attn(sub_conv).masked_fill(~(x_subwords != self.pad_idx).unsqueeze(-1), -1e9)
        agg_sub_emb = (sub_conv * torch.nan_to_num(torch.softmax(attn_logits, dim=1), nan=0.0)).sum(dim=1)

        x = self.proj(torch.cat([self.type_emb(data.x_type), data.x_struct, agg_sub_emb], dim=-1))
        h_all = []
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
            h_all.append(x)
        h_jk = torch.cat(h_all, dim=-1)

        virtual_indices = torch.cumsum(torch.bincount(data.batch), dim=0) - 1
        ast_mask = torch.ones(h_jk.size(0), dtype=torch.bool, device=h_jk.device)
        ast_mask[virtual_indices] = False

        h_jk_ast, batch_ast = h_jk[ast_mask], data.batch[ast_mask]
        att_weights = softmax(self.pool_att(torch.cat([h_jk_ast, data.x_struct[ast_mask]], dim=-1)), batch_ast)

        graph_emb = torch.cat([global_add_pool(h_jk_ast * att_weights, batch_ast), global_max_pool(h_jk_ast, batch_ast), h_jk[virtual_indices]], dim=-1)
        return self.classifier(graph_emb * torch.sigmoid(self.film_gate(data.global_stats))).view(-1)


def _checkpoint_dict(model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds, hyperparams=None):
    return {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': val_f1,
        'val_roc': val_roc,
        'optimal_threshold': threshold,
        'normalization_means': means,
        'normalization_stds': stds,
        'global_means': g_means,
        'global_stds': g_stds,
        'hyperparams': dict(hyperparams) if hyperparams else dict(HPARAM_DEFAULTS),
    }


def save_python_checkpoint(filepath, model, optimizer, epoch, val_f1, val_roc, threshold, means=None, stds=None, g_means=None, g_stds=None, hyperparams=None):
    torch.save(_checkpoint_dict(model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds, hyperparams), filepath)
    fsize_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"[+] Checkpoint preserved at: {filepath} ({fsize_mb:.2f} MB)")


def save_cpp_checkpoint(filepath, model, optimizer, epoch, val_f1, val_roc, threshold, means=None, stds=None, g_means=None, g_stds=None, hyperparams=None):
    torch.save(_checkpoint_dict(model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds, hyperparams), filepath)
    fsize_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"[+] Checkpoint saved at: {filepath} ({fsize_mb:.2f} MB)")


def save_checkpoint(filepath, model, optimizer, epoch, val_f1, val_roc, threshold, means=None, stds=None, g_means=None, g_stds=None, hyperparams=None):
    torch.save(_checkpoint_dict(model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds, hyperparams), filepath)
    print(f"[+] Checkpoint safely preserved at: {filepath}")


def save_checkpoint_for_language(language, filepath, model, optimizer, epoch, val_f1, val_roc, threshold, means=None, stds=None, g_means=None, g_stds=None, hyperparams=None):
    if language == "python":
        return save_python_checkpoint(filepath, model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds, hyperparams)
    elif language == "cpp":
        return save_cpp_checkpoint(filepath, model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds, hyperparams)
    else:
        return save_checkpoint(filepath, model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds, hyperparams)


def load_checkpoint(filepath, model, device, optimizer=None):
    try:
        ckpt = torch.load(filepath, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.6 without the weights_only kwarg
        ckpt = torch.load(filepath, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
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


def build_encoder(ctx, hparams=None, device="cpu", bpe_vocab_size=None):
    """Rebuild the encoder for eval: checkpoint hyperparams win, else notebook defaults."""
    from language_configs import BPE_VOCAB_SIZE
    hp = dict(HPARAM_DEFAULTS)
    hp.update(hparams or {})
    model = AdvancedASTGraphEncoder(
        num_node_types=ctx.vocab_size,
        bpe_vocab_size=bpe_vocab_size or BPE_VOCAB_SIZE,
        pad_idx=ctx.pad_id,
        type_dim=hp['type_dim'], subword_dim=hp['subword_dim'],
        hidden_dim=hp['hidden_dim'], global_dim=hp['global_dim'],
        num_layers=hp['num_layers'], dropout_gnn=hp['dropout_gnn'],
        pool_hidden=hp['pool_hidden'], film_hidden=hp['film_hidden'],
        cls_hidden1=hp['cls_hidden1'], cls_hidden2=hp['cls_hidden2'],
        dropout_cls1=hp['dropout_cls1'], dropout_cls2=hp['dropout_cls2'],
        mask_rate=hp['mask_rate'],
    ).to(device)
    return model, hp


def build_encoder_from_checkpoint(ctx, filepath, device="cpu"):
    hp = checkpoint_hparams(filepath, device)
    model, hp = build_encoder(ctx, hp, device)
    load_checkpoint(filepath, model, device)
    return model, hp
