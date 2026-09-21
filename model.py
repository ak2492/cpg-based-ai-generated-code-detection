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
    def __init__(self, hidden_dim, num_relations):
        super().__init__()
        self.conv = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=None)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x, edge_index, edge_type):
        h = self.dropout(F.gelu(self.norm(self.conv(x, edge_index, edge_type))))
        g = torch.sigmoid(self.gate(torch.cat([x, h], dim=-1)))
        return g * h + (1.0 - g) * x


class AdvancedASTGraphEncoder(nn.Module):
    def __init__(self, num_node_types, bpe_vocab_size, type_dim=64, struct_dim=34, subword_dim=128, hidden_dim=256, num_relations=16, pad_idx=0, global_dim=16):
        super().__init__()
        self.pad_idx = pad_idx
        self.type_emb = nn.Embedding(num_node_types, type_dim)
        self.subword_emb = nn.Embedding(bpe_vocab_size, subword_dim, padding_idx=pad_idx)
        self.subword_conv = nn.Conv1d(subword_dim, subword_dim, 3, padding=1)
        self.subword_attn = nn.Linear(subword_dim, 1)
        self.proj = nn.Sequential(nn.Linear(type_dim + struct_dim + subword_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.layers = nn.ModuleList([GatedGNNLayer(hidden_dim, num_relations) for _ in range(4)])
        self.pool_att = nn.Sequential(nn.Linear(hidden_dim * 4 + struct_dim, 128), nn.GELU(), nn.Linear(128, 1))
        self.film_gate = nn.Sequential(nn.Linear(global_dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, hidden_dim * 12))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 12, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, data, enable_token_masking=False):
        edge_index, edge_type = data.edge_index, data.edge_attr
        x_subwords = data.x_subwords.clone()

        if self.training:
            pass  # Removed asymmetric edge dropout

            # Step 2: In-loop token masking activated only for Adversarial variant
            if enable_token_masking:
                subword_mask = torch.rand(x_subwords.size(0), device=x_subwords.device) < 0.15
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


def _checkpoint_dict(model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds):
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
    }


def save_python_checkpoint(filepath, model, optimizer, epoch, val_f1, val_roc, threshold, means=None, stds=None, g_means=None, g_stds=None):
    torch.save(_checkpoint_dict(model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds), filepath)
    fsize_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"[+] Checkpoint preserved at: {filepath} ({fsize_mb:.2f} MB)")


def save_cpp_checkpoint(filepath, model, optimizer, epoch, val_f1, val_roc, threshold, means=None, stds=None, g_means=None, g_stds=None):
    torch.save(_checkpoint_dict(model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds), filepath)
    fsize_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"[+] Checkpoint saved at: {filepath} ({fsize_mb:.2f} MB)")


def save_checkpoint(filepath, model, optimizer, epoch, val_f1, val_roc, threshold, means=None, stds=None, g_means=None, g_stds=None):
    torch.save(_checkpoint_dict(model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds), filepath)
    print(f"[+] Checkpoint safely preserved at: {filepath}")


def save_checkpoint_for_language(language, filepath, model, optimizer, epoch, val_f1, val_roc, threshold, means=None, stds=None, g_means=None, g_stds=None):
    if language == "python":
        return save_python_checkpoint(filepath, model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds)
    elif language == "cpp":
        return save_cpp_checkpoint(filepath, model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds)
    else:
        return save_checkpoint(filepath, model, optimizer, epoch, val_f1, val_roc, threshold, means, stds, g_means, g_stds)


def load_checkpoint(filepath, model, device, optimizer=None):
    try:
        ckpt = torch.load(filepath, map_location=device, weights_only=False)
    except TypeError:  # torch < 2.6 without the weights_only kwarg
        ckpt = torch.load(filepath, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt
