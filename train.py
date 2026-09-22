"""Training — mirrors Cell 3 (clean) / Cell 4 (adversarial) of each notebook.

Trains EXACTLY ONE model per run (hybrid style): default trains Model 1
clean; --adversarial trains Model 2 adv. Never builds CPGs — graphs always
come from the {language}_cpg_bundle.pt saved by main.py. Every
output-affecting hyperparameter is a CLI flag defaulting to the notebook
value, e.g. grid search:

  python train.py --language java --hidden_dim 128 --lr 1e-4 --epochs 60
  python main.py --language python [--adversarial]
  python train.py --language python [--adversarial]
"""
import argparse
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torch_geometric.loader import DataLoader

from language_configs import (
    BPE_VOCAB_SIZE,
    DEFAULT_BATCH_SIZE, CLEAN_CHECKPOINT, ADV_CHECKPOINT,
)
from model import AdvancedASTGraphEncoder, save_checkpoint_for_language
from attack_utils import current_rss_mb
from pipeline import load_bundle, require_adv_graphs, seed_everything


TRAIN_TITLE = {
    ("python", False): "Training Model 1: Clean Baseline Python CPG on {}...",
    ("python", True): "Training Model 2: Adversarially Augmented Python CPG on {}...",
    ("cpp", False): "Training Model 1: Clean Baseline C++ CPG on {}...",
    ("cpp", True): "Training Model 2: Adversarially Augmented C++ CPG on {}...",
    ("java", False): "Training Model 1: Clean Baseline CPG on {}...",
    ("java", True): "Training Model 2: Adversarially Augmented CPG on {}...",
}


def train_single_model(train_graphs, val_graphs, ctx, device, batch_size, epochs=45, patience=10,
                       adversarial=False, language="python",
                       lr=5e-4, weight_decay=1e-3, tmax=45, eta_min=1e-6,
                       accum_steps=2, smooth_pos=0.975, smooth_neg=0.025,
                       grad_clip=1.0, threshold=0.50,
                       type_dim=64, subword_dim=128, hidden_dim=256,
                       num_layers=4, dropout_gnn=0.15, pool_hidden=128, film_hidden=128,
                       cls_hidden1=256, cls_hidden2=64,
                       dropout_cls1=0.3, dropout_cls2=0.2, mask_rate=0.15):
    model = AdvancedASTGraphEncoder(
        num_node_types=ctx.vocab_size,
        bpe_vocab_size=BPE_VOCAB_SIZE,
        pad_idx=ctx.pad_id,
        type_dim=type_dim, subword_dim=subword_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, dropout_gnn=dropout_gnn,
        pool_hidden=pool_hidden, film_hidden=film_hidden,
        cls_hidden1=cls_hidden1, cls_hidden2=cls_hidden2,
        dropout_cls1=dropout_cls1, dropout_cls2=dropout_cls2,
        mask_rate=mask_rate,
    ).to(device)
    hyperparams = {
        'type_dim': type_dim, 'subword_dim': subword_dim, 'hidden_dim': hidden_dim,
        'num_layers': num_layers, 'dropout_gnn': dropout_gnn,
        'pool_hidden': pool_hidden, 'film_hidden': film_hidden,
        'cls_hidden1': cls_hidden1, 'cls_hidden2': cls_hidden2,
        'dropout_cls1': dropout_cls1, 'dropout_cls2': dropout_cls2,
        'mask_rate': mask_rate,
    }

    num_pos = sum(1 for g in train_graphs if g.y.item() == 1.0)
    class_weight = torch.tensor([(len(train_graphs) - num_pos) / max(1.0, float(num_pos))]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax, eta_min=eta_min)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    best_f1, best_roc = 0.0, 0.0
    best_weights = None
    epochs_no_improve = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    peak_ram_mb = current_rss_mb()

    print("\n" + TRAIN_TITLE[(language, adversarial)].format(device))
    for epoch in range(epochs):
        ep_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            logits = model(batch, enable_token_masking=adversarial)
            loss = criterion(logits, batch.y * (smooth_pos - smooth_neg) + smooth_neg) / accum_steps
            loss.backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps

        scheduler.step()
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_probs.extend(torch.sigmoid(model(batch)).cpu().numpy().flatten())
                val_labels.extend(batch.y.cpu().numpy().flatten())

        val_probs, val_labels = np.array(val_probs), np.array(val_labels)
        prec_v, rec_v, thres_v = precision_recall_curve(val_labels, val_probs)
        f1_curve = 2 * (prec_v * rec_v) / np.maximum(1e-6, prec_v + rec_v)
        cur_f1 = np.max(f1_curve)
        cur_roc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.0
        peak_ram_mb = max(peak_ram_mb, current_rss_mb())

        ep_duration = time.perf_counter() - ep_start
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val ROC: {cur_roc:.4f} | Val F1: {cur_f1:.4f} | Time: {ep_duration:.1f}s")

        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_roc = cur_roc
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        elif (epochs_no_improve := epochs_no_improve + 1) >= patience:
            print(f"-> Early stopping cleanly triggered at epoch {epoch+1}.")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    ckpt_file = (ADV_CHECKPOINT if adversarial else CLEAN_CHECKPOINT)[language]
    save_checkpoint_for_language(language, ckpt_file, model, optimizer, epoch,
                                 best_f1, best_roc, threshold,
                                 ctx.means, ctx.stds, ctx.g_means, ctx.g_stds,
                                 hyperparams)

    total_time = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    peak_ram_mb = max(peak_ram_mb, current_rss_mb())
    tag = "MODEL 2 ADVERSARIAL" if adversarial else "MODEL 1 CLEAN"
    print(f"\n[DIAGNOSTICS - {tag}]")
    print(f"Total Training Duration : {total_time/60:.2f} minutes")
    print(f"Peak VRAM Consumption   : {peak_vram:.2f} MB")
    print(f"Peak CPU RAM Consumption: {peak_ram_mb:.2f} MB")

    return model, best_f1, best_roc


def train_model(language="python", epochs=45, batch_size=None, adversarial=False, patience=10,
                lr=5e-4, weight_decay=1e-3, tmax=45, eta_min=1e-6,
                accum_steps=2, smooth_pos=0.975, smooth_neg=0.025,
                grad_clip=1.0, threshold=0.50,
                type_dim=64, subword_dim=128, hidden_dim=256,
                num_layers=4, dropout_gnn=0.15, pool_hidden=128, film_hidden=128,
                cls_hidden1=256, cls_hidden2=64,
                dropout_cls1=0.3, dropout_cls2=0.2, mask_rate=0.15):
    """Train one model from the saved bundle. Never builds CPGs.

    Seeds first so the DataLoader shuffle / dropout / token-masking trajectory
    matches the notebook, where Cell 1 seeding carries over into Cells 3/4 in
    the same process. Without this, the split processes diverge and test
    accuracy shifts by ~1pt on identical code.
    """
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE[language]
    bundle = load_bundle(language)
    ctx = bundle["ctx"]
    if adversarial:
        train_graphs = require_adv_graphs(bundle, language)
    else:
        train_graphs = bundle["train_clean_graphs"]
    val_graphs = bundle["val_graphs"]
    return train_single_model(train_graphs, val_graphs, ctx, device, batch_size,
                              epochs=epochs, patience=patience,
                              adversarial=adversarial, language=language,
                              lr=lr, weight_decay=weight_decay, tmax=tmax, eta_min=eta_min,
                              accum_steps=accum_steps, smooth_pos=smooth_pos, smooth_neg=smooth_neg,
                              grad_clip=grad_clip, threshold=threshold,
                              type_dim=type_dim, subword_dim=subword_dim, hidden_dim=hidden_dim,
                              num_layers=num_layers, dropout_gnn=dropout_gnn,
                              pool_hidden=pool_hidden, film_hidden=film_hidden,
                              cls_hidden1=cls_hidden1, cls_hidden2=cls_hidden2,
                              dropout_cls1=dropout_cls1, dropout_cls2=dropout_cls2,
                              mask_rate=mask_rate)


def _add_hyper_args(parser):
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--tmax", type=int, default=45)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--smooth_pos", type=float, default=0.975)
    parser.add_argument("--smooth_neg", type=float, default=0.025)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--type_dim", type=int, default=64)
    parser.add_argument("--subword_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout_gnn", type=float, default=0.15)
    parser.add_argument("--pool_hidden", type=int, default=128)
    parser.add_argument("--film_hidden", type=int, default=128)
    parser.add_argument("--cls_hidden1", type=int, default=256)
    parser.add_argument("--cls_hidden2", type=int, default=64)
    parser.add_argument("--dropout_cls1", type=float, default=0.3)
    parser.add_argument("--dropout_cls2", type=float, default=0.2)
    parser.add_argument("--mask_rate", type=float, default=0.15)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--adversarial", action="store_true",
                        help="Train the adversarially augmented variant (token masking on)")
    parser.add_argument("--patience", type=int, default=10)
    parser = _add_hyper_args(parser)
    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE[args.language]
    train_model(language=args.language, epochs=args.epochs, batch_size=args.batch_size,
                adversarial=args.adversarial, patience=args.patience,
                lr=args.lr, weight_decay=args.weight_decay, tmax=args.tmax, eta_min=args.eta_min,
                accum_steps=args.accum_steps, smooth_pos=args.smooth_pos, smooth_neg=args.smooth_neg,
                grad_clip=args.grad_clip, threshold=args.threshold,
                type_dim=args.type_dim, subword_dim=args.subword_dim, hidden_dim=args.hidden_dim,
                num_layers=args.num_layers, dropout_gnn=args.dropout_gnn,
                pool_hidden=args.pool_hidden, film_hidden=args.film_hidden,
                cls_hidden1=args.cls_hidden1, cls_hidden2=args.cls_hidden2,
                dropout_cls1=args.dropout_cls1, dropout_cls2=args.dropout_cls2,
                mask_rate=args.mask_rate)
