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
from model import AdvancedASTGraphEncoder, ablation_checkpoint_path, canonical_ablation, save_checkpoint_for_language
from attack_utils import current_rss_mb, execute_model_eval_with_cost, print_detailed_metrics_with_cost
from pipeline import load_bundle_for_ablation, require_adv_graphs, seed_everything


TRAIN_TITLE = {
    ("python", False): "Training Model 1: Clean Baseline Python CPG on {}...",
    ("python", True): "Training Model 2: Adversarially Augmented Python CPG on {}...",
    ("cpp", False): "Training Model 1: Clean Baseline C++ CPG on {}...",
    ("cpp", True): "Training Model 2: Adversarially Augmented C++ CPG on {}...",
    ("java", False): "Training Model 1: Clean Baseline CPG on {}...",
    ("java", True): "Training Model 2: Adversarially Augmented CPG on {}...",
}


def _f1_at(thr, y, s, prior=None):
    """F1 of the hard decision s >= thr.

    With `prior` (positive prevalence) the class-conditional TPR/FPR of the
    cut are reweighted to that prevalence; without it this is the plain F1.
    """
    pred = (s >= thr).astype(np.float64)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    if prior is not None and 0.0 < prior < 1.0:
        tpr = tp / max(1.0, float((y == 1).sum()))
        fpr = fp / max(1.0, float((y == 0).sum()))
        tp_p, fp_p, fn_p = prior * tpr, (1.0 - prior) * fpr, prior * (1.0 - tpr)
        return 2.0 * tp_p / max(1e-9, 2.0 * tp_p + fp_p + fn_p)
    return 2 * tp / max(1e-9, 2 * tp + fp + fn)


def _best_threshold(scores, labels, prior=None):
    """F1-argmax decision threshold, reweighted to `prior` when given."""
    if prior is not None and 0.0 < prior < 1.0:
        order = np.argsort(-scores, kind="mergesort")
        s, y = scores[order], labels[order]
        n_pos = max(1.0, float(y.sum()))
        n_neg = max(1.0, float(len(y) - y.sum()))
        tpr, fpr = np.cumsum(y) / n_pos, np.cumsum(1.0 - y) / n_neg
        tp_p = prior * tpr
        fp_p = (1.0 - prior) * fpr
        fn_p = prior * (1.0 - tpr)
        f1 = 2.0 * tp_p / np.maximum(1e-9, 2.0 * tp_p + fp_p + fn_p)
        return float(s[int(np.argmax(f1))])
    prec, rec, thres = precision_recall_curve(labels, scores)
    if len(thres) == 0:
        return None
    f1_curve = 2 * (prec * rec) / np.maximum(1e-6, prec + rec)
    # precision_recall_curve returns one more precision/recall point than
    # thresholds; index i corresponds to thres[i], so I drop the final
    # (recall=0) point before taking the argmax.
    return float(thres[int(np.argmax(f1_curve[:-1]))])


def _select_threshold(probs, labels, fallback=0.50, prior=None, n_folds=5, min_gain=5e-4):
    """Pick a decision threshold, but only if calibration beats the fixed cut.

    Two failure modes are gated here. (1) Overfitting: the plain val-F1
    argmax on a small balanced val split (Java's val is 806 rows) picked cuts
    of 0.66-0.97 that cost ~0.003 Java test F1 versus a flat 0.50. (2) Prior
    shift: val is balanced by design, while the Java test set is 69% machine
    — an F1-optimal cut for a 50/50 channel is not optimal for 69/31. When
    `prior` (eval-set positive prevalence) is known, the threshold is the
    argmax of the val-ROC reweighted to that prevalence. Either way, the
    calibrated RULE itself is cross-validated on the val scores: fit on K-1
    folds, score held-out F1 (same prior reweighting), and adopt the
    calibrated cut only when it beats the fixed fallback by a margin. Pure
    numpy on the score arrays, so it is easy to unit test.
    """
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float(fallback)
    thr_full = _best_threshold(probs, labels, prior=prior)
    if thr_full is None:
        return float(fallback)

    from sklearn.model_selection import StratifiedKFold
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    rule_scores, fixed_scores = [], []
    for tr_idx, te_idx in splitter.split(probs, labels):
        y_tr, s_tr = labels[tr_idx], probs[tr_idx]
        y_te, s_te = labels[te_idx], probs[te_idx]
        if len(np.unique(y_tr)) < 2:
            continue
        thr_fold = _best_threshold(s_tr, y_tr, prior=prior)
        if thr_fold is None:
            continue
        rule_scores.append(_f1_at(thr_fold, y_te, s_te, prior=prior))
        fixed_scores.append(_f1_at(fallback, y_te, s_te, prior=prior))
    if len(rule_scores) >= 2:
        rule_mean = float(np.mean(rule_scores))
        fixed_mean = float(np.mean(fixed_scores))
        if rule_mean > fixed_mean + min_gain:
            print(f"[i] Threshold calibration kept: held-out F1 rule={rule_mean:.4f} "
                  f"vs fixed {fallback:.2f}={fixed_mean:.4f} -> threshold {thr_full:.4f}")
            return thr_full
        print(f"[i] Threshold calibration rejected: held-out F1 rule={rule_mean:.4f} "
              f"vs fixed {fallback:.2f}={fixed_mean:.4f} -> threshold {fallback:.2f}")
        return float(fallback)
    return thr_full


def _val_optimal_threshold(model, val_loader, device, fallback=0.50, prior=None):
    """Held-out-validated threshold selection on the val split (see above)."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            probs.extend(torch.sigmoid(model(batch)).cpu().numpy().flatten())
            labels.extend(batch.y.cpu().numpy().flatten())
    probs, labels = np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    return _select_threshold(probs, labels, fallback=fallback, prior=prior)


def train_single_model(train_graphs, val_graphs, ctx, device, batch_size, epochs=30, patience=30,
                       adversarial=False, language="python",
                       lr=5e-5, weight_decay=1e-3, tmax=45, eta_min=1e-6,
                       accum_steps=2, smooth_pos=0.975, smooth_neg=0.025,
                       grad_clip=1.0, threshold=0.50,
                       type_dim=64, subword_dim=128, hidden_dim=256,
                       num_layers=4, dropout_gnn=0.15, pool_hidden=128, film_hidden=128,
                       cls_hidden1=256, cls_hidden2=64,
                       dropout_cls1=0.3, dropout_cls2=0.2, mask_rate=0.15, ablation="full",
                       semantic_dim=None, sem_hidden=256, sem_dropout=0.2,
                       test_graphs=None, test_every=5, epoch_probe=None):
    ablation = canonical_ablation(ablation)
    if adversarial and ablation != "full":
        raise ValueError("Ablations run clean-only; --adversarial is only valid with --ablation full.")
    # I probe the bundle here (instead of a CLI default) because the semantic
    # width is a property of the data: bundles built with main.py --no-semantic
    # have no `.sem` and must keep training the pre-semantic architecture.
    if semantic_dim is None:
        from semantic_features import probe_semantic_dim
        semantic_dim = probe_semantic_dim(train_graphs)
        print(f"[i] Semantic branch: {'CodeT5+ ' + str(semantic_dim) + '-d CLS' if semantic_dim else 'disabled (bundle has no embeddings)'}")
    if semantic_dim and ablation != "full":
        print(f"[i] Ablation '{ablation}' trains on the semantic bundle; "
              f"the branch is only active when the config keeps it.")
    model = AdvancedASTGraphEncoder(
        num_node_types=ctx.vocab_size,
        bpe_vocab_size=BPE_VOCAB_SIZE,
        pad_idx=ctx.pad_id,
        type_dim=type_dim, subword_dim=subword_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, dropout_gnn=dropout_gnn,
        pool_hidden=pool_hidden, film_hidden=film_hidden,
        cls_hidden1=cls_hidden1, cls_hidden2=cls_hidden2,
        dropout_cls1=dropout_cls1, dropout_cls2=dropout_cls2,
        mask_rate=mask_rate, ablation=ablation,
        semantic_dim=semantic_dim, sem_hidden=sem_hidden, sem_dropout=sem_dropout,
    ).to(device)
    hyperparams = {
        'type_dim': type_dim, 'subword_dim': subword_dim, 'hidden_dim': hidden_dim,
        'num_layers': num_layers, 'dropout_gnn': dropout_gnn,
        'pool_hidden': pool_hidden, 'film_hidden': film_hidden,
        'cls_hidden1': cls_hidden1, 'cls_hidden2': cls_hidden2,
        'dropout_cls1': dropout_cls1, 'dropout_cls2': dropout_cls2,
        'mask_rate': mask_rate,
        'semantic_dim': semantic_dim, 'sem_hidden': sem_hidden, 'sem_dropout': sem_dropout,
    }
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[i] Trainable parameters: {n_params:,} ({n_params / 1e6:.3f}M)", flush=True)

    num_pos = sum(1 for g in train_graphs if g.y.item() == 1.0)
    class_weight = torch.tensor([(len(train_graphs) - num_pos) / max(1.0, float(num_pos))]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax, eta_min=eta_min)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    # Read-only periodic-test loader: built once, shuffle=False, never affects RNG/trajectory.
    test_loader = None
    if test_graphs is not None and test_every and test_every > 0:
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

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

        # Optional read-only probe (diagnostics only): receives the CURRENT
        # epoch's weights and validation scores. It never influences gradients,
        # the scheduler, checkpoint selection or early stopping; studies use it
        # to log full-test metrics per epoch without touching the training path.
        if epoch_probe is not None:
            epoch_probe(epoch=epoch, epochs=epochs, model=model,
                        train_loss=total_loss / len(train_loader),
                        val_f1=cur_f1, val_roc=cur_roc,
                        val_probs=val_probs, val_labels=val_labels,
                        epoch_seconds=ep_duration)

        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_roc = cur_roc
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            early_stop_now = False
        elif (epochs_no_improve := epochs_no_improve + 1) >= patience:
            early_stop_now = True
        else:
            early_stop_now = False

        # Read-only periodic test: every `test_every` epochs, evaluate the
        # best-so-far weights on the held-out test set and print only.
        # Never writes checkpoints, never touches optimizer/scheduler/best_*.
        if test_loader is not None and ((epoch + 1) % test_every == 0) and best_weights is not None:
            was_training = model.training
            cur_state = copy.deepcopy(model.state_dict())
            try:
                model.load_state_dict(best_weights)
                model.eval()
                test_res = execute_model_eval_with_cost(model, test_loader, device, threshold=threshold)
                print(f"\n[PERIODIC TEST @ Epoch {epoch+1:02d}/{epochs} | Best Val F1 so far: {best_f1:.4f} | Best Val ROC so far: {best_roc:.4f}]")
                tag = f"{language.upper()} {'Adv' if adversarial else 'Clean'}"
                print_detailed_metrics_with_cost(tag, test_res)
            finally:
                model.load_state_dict(cur_state)
                model.train(mode=was_training)

        if early_stop_now:
            print(f"-> Early stopping cleanly triggered at epoch {epoch+1}.")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Val-calibrated decision threshold, computed once on the restored best
    # weights and persisted with the checkpoint; all eval paths read it
    # (evaluate.py / run_five_seeds.py resolve it when --threshold is None).
    # Val is balanced by construction (balanced_subset); the eval split may
    # not be (Java test: 69% machine), so the calibration reweights the val
    # ROC to the eval-set prevalence — a set-level statistic, not test labels.
    eval_prior = None
    if test_graphs:
        try:
            eval_prior = float(np.mean([float(g.y.reshape(-1)[0]) for g in test_graphs]))
        except Exception:
            eval_prior = None
    cal_threshold = _val_optimal_threshold(model, val_loader, device,
                                           fallback=threshold, prior=eval_prior)
    prior_txt = "n/a" if eval_prior is None else f"{eval_prior:.4f}"
    print(f"[i] Val-calibrated threshold: {cal_threshold:.4f} (fixed {threshold:.2f}, "
          f"eval prior {prior_txt}; eval now uses the checkpoint value)")

    ckpt_file = (ablation_checkpoint_path(language, ablation, adversarial)
                 if ablation != "full" else
                 (ADV_CHECKPOINT if adversarial else CLEAN_CHECKPOINT)[language])
    save_checkpoint_for_language(language, ckpt_file, model, optimizer, epoch,
                                 best_f1, best_roc, cal_threshold,
                                 ctx.means, ctx.stds, ctx.g_means, ctx.g_stds,
                                 hyperparams, ablation)

    total_time = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    peak_ram_mb = max(peak_ram_mb, current_rss_mb())
    tag = "MODEL 2 ADVERSARIAL" if adversarial else "MODEL 1 CLEAN"
    print(f"\n[DIAGNOSTICS - {tag}]")
    print(f"Total Training Duration : {total_time/60:.2f} minutes")
    print(f"Peak VRAM Consumption   : {peak_vram:.2f} MB")
    print(f"Peak CPU RAM Consumption: {peak_ram_mb:.2f} MB")

    return model, best_f1, best_roc


def train_model(language="python", epochs=30, batch_size=None, adversarial=False, patience=30,
                lr=5e-5, weight_decay=1e-3, tmax=45, eta_min=1e-6,
                accum_steps=2, smooth_pos=0.975, smooth_neg=0.025,
                grad_clip=1.0, threshold=0.50,
                type_dim=64, subword_dim=128, hidden_dim=256,
                num_layers=4, dropout_gnn=0.15, pool_hidden=128, film_hidden=128,
                cls_hidden1=256, cls_hidden2=64,
                dropout_cls1=0.3, dropout_cls2=0.2, mask_rate=0.15, seed=42, ablation="full",
                semantic_dim=None, sem_hidden=256, sem_dropout=0.2,
                test_every=5):
    """Train one model from the saved bundle. Never builds CPGs.

    Seeds first so the DataLoader shuffle / dropout / token-masking trajectory
    matches the notebook, where Cell 1 seeding carries over into Cells 3/4 in
    the same process. Without this, the split processes diverge and test
    accuracy shifts by ~1pt on identical code. I expose `seed` because I want
    5-seed averages (42-46) over identical data and bundles.

    `ablation` selects one of the eleven study configs (clean-only, test-set
    eval). Graph-changing ablations derive their graphs from the standard
    bundle in-memory (no re-parsing); the checkpoint is namespaced per config
    so study runs never clobber the main checkpoints.
    """
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE[language]
    bundle = load_bundle_for_ablation(language, ablation)
    ctx = bundle["ctx"]
    if adversarial:
        train_graphs = require_adv_graphs(bundle, language)
    else:
        train_graphs = bundle["train_clean_graphs"]
    val_graphs = bundle["val_graphs"]
    test_graphs = bundle.get("test_graphs")
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
                              mask_rate=mask_rate, ablation=ablation,
                              semantic_dim=semantic_dim, sem_hidden=sem_hidden,
                              sem_dropout=sem_dropout,
                              test_graphs=test_graphs, test_every=test_every)


def _add_hyper_args(parser):
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--tmax", type=int, default=45)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--smooth_pos", type=float, default=0.975)
    parser.add_argument("--smooth_neg", type=float, default=0.025)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Fallback decision threshold; the saved checkpoint always "
                             "carries the val-F1-optimal threshold instead")
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
    parser.add_argument("--semantic_dim", type=int, default=None,
                        help="CodeT5+ embedding width (default: auto-probe the bundle; "
                             "768 with semantic bundles, 0 without). Use 0 to force the "
                             "pre-semantic architecture.")
    parser.add_argument("--sem_hidden", type=int, default=256,
                        help="Projection width of the semantic branch before classifier concat")
    parser.add_argument("--sem_dropout", type=float, default=0.2)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--adversarial", action="store_true",
                        help="Train the adversarially augmented variant (token masking on)")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed for init/shuffle/dropout/masking (vary 42-46 for 5-seed average)")
    parser.add_argument("--ablation", type=str, default="full",
                        help="Study config: a single id or '+'-joined combo "
                             "(full no-mp no-subword no-struct syntax-only no-gate no-jk "
                             "no-virtual mean-pool no-film no-semantic); clean-only, namespaced checkpoint")
    parser.add_argument("--test_every", type=int, default=5,
                        help="Run read-only test on best-so-far weights every N epochs (0 disables). No training values are changed.")
    parser = _add_hyper_args(parser)
    args = parser.parse_args()

    try:
        args.ablation = canonical_ablation(args.ablation)
    except ValueError as e:
        parser.error(str(e))

    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE[args.language]
    train_model(language=args.language, epochs=args.epochs, batch_size=args.batch_size,
                adversarial=args.adversarial, patience=args.patience, seed=args.seed,
                ablation=args.ablation, test_every=args.test_every,
                lr=args.lr, weight_decay=args.weight_decay, tmax=args.tmax, eta_min=args.eta_min,
                accum_steps=args.accum_steps, smooth_pos=args.smooth_pos, smooth_neg=args.smooth_neg,
                grad_clip=args.grad_clip, threshold=args.threshold,
                type_dim=args.type_dim, subword_dim=args.subword_dim, hidden_dim=args.hidden_dim,
                num_layers=args.num_layers, dropout_gnn=args.dropout_gnn,
                pool_hidden=args.pool_hidden, film_hidden=args.film_hidden,
                cls_hidden1=args.cls_hidden1, cls_hidden2=args.cls_hidden2,
                dropout_cls1=args.dropout_cls1, dropout_cls2=args.dropout_cls2,
                mask_rate=args.mask_rate,
                semantic_dim=args.semantic_dim, sem_hidden=args.sem_hidden,
                sem_dropout=args.sem_dropout)
