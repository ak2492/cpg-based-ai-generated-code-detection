"""Eval-time validation threshold calibration, matching the CPG/hybrid rule.

The legacy checkpoints store whatever `train.py --threshold` was given -- 0.50
in the five-seed protocol -- so they do not carry a validation-optimal cut.
The CPG folder and the hybrid folder both calibrate a cut from clean
validation and use it for every eval row. To make the legacy attack table
joinable with those two tables, the same cut has to be derived here at eval
time.

The rule is copied verbatim from
`cpg-based-ai-generated-code-detection/train.py::_select_threshold`:

  * F1-argmax on clean validation probabilities,
  * the val ROC reweighted to the evaluation-set positive prevalence,
  * the calibration rule itself cross-validated on the val scores,
  * adopted only when held-out F1 beats the fixed fallback by >= 5e-4.

Nothing in this module trains or writes a checkpoint. Calibration only reads
the clean validation split, so attacked samples never influence the cut.
"""
import gc

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch_geometric.loader import DataLoader

from language_configs import ADV_CHECKPOINT, CLEAN_CHECKPOINT, DEFAULT_BATCH_SIZE
from model import build_encoder_from_checkpoint


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


def _select_threshold(probs, labels, fallback=0.50, prior=None, n_folds=5,
                      min_gain=5e-4):
    """Pick a decision threshold, but only if calibration beats the fixed cut.

    This is the CPG rule verbatim. The val split is balanced by construction,
    while the eval split may not be (Java test is 69% machine), so `prior`
    reweights the val ROC to the eval-set prevalence. The calibration RULE is
    cross-validated on the val scores and kept only when held-out F1 beats
    the fixed fallback by `min_gain`.
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


def test_prior(language, bundle=None):
    """Positive prevalence of the clean test split (CPG's `eval_prior`).

    This is a set-level statistic computed from labels in the saved bundle,
    never from attacked data and never used for training.
    """
    owns_bundle = bundle is None
    if owns_bundle:
        from pipeline import load_bundle
        bundle = load_bundle(language)
    try:
        ys = []
        for graph in bundle["test_graphs"]:
            try:
                ys.append(float(graph.y.reshape(-1)[0]))
            except Exception:
                continue
        return float(np.mean(ys)) if ys else None
    finally:
        if owns_bundle:
            del bundle
            gc.collect()


def calibrate_threshold(bundle, checkpoint, adversarial=False, fallback=0.50,
                        prior=None, batch_size=None, device=None):
    """Return the clean-validation cut for one checkpoint.

    `bundle` must already be loaded (and must match the checkpoint's ablation);
    this keeps callers from paying for a second bundle load. The model and
    loader are released before returning so the caller can reuse the GPU.
    """
    language = bundle["ctx"].language
    if checkpoint is None:
        checkpoint = (ADV_CHECKPOINT if adversarial else CLEAN_CHECKPOINT)[language]
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE[language]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = build_encoder_from_checkpoint(bundle["ctx"], checkpoint, device)
    model.eval()
    loader = DataLoader(bundle["val_graphs"], batch_size=batch_size, shuffle=False)

    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probs.extend(torch.sigmoid(model(batch)).cpu().numpy().flatten())
            labels.extend(batch.y.cpu().numpy().flatten())

    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    threshold = _select_threshold(probs, labels, fallback=fallback, prior=prior)

    del loader, model, probs, labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return threshold


def compute_seed_threshold(language, adversarial=False, checkpoint=None,
                           fallback=0.50, prior=None, batch_size=None,
                           device=None, bundle=None):
    """Convenience wrapper that loads the bundle when the caller has none."""
    owns_bundle = bundle is None
    if owns_bundle:
        from pipeline import load_bundle
        bundle = load_bundle(language)
    try:
        if prior is None:
            prior = test_prior(language, bundle=bundle)
        return calibrate_threshold(bundle, checkpoint, adversarial=adversarial,
                                   fallback=fallback, prior=prior,
                                   batch_size=batch_size, device=device)
    finally:
        if owns_bundle:
            del bundle
            gc.collect()
