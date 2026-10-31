"""STEP 5 - metrics. y = ground truth (bool), s = scores, both over valid pixels only."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def recall_at_fpr(y, s, fpr_target):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def compute_metrics(y, s):
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "recall@1%fpr": recall_at_fpr(y, s, 0.01),
        "recall@0.1%fpr": recall_at_fpr(y, s, 0.001),
    }


def metrics_by_size(truth, sizes, valid, s, min_count=3):
    """One result row per target size. Other-size targets are excluded from the negatives."""
    t = truth.ravel()[valid]
    z = sizes.ravel()[valid]
    rows = []
    for size in np.unique(z[t]):
        group = t & (z == size)
        if group.sum() < min_count:
            continue
        keep = ~t | group
        rows.append({"size": float(size), "n_pixels": int(group.sum()),
                     **compute_metrics(group[keep], s[keep])})
    return rows