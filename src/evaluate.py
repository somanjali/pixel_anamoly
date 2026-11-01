"""STEP 5 - metrics.
Ground truth is one pixel per panel, so small registration/GPS errors would count as misses.
`radius` (default 1) scores every pixel by the best score within that many pixels and drops the
target's neighbourhood from the negatives. radius=0 is the strict single-pixel evaluation."""
import numpy as np
from scipy.ndimage import binary_dilation, maximum_filter
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


def tolerant_arrays(score_map, pos, truth_all, valid_map, radius=1):
    """Return (y, s). Positives = pixels in `pos`; negatives = valid pixels outside every target's
    neighbourhood. With radius > 0 EVERY pixel is scored by the max score within `radius` pixels,
    positives and negatives alike. (Giving only the targets the local max biases the metric:
    on pure noise it scores ~0.78 instead of 0.5.)"""
    pos = pos & valid_map
    if radius > 0:
        near = binary_dilation(truth_all, structure=np.ones((3, 3), bool), iterations=radius)
        score_map = maximum_filter(score_map, size=2 * radius + 1)
    else:
        near = truth_all
    neg = ~near & valid_map
    sp, sn = score_map[pos], score_map[neg]
    y = np.concatenate([np.ones(len(sp), bool), np.zeros(len(sn), bool)])
    return y, np.concatenate([sp, sn])


def metrics_by_size(truth, sizes, valid_map, score_map, radius=1, min_count=3):
    """One result row per target size. Other-size targets are excluded from the negatives."""
    rows = []
    for size in np.unique(sizes[truth]):
        group = truth & (sizes == size)
        if (group & valid_map).sum() < min_count:
            continue
        y, s = tolerant_arrays(score_map, group, truth, valid_map, radius)
        rows.append({"size": float(size), "n_pixels": int(group.sum()), **compute_metrics(y, s)})
    return rows