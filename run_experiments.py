"""STEP 7 - run every detector, several seeds, with and without PCA; write CSVs + score maps.
Usage:  python run_experiments.py --scene data/<file>.mat --kind muufl --seeds 5 --pca 0,20"""
import argparse
import os
import time
import numpy as np
import pandas as pd
from src.detectors import DETECTORS, DETERMINISTIC
from src.evaluate import compute_metrics, metrics_by_size
from src.loader import load_scene
from src.preprocess import prepare, scores_to_map

METRICS = ["roc_auc", "pr_auc", "recall@1%fpr", "recall@0.1%fpr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--kind", required=True, choices=["muufl", "abu"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pca", default="0,20", help="comma list; 0 = no PCA")
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.scene))[0]
    sc = load_scene(a.scene, a.kind)
    rows, size_rows = [], []

    for n_pca in [int(x) for x in a.pca.split(",")]:
        X, valid, shape = prepare(sc["cube"], n_components=n_pca or None)
        y = sc["truth"].ravel()[valid]
        if y.sum() == 0:
            raise SystemExit("No ground-truth pixels found - check src/loader.py with check_scene.py")
        for det in a.detectors.split(","):
            seeds = [0] if det in DETERMINISTIC else range(a.seeds)
            for seed in seeds:
                t0 = time.time()
                s = DETECTORS[det](seed=seed).fit_score(X)
                rows.append(dict(scene=name, detector=det, pca=n_pca, seed=seed,
                                 seconds=round(time.time() - t0, 1), **compute_metrics(y, s)))
                print(rows[-1])
                if seed == 0:
                    np.save(f"{a.out}/{name}_{det}_pca{n_pca}.npy", scores_to_map(s, valid, shape))
                    if sc["sizes"] is not None:
                        size_rows += [dict(scene=name, detector=det, pca=n_pca, **r)
                                      for r in metrics_by_size(sc["truth"], sc["sizes"], valid, s)]

    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/{name}_runs.csv", index=False)
    summary = df.groupby(["scene", "detector", "pca"])[METRICS].agg(["mean", "std"]).round(3)
    summary.to_csv(f"{a.out}/{name}_summary.csv")
    print("\n", summary.to_string())
    if size_rows:
        pd.DataFrame(size_rows).to_csv(f"{a.out}/{name}_by_size.csv", index=False)


if __name__ == "__main__":
    main()