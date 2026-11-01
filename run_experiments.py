"""STEP 7 - run every detector, several seeds, with and without PCA; write CSVs + score maps.
Usage:  python run_experiments.py --scene data/<file>.mat --kind muufl --seeds 5 --pca 0,20
Options: --tol 1 (target tolerance in pixels; 0 = strict)   --drop-edge 4 (drop N noisy bands at each end)"""
import argparse
import os
import time
import numpy as np
import pandas as pd
from src.detectors import DETECTORS, DETERMINISTIC
from src.evaluate import compute_metrics, metrics_by_size, tolerant_arrays
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
    ap.add_argument("--tol", type=int, default=1, help="target tolerance in pixels (0 = strict)")
    ap.add_argument("--drop-edge", type=int, default=0, help="drop this many bands at each end")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.scene))[0]
    tag = f"{name}_tol{a.tol}_edge{a.drop_edge}"
    sc = load_scene(a.scene, a.kind)
    truth = sc["truth"]
    n_bands = sc["cube"].shape[2]
    bands = np.arange(a.drop_edge, n_bands - a.drop_edge) if a.drop_edge else None
    rows, size_rows = [], []

    for n_pca in [int(x) for x in a.pca.split(",")]:
        X, valid, shape = prepare(sc["cube"], bands=bands, n_components=n_pca or None,
                                  valid_mask=sc.get("valid_mask"))
        valid_map = valid.reshape(shape)
        if not (truth & valid_map).any():
            raise SystemExit("No ground-truth pixels found - check src/loader.py with check_scene.py")
        for det in a.detectors.split(","):
            seeds = [0] if det in DETERMINISTIC else range(a.seeds)
            for seed in seeds:
                t0 = time.time()
                model = DETECTORS[det](seed=seed)
                s = model.fit_score(X, valid, shape) if getattr(model, "needs_grid", False) else model.fit_score(X)
                smap = scores_to_map(s, valid, shape)
                y, s_eval = tolerant_arrays(smap, truth, truth, valid_map, a.tol)
                rows.append(dict(scene=name, detector=det, pca=n_pca, seed=seed,
                                 seconds=round(time.time() - t0, 1), **compute_metrics(y, s_eval)))
                print(rows[-1])
                if seed == 0:
                    np.save(f"{a.out}/{tag}_{det}_pca{n_pca}.npy", smap)
                    if sc["sizes"] is not None:
                        size_rows += [dict(scene=name, detector=det, pca=n_pca, **r)
                                      for r in metrics_by_size(truth, sc["sizes"], valid_map, smap, a.tol)]

    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/{tag}_runs.csv", index=False)
    summary = df.groupby(["scene", "detector", "pca"])[METRICS].agg(["mean", "std"]).round(3)
    summary.to_csv(f"{a.out}/{tag}_summary.csv")
    print(f"\n[tol={a.tol}, drop_edge={a.drop_edge}]\n", summary.to_string())
    if size_rows:
        pd.DataFrame(size_rows).to_csv(f"{a.out}/{tag}_by_size.csv", index=False)


if __name__ == "__main__":
    main()