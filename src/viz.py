"""STEP 6 - images for the UI and for sanity checks."""
import matplotlib
import numpy as np
from scipy.ndimage import binary_dilation


def false_color(cube, wavelengths=None):
    b = cube.shape[2]
    if wavelengths is not None and len(wavelengths) == b:
        wl = np.asarray(wavelengths, float)
        if wl.max() < 10:
            wl = wl * 1000
        idx = [int(np.argmin(np.abs(wl - t))) for t in (640, 550, 460)]   # approx R, G, B
    else:
        idx = [int(b * 0.7), int(b * 0.4), int(b * 0.15)]
    rgb = cube[:, :, idx].astype(np.float32)
    lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1))
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1)
    return (rgb * 255).astype(np.uint8)


def heatmap_rgb(score_map, p_low=1, p_high=99.5, cmap="inferno"):
    lo, hi = np.percentile(score_map, [p_low, p_high])
    x = np.clip((score_map - lo) / (hi - lo + 1e-12), 0, 1)
    return (matplotlib.colormaps[cmap](x)[..., :3] * 255).astype(np.uint8)


def overlay_truth(rgb, truth, iterations=2):
    out = rgb.copy()
    ring = binary_dilation(truth, iterations=iterations) & ~truth
    out[ring] = (255, 0, 0)
    return out