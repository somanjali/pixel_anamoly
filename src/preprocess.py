"""STEP 3 - cube -> 2D matrix for the models, and scores -> 2D heatmap.
Order: select bands -> reshape -> mask invalid pixels -> standardise -> (optional) PCA."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def good_bands(wavelengths, bad_ranges=((1340, 1470), (1790, 1960))):
    """Indices of bands outside the water-absorption windows. Matters for SWIR sensors (e.g. AVIRIS-NG);
    a no-op for visible/near-infrared-only data."""
    wl = np.asarray(wavelengths, float)
    if wl.max() < 10:                                    # micrometres -> nanometres
        wl = wl * 1000
    keep = np.ones(len(wl), bool)
    for lo, hi in bad_ranges:
        keep &= ~((wl >= lo) & (wl <= hi))
    return np.where(keep)[0]


def prepare(cube, bands=None, n_components=None, valid_mask=None):
    """Returns X (n_valid_pixels, n_features), valid (H*W,) bool mask, (H, W)."""
    cube = cube.astype(np.float32)
    if bands is not None:
        cube = cube[:, :, bands]
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)                           # row-major: pixel i = (i // w, i % w)
    valid = np.isfinite(flat).all(axis=1) & (np.abs(flat).sum(axis=1) > 0)
    if valid_mask is not None:                           # the file's own mask of real pixels
        valid &= np.asarray(valid_mask, bool).ravel()
    X = StandardScaler().fit_transform(flat[valid])
    if n_components:
        X = PCA(n_components=n_components, random_state=0).fit_transform(X)
    return X.astype(np.float32), valid, (h, w)


def scores_to_map(scores, valid, shape):
    """Put scores back on the pixel grid; invalid pixels get the lowest score."""
    full = np.full(valid.shape, np.min(scores), dtype=np.float32)
    full[valid] = scores
    return full.reshape(shape)