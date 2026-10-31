"""STEP 2 - turn a downloaded file into a standard scene dict:
    cube        (H, W, B) float32
    wavelengths (B,) or None
    truth       (H, W) bool   - True where a real anomaly/target is (used ONLY for evaluation)
    sizes       (H, W) float or None - target size per truth pixel, for the per-size breakdown
"""
import numpy as np
import scipy.io as sio

# ---- MUUFL: fill these in AFTER running inspect_mat.py (dotted paths, e.g. "hsi.Data") ----
MUUFL_KEYS = {
   
    "cube": "hsi.Data",
    "wavelengths": "hsi.info.wavelength",
    "target_rows": "hsi.groundTruth.Targets_rowIndices",
    "target_cols": "hsi.groundTruth.Targets_colIndices",
    "target_sizes": "hsi.groundTruth.Targets_Size",
}
MUUFL_INDEX_BASE = 1 # optional: size of each panel
     # MATLAB indices start at 1; set 0 if the file is already 0-based


def _load_mat(path):
    try:
        return sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    except NotImplementedError:
        raise RuntimeError("This is a MATLAB v7.3 (HDF5) file. Open it with h5py and adapt this loader.")


def _get(obj, dotted):
    for part in dotted.split("."):
        obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
    return obj


def load_muufl(path):
    m = _load_mat(path)
    k = MUUFL_KEYS
    if k["cube"] is None or k["target_rows"] is None or k["target_cols"] is None:
        raise RuntimeError("Fill in MUUFL_KEYS in src/loader.py using the output of inspect_mat.py")
    cube = np.asarray(_get(m, k["cube"]), dtype=np.float32)
    h, w = cube.shape[:2]
    wl = np.asarray(_get(m, k["wavelengths"])).ravel() if k["wavelengths"] else None
    rows = np.atleast_1d(_get(m, k["target_rows"])).astype(int) - MUUFL_INDEX_BASE
    cols = np.atleast_1d(_get(m, k["target_cols"])).astype(int) - MUUFL_INDEX_BASE
    ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    truth = np.zeros((h, w), bool)
    truth[rows[ok], cols[ok]] = True
    sizes = None
    if k["target_sizes"]:
        s = np.atleast_1d(_get(m, k["target_sizes"])).astype(float)
        sizes = np.zeros((h, w), float)
        sizes[rows[ok], cols[ok]] = s[ok]
    return dict(cube=cube, wavelengths=wl, truth=truth, sizes=sizes)


def load_abu(path):
    """ABU benchmark scenes (Airport-Beach-Urban). Expects keys 'data' and 'map' - verify with inspect_mat.py."""
    m = sio.loadmat(path)
    cube = np.asarray(m["data"], dtype=np.float32)
    truth = np.asarray(m["map"]) > 0
    return dict(cube=cube, wavelengths=None, truth=truth, sizes=None)


LOADERS = {"muufl": load_muufl, "abu": load_abu}


def load_scene(path, kind):
    return LOADERS[kind](path)