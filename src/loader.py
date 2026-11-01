"""STEP 2 - turn a downloaded file into a standard scene dict:
    cube        (H, W, B) float32
    wavelengths (B,) or None
    truth       (H, W) bool   - True where a real anomaly/target is (used ONLY for evaluation)
    sizes       (H, W) float or None - target size per truth pixel, for the per-size breakdown
    valid_mask  (H, W) bool or None  - pixels that hold real data
"""
import numpy as np
import scipy.io as sio

# ---- MUUFL field paths (filled in from inspect_mat.py output) ----
MUUFL_KEYS = {
    "cube": "hsi.Data",
    "wavelengths": "hsi.info.wavelength",
    "target_rows": "hsi.groundTruth.Targets_rowIndices",
    "target_cols": "hsi.groundTruth.Targets_colIndices",
    "target_sizes": "hsi.groundTruth.Targets_Size",
    "valid_mask": "hsi.valid_mask",
}
MUUFL_INDEX_BASE = 1       # confirmed 1-based with check_base.py


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
    vm = None
    if k.get("valid_mask"):
        vm = np.asarray(_get(m, k["valid_mask"])) > 0
    return dict(cube=cube, wavelengths=wl, truth=truth, sizes=sizes, valid_mask=vm)


def load_abu(path):
    """ABU benchmark scenes (Airport-Beach-Urban). Expects keys 'data' and 'map' - verify with inspect_mat.py."""
    m = sio.loadmat(path)
    cube = np.asarray(m["data"], dtype=np.float32)
    truth = np.asarray(m["map"]) > 0
    return dict(cube=cube, wavelengths=None, truth=truth, sizes=None, valid_mask=None)


LOADERS = {"muufl": load_muufl, "abu": load_abu}


def load_scene(path, kind):
    return LOADERS[kind](path)