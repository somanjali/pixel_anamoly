"""STEP 1 - print everything inside a .mat file so you know the real key names.
Usage:  python inspect_mat.py data/muufl_gulfport_campus_w_lidar_1.mat
(Not called inspect.py on purpose: that name would shadow a Python standard-library module.)
"""
import sys
import scipy.io as sio


def walk(obj, name=""):
    if hasattr(obj, "_fieldnames"):                      # MATLAB struct
        for f in obj._fieldnames:
            walk(getattr(obj, f), f"{name}.{f}" if name else f)
    else:
        print(f"{name:45s} {str(getattr(obj, 'shape', type(obj))):22s} {getattr(obj, 'dtype', '')}")


path = sys.argv[1]
try:
    m = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    for key, val in m.items():
        if not key.startswith("__"):
            walk(val, key)
except NotImplementedError:                              # MATLAB v7.3 files are HDF5
    import h5py
    with h5py.File(path, "r") as f:
        f.visititems(lambda n, o: print(f"{n:45s} {getattr(o, 'shape', '')}"))