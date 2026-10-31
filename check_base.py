import scipy.io as sio
from src.preprocess import prepare, scores_to_map
from src.detectors import RX

hsi = sio.loadmat('data/muufl_gulfport_campus_w_lidar_1.mat',
                  struct_as_record=False, squeeze_me=True)['hsi']
rows = hsi.groundTruth.Targets_rowIndices.astype(int)
cols = hsi.groundTruth.Targets_colIndices.astype(int)

X, valid, shape = prepare(hsi.Data)
score_map = scores_to_map(RX().fit_score(X), valid, shape)

for base in (0, 1):
    print("base", base, "-> mean RX score at targets:",
          score_map[rows - base, cols - base].mean())
          