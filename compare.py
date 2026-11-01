import sys
import numpy as np
import matplotlib.pyplot as plt
from src.loader import load_scene
from src.viz import false_color, heatmap_rgb, overlay_truth

scene, kind, npy, out = sys.argv[1:5]
sc = load_scene(scene, kind)
smap = np.load(npy)
rgb = false_color(sc["cube"], sc["wavelengths"])

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(rgb, interpolation="nearest")
ax[0].set_title("Scene")
ax[1].imshow(overlay_truth(rgb, sc["truth"]), interpolation="nearest")
ax[1].set_title("Scene + true targets (red)")
ax[2].imshow(heatmap_rgb(smap), interpolation="nearest")
ax[2].contour(sc["truth"], levels=[0.5], colors="cyan", linewidths=0.8)
ax[2].set_title("Heatmap + true targets (cyan)")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(out, dpi=150)