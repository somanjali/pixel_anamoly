"""STEP 2b - confirm the ground truth lines up with the image BEFORE running any model.
Usage:  python check_scene.py data/<file>.mat muufl
Red rings should sit on the visible target panels. If they look shifted or mirrored,
rows/cols are swapped or the index base is wrong in src/loader.py."""
import os
import sys
import matplotlib.pyplot as plt
from src.loader import load_scene
from src.viz import false_color, overlay_truth

path, kind = sys.argv[1], sys.argv[2]
sc = load_scene(path, kind)
print("cube:", sc["cube"].shape, "| truth pixels:", int(sc["truth"].sum()))
os.makedirs("results", exist_ok=True)
out = f"results/check_{os.path.splitext(os.path.basename(path))[0]}.png"
plt.imsave(out, overlay_truth(false_color(sc["cube"], sc["wavelengths"]), sc["truth"]))
print("saved", out)