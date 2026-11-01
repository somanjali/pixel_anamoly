"""STEP 8 - demo UI.   Run:  python app.py   then open the address it prints.
Upload a scene, pick a detector, click Run. Shows the scene, the anomaly heatmap, and (if the file
contains target labels) the heatmap with the true targets ringed in red plus the scores.
Only fast detectors run live; the autoencoder is trained offline with run_experiments.py."""
import numpy as np
import gradio as gr
from src.detectors import DETECTORS
from src.evaluate import compute_metrics, tolerant_arrays
from src.loader import load_scene
from src.preprocess import prepare, scores_to_map
from src.viz import false_color, heatmap_rgb, overlay_truth

LIVE = ["localrx", "localrx_wide", "rx", "iforest", "ocsvm"]
SHOW_PX = 600                                       # small scenes are enlarged to about this many pixels
VIEW_H = 450                                        # height of each image box on the page


def enlarge(img, target=SHOW_PX):
    """Whole-number nearest-neighbour upscale, so every pixel stays a sharp square when you zoom."""
    f = max(1, target // max(img.shape[:2]))
    return np.repeat(np.repeat(img, f, axis=0), f, axis=1)


def run(file, kind, detector, use_pca):
    if file is None:
        raise gr.Error("Upload a .mat scene first.")
    try:
        sc = load_scene(file, kind)
    except Exception as e:
        raise gr.Error(f"Could not read this file as format '{kind}': {e}")

    X, valid, shape = prepare(sc["cube"], n_components=20 if use_pca else None,
                              valid_mask=sc.get("valid_mask"))
    model = DETECTORS[detector](seed=0)
    scores = model.fit_score(X, valid, shape) if getattr(model, "needs_grid", False) else model.fit_score(X)
    smap = scores_to_map(scores, valid, shape)

    rgb = false_color(sc["cube"], sc["wavelengths"])
    heat = heatmap_rgb(smap)
    truth, valid_map = sc["truth"], valid.reshape(shape)
    if (truth & valid_map).any():
        y, s = tolerant_arrays(smap, truth, truth, valid_map, 0)
        m = compute_metrics(y, s)
        text = (f"{detector}: ROC-AUC {m['roc_auc']:.3f}  |  PR-AUC {m['pr_auc']:.3f}  |  "
                f"recall @1% FPR {m['recall@1%fpr']:.0%}  |  recall @0.1% FPR {m['recall@0.1%fpr']:.0%}")
        heat_t = overlay_truth(heat, truth, iterations=1)
    else:
        text, heat_t = "No target labels in this file, so no scores are shown.", heat
    return enlarge(rgb), enlarge(heat), enlarge(heat_t), text


with gr.Blocks(title="SpectraQ") as demo:
    gr.Markdown("# SpectraQ\nUpload a hyperspectral scene to see which pixels don't fit the background. "
                "Bright areas of the heatmap are the most unusual pixels.")
    with gr.Row():
        file = gr.File(label="Hyperspectral scene (.mat)", type="filepath")
        with gr.Column():
            kind = gr.Dropdown(["abu", "muufl"], value="abu", label="Scene format")
            detector = gr.Dropdown(LIVE, value="localrx", label="Detector")
            use_pca = gr.Checkbox(value=False, label="PCA to 20 components")
            go = gr.Button("Run", variant="primary")
    scores_box = gr.Textbox(label="Scores (computed from the labels in the file, if any)", interactive=False)
    with gr.Row():
        out_rgb = gr.Image(label="Original (false colour)", height=VIEW_H, interactive=False)
        out_heat = gr.Image(label="Anomaly heatmap", height=VIEW_H, interactive=False)
        out_truth = gr.Image(label="Heatmap + true targets (red rings)", height=VIEW_H, interactive=False)
    go.click(run, [file, kind, detector, use_pca], [out_rgb, out_heat, out_truth, scores_box])

if __name__ == "__main__":
    demo.launch()