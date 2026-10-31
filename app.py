"""STEP 8 - demo UI.  Run:  python app.py   (Gradio opens in the browser)
Only fast detectors run live. Train the autoencoder offline with run_experiments.py and show its saved maps."""
import gradio as gr
from src.detectors import DETECTORS
from src.loader import load_scene
from src.preprocess import prepare, scores_to_map
from src.viz import false_color, heatmap_rgb

LIVE = ["rx", "iforest", "ocsvm"]


def run(file, kind, detector, use_pca):
    if file is None:
        raise gr.Error("Upload a .mat scene first.")
    sc = load_scene(file, kind)
    X, valid, shape = prepare(sc["cube"], n_components=20 if use_pca else None)
    scores = DETECTORS[detector](seed=0).fit_score(X)
    return false_color(sc["cube"], sc["wavelengths"]), heatmap_rgb(scores_to_map(scores, valid, shape))


demo = gr.Interface(
    fn=run,
    inputs=[gr.File(label="Hyperspectral scene (.mat)"),
            gr.Dropdown(["muufl", "abu"], value="muufl", label="Scene format"),
            gr.Dropdown(LIVE, value="rx", label="Detector"),
            gr.Checkbox(value=False, label="PCA to 20 components")],
    outputs=[gr.Image(label="Original (false colour)"), gr.Image(label="Anomaly heatmap")],
    title="SpectraQ",
    description="Upload a hyperspectral scene to see which pixels don't fit the background.",
)

if __name__ == "__main__":
    demo.launch()