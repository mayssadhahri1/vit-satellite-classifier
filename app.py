"""Démo web Gradio : déposez une image satellite, le ViT prédit le type de terrain
et montre les zones qui ont guidé sa décision.

    python app.py                       # utilise outputs/best
    MODEL_DIR=chemin/du/modele python app.py
"""
import os

import gradio as gr

from vitsat.predict import Predictor

MODEL_DIR = os.getenv("MODEL_DIR", "outputs/best")
predictor = Predictor(MODEL_DIR)


def classify(image):
    if image is None:
        return {}, None
    preds, heatmap = predictor.predict(image, top_k=3)
    return {p.label_fr: p.score for p in preds}, heatmap


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Image satellite"),
    outputs=[gr.Label(num_top_classes=3, label="Type de terrain"), gr.Image(label="Carte d'attention du ViT")],
    title="🛰️ ViT-Sat — classification d'images satellite",
    description="Vision Transformer fine-tuné sur EuroSAT (Sentinel-2, 10 classes). "
    "La carte d'attention (attention rollout) montre les zones de l'image qui ont le plus compté.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
