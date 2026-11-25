"""Prédiction sur une image + carte d'attention.

    python -m vitsat.predict outputs/best image.jpg
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import torch
from PIL import Image

from .attention import attention_rollout, overlay
from .data import LABELS_FR, build_transforms
from .model import build_model


@dataclass
class Prediction:
    label: str
    label_fr: str
    score: float


class Predictor:
    def __init__(self, model_dir: str | None = None, model=None, device: str = "cpu"):
        self.model = (model or build_model(model_dir)).to(device).eval()
        self.device = device
        self.transform = build_transforms(False, self.model.config.image_size)

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 3) -> tuple[list[Prediction], Image.Image]:
        """Retourne les top_k classes avec leur probabilité et l'image avec la carte d'attention."""
        image = image.convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        out = self.model(pixel_values=x, output_attentions=True)
        probs = out.logits.softmax(-1)[0].cpu()
        scores, ids = probs.topk(min(top_k, probs.numel()))
        labels = self.model.config.id2label
        preds = [Prediction(labels[i], LABELS_FR.get(labels[i], labels[i]), float(s)) for s, i in zip(scores, ids.tolist(), strict=True)]
        mask = attention_rollout([a.cpu() for a in out.attentions], discard_ratio=0.9)[0].numpy()
        return preds, overlay(image, mask)


if __name__ == "__main__":
    model_dir, path = sys.argv[1], sys.argv[2]
    preds, heat = Predictor(model_dir).predict(Image.open(path))
    for p in preds:
        print(f"{p.label_fr:<24} {p.score:.1%}")
    heat.save("attention.png")
    print("Carte d'attention enregistrée dans attention.png")
