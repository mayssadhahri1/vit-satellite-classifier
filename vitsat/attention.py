"""Explicabilité : « attention rollout » (Abnar & Zuidema, 2020).

On combine les matrices d'attention de toutes les couches pour savoir quelles zones de
l'image ont le plus influencé le token [CLS], donc la décision du modèle.
"""
from __future__ import annotations

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image


def attention_rollout(attentions, discard_ratio: float = 0.0) -> torch.Tensor:
    """attentions : tuple (une entrée par couche) de tenseurs (B, têtes, T, T).

    Retourne une carte (B, g, g) normalisée entre 0 et 1, où g × g est la grille de patchs.
    """
    batch, tokens = attentions[0].shape[0], attentions[0].shape[-1]
    result = torch.eye(tokens).expand(batch, tokens, tokens).clone()
    for layer in attentions:
        a = layer.mean(dim=1)  # moyenne des têtes
        if discard_ratio > 0:
            # on ignore les liens d'attention les plus faibles (réduit le bruit)
            flat = a.view(batch, -1)
            k = int(flat.shape[1] * discard_ratio)
            low = flat.topk(k, dim=1, largest=False).indices
            flat.scatter_(1, low, 0)
        a = a + torch.eye(tokens)  # connexion résiduelle
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result

    mask = result[:, 0, 1:]  # attention du [CLS] vers chaque patch
    grid = int(mask.shape[-1] ** 0.5)
    mask = mask.reshape(batch, grid, grid)
    mn = mask.amin(dim=(1, 2), keepdim=True)
    mx = mask.amax(dim=(1, 2), keepdim=True)
    return (mask - mn) / (mx - mn).clamp_min(1e-8)


def overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Superpose la carte d'attention (colormap « inferno ») sur l'image d'origine."""
    image = image.convert("RGB")
    heat = Image.fromarray(np.uint8(mask * 255)).resize(image.size, Image.BICUBIC)
    colored = colormaps["inferno"](np.asarray(heat) / 255.0)[..., :3]
    blended = (1 - alpha) * np.asarray(image) / 255.0 + alpha * colored
    return Image.fromarray(np.uint8(np.clip(blended, 0, 1) * 255))
