"""Construction du Vision Transformer."""
from __future__ import annotations

from transformers import ViTConfig, ViTForImageClassification

from .data import CLASSES

DEFAULT_CHECKPOINT = "google/vit-base-patch16-224-in21k"


def build_model(checkpoint: str | None = DEFAULT_CHECKPOINT, config: ViTConfig | None = None,
                freeze_layers: int = 0) -> ViTForImageClassification:
    """Crée un ViT avec une tête de classification à 10 classes.

    - checkpoint : modèle pré-entraîné sur ImageNet-21k (transfer learning)
    - config     : configuration explicite (utilisée pour les tests avec un mini-ViT)
    - freeze_layers : nombre de premiers blocs Transformer à geler
    """
    id2label = dict(enumerate(CLASSES))
    label2id = {v: k for k, v in id2label.items()}
    if config is not None:
        config.num_labels, config.id2label, config.label2id = len(CLASSES), id2label, label2id
        # "eager" pour pouvoir récupérer les cartes d'attention
        model = ViTForImageClassification(config)
    else:
        model = ViTForImageClassification.from_pretrained(
            checkpoint, num_labels=len(CLASSES), id2label=id2label, label2id=label2id, attn_implementation="eager"
        )

    if freeze_layers:
        for p in model.vit.embeddings.parameters():
            p.requires_grad = False
        # transformers 5 : vit.layers ; versions 4.x : vit.encoder.layer
        blocks = model.vit.layers if hasattr(model.vit, "layers") else model.vit.encoder.layer
        for layer in blocks[:freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
    return model


def tiny_config(image_size: int = 32) -> ViTConfig:
    """Mini-ViT (quelques milliers de paramètres) pour les tests et la CI."""
    return ViTConfig(
        image_size=image_size, patch_size=8, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=64, attn_implementation="eager",
    )


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
