"""Boucles d'entraînement et d'évaluation."""
from __future__ import annotations

import torch
from torch import nn


def train_one_epoch(model, loader, optimizer, scheduler, device, scaler=None, label_smoothing: float = 0.1):
    model.train()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        # précision mixte automatique sur GPU (plus rapide, moins de mémoire)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
            logits = model(pixel_values=images).logits
            loss = loss_fn(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, device):
    """Retourne (perte, précision, prédictions, vraies classes)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, preds, targets = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(pixel_values=images).logits
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        preds += logits.argmax(1).tolist()
        targets += labels.tolist()
    acc = sum(p == t for p, t in zip(preds, targets, strict=True)) / len(targets)
    return total_loss / len(targets), acc, preds, targets
