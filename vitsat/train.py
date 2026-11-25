"""Fine-tuning du ViT sur EuroSAT.

Exemple :
    python -m vitsat.train --epochs 5 --batch-size 32
    python -m vitsat.train --subset 0.1 --epochs 1      # essai rapide
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from .data import CLASSES, make_loaders
from .engine import evaluate, train_one_epoch
from .model import DEFAULT_CHECKPOINT, build_model, count_parameters, tiny_config


def cosine_with_warmup(optimizer, warmup: int, total: int):
    """Montée linéaire du learning rate puis décroissance en cosinus."""
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fine-tuning d'un ViT sur EuroSAT")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--freeze-layers", type=int, default=0, help="blocs Transformer à geler")
    p.add_argument("--subset", type=float, default=1.0, help="fraction du jeu de données (essais rapides)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tiny", action="store_true", help="mini-ViT non pré-entraîné en 32×32 (test rapide du pipeline)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    image_size = 32 if args.tiny else 224
    train_dl, val_dl, test_dl = make_loaders(args.data_dir, image_size, args.batch_size, args.num_workers, args.subset, args.seed)
    config = tiny_config(image_size) if args.tiny else None
    model = build_model(args.checkpoint, config=config, freeze_layers=args.freeze_layers).to(device)
    total, trainable = count_parameters(model)
    print(f"Appareil : {device} | paramètres : {total / 1e6:.1f} M (entraînables : {trainable / 1e6:.1f} M)")
    print(f"Images : {len(train_dl.dataset)} train / {len(val_dl.dataset)} val / {len(test_dl.dataset)} test")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(train_dl)
    scheduler = cosine_with_warmup(optimizer, int(args.warmup_ratio * steps), steps)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    best_acc, history = 0.0, []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, scheduler, device, scaler)
        va_loss, va_acc, _, _ = evaluate(model, val_dl, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        duration = time.time() - start
        print(f"Époque {epoch}/{args.epochs} | train {tr_loss:.3f} / {tr_acc:.2%} | val {va_loss:.3f} / {va_acc:.2%} | {duration:.0f}s")
        if va_acc > best_acc:
            # on garde uniquement le meilleur modèle sur la validation
            best_acc = va_acc
            model.save_pretrained(os.path.join(args.output_dir, "best"))

    # Évaluation finale du meilleur modèle sur le jeu de test (jamais vu pendant l'entraînement)
    model = build_model(os.path.join(args.output_dir, "best")).to(device)
    te_loss, te_acc, preds, targets = evaluate(model, test_dl, device)
    report = classification_report(targets, preds, target_names=CLASSES, digits=4, output_dict=True)
    print(classification_report(targets, preds, target_names=CLASSES, digits=4))

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"best_val_acc": best_acc, "test_acc": te_acc, "test_loss": te_loss, "history": history, "report": report}, f, indent=2)
    save_plots(history, confusion_matrix(targets, preds), args.output_dir)
    print(f"Précision sur le test : {te_acc:.2%} — résultats dans {args.output_dir}/")


def save_plots(history, cm, out_dir):
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    ax[0].plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax[0].set_title("Perte")
    ax[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    ax[1].plot(epochs, [h["val_acc"] for h in history], label="validation")
    ax[1].set_title("Précision")
    for a in ax:
        a.set_xlabel("époque")
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "courbes.png"), dpi=120)

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "matrice_confusion.png"), dpi=120)
    plt.close("all")


if __name__ == "__main__":
    main()
