"""Jeu de données EuroSAT : classes, transformations et découpage train / validation / test."""
from __future__ import annotations

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Ordre des classes de torchvision.datasets.EuroSAT (ordre alphabétique des dossiers)
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# Libellés lisibles pour l'interface
LABELS_FR = {
    "AnnualCrop": "Cultures annuelles",
    "Forest": "Forêt",
    "HerbaceousVegetation": "Végétation herbacée",
    "Highway": "Autoroute",
    "Industrial": "Zone industrielle",
    "Pasture": "Pâturage",
    "PermanentCrop": "Cultures permanentes",
    "Residential": "Zone résidentielle",
    "River": "Rivière",
    "SeaLake": "Mer / lac",
}

# Le ViT pré-entraîné (google/vit-base-patch16-224-in21k) attend des pixels normalisés à [-1, 1]
MEAN = STD = (0.5, 0.5, 0.5)


class RandomRot90:
    """Rotation aléatoire de 0, 90, 180 ou 270° : une image satellite n'a pas de « haut »."""

    def __call__(self, img):
        import random

        return img.rotate(90 * random.randint(0, 3))


def build_transforms(train: bool, image_size: int = 224):
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        # Augmentations adaptées aux vues aériennes
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRot90(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


def stratified_split(targets: list[int], val: float = 0.1, test: float = 0.1, seed: int = 42):
    """Découpe stratifiée : chaque classe garde la même proportion dans les trois ensembles."""
    idx = list(range(len(targets)))
    train_idx, rest = train_test_split(idx, test_size=val + test, stratify=targets, random_state=seed)
    rest_targets = [targets[i] for i in rest]
    val_idx, test_idx = train_test_split(rest, test_size=test / (val + test), stratify=rest_targets, random_state=seed)
    return train_idx, val_idx, test_idx


def make_loaders(root: str, image_size: int = 224, batch_size: int = 32, num_workers: int = 2,
                 subset: float = 1.0, seed: int = 42):
    """Télécharge EuroSAT (≈ 90 Mo, 27 000 images 64×64) et crée les trois DataLoaders."""
    train_ds = datasets.EuroSAT(root, transform=build_transforms(True, image_size), download=True)
    eval_ds = datasets.EuroSAT(root, transform=build_transforms(False, image_size), download=False)
    targets = list(train_ds.targets)

    if subset < 1.0:
        # Sous-échantillon stratifié pour des essais rapides
        keep, _ = train_test_split(list(range(len(targets))), train_size=subset, stratify=targets, random_state=seed)
    else:
        keep = list(range(len(targets)))
    tr, va, te = stratified_split([targets[i] for i in keep], seed=seed)
    tr, va, te = ([keep[i] for i in part] for part in (tr, va, te))

    def loader(ds, ids, shuffle):
        return DataLoader(Subset(ds, ids), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return loader(train_ds, tr, True), loader(eval_ds, va, False), loader(eval_ds, te, False)
