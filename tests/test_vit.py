import numpy as np
import pytest
import torch
from PIL import Image

from vitsat.attention import attention_rollout, overlay
from vitsat.data import CLASSES, LABELS_FR, build_transforms, stratified_split
from vitsat.engine import evaluate, train_one_epoch
from vitsat.model import build_model, count_parameters, tiny_config
from vitsat.predict import Predictor
from vitsat.train import cosine_with_warmup


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return build_model(config=tiny_config(32))


def random_image(size=64):
    return Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))


def test_toutes_les_classes_ont_un_libelle_francais():
    assert len(CLASSES) == 10 and set(LABELS_FR) == set(CLASSES)


def test_transformations():
    x = build_transforms(train=True, image_size=32)(random_image())
    assert x.shape == (3, 32, 32)
    assert -1.0 <= x.min() and x.max() <= 1.0


def test_decoupage_stratifie_sans_fuite():
    targets = [i % 10 for i in range(1000)]
    tr, va, te = stratified_split(targets)
    assert len(tr) == 800 and len(va) == 100 and len(te) == 100
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    # chaque classe est présente 10 fois dans la validation
    assert all(sum(targets[i] == c for i in va) == 10 for c in range(10))


def test_sortie_du_modele(tiny_model):
    out = tiny_model(pixel_values=torch.randn(4, 3, 32, 32), output_attentions=True)
    assert out.logits.shape == (4, 10)
    assert len(out.attentions) == 2
    assert out.attentions[0].shape == (4, 2, 17, 17)  # 16 patchs + [CLS]
    assert tiny_model.config.id2label[1] == "Forest"


def test_gel_des_couches():
    model = build_model(config=tiny_config(32), freeze_layers=1)
    total, trainable = count_parameters(model)
    assert 0 < trainable < total


def test_attention_rollout():
    attn = tuple(torch.softmax(torch.randn(2, 2, 17, 17), -1) for _ in range(3))
    mask = attention_rollout(attn)
    assert mask.shape == (2, 4, 4)
    assert torch.isclose(mask.min(), torch.tensor(0.0)) and torch.isclose(mask.max(), torch.tensor(1.0))
    assert attention_rollout(attn, discard_ratio=0.9).shape == (2, 4, 4)


def test_rollout_attention_identite_ne_plante_pas():
    attn = (torch.eye(17).expand(1, 2, 17, 17),)
    mask = attention_rollout(attn)
    assert torch.isfinite(mask).all()


def test_overlay_garde_la_taille_de_l_image():
    img = random_image(100)
    out = overlay(img, np.random.rand(4, 4))
    assert out.size == (100, 100) and out.mode == "RGB"


def test_le_modele_apprend(tiny_model):
    # sur un petit lot fixe, la perte doit baisser : la boucle d'entraînement fonctionne
    torch.manual_seed(0)
    x, y = torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))
    loader = [(x, y)]
    opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-2)
    first, _ = train_one_epoch(tiny_model, loader, opt, None, torch.device("cpu"), label_smoothing=0.0)
    for _ in range(30):
        last, acc = train_one_epoch(tiny_model, loader, opt, None, torch.device("cpu"), label_smoothing=0.0)
    assert last < first / 2
    _, eval_acc, preds, targets = evaluate(tiny_model, loader, torch.device("cpu"))
    assert len(preds) == len(targets) == 16 and eval_acc > 0.5


def test_planning_du_learning_rate():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = cosine_with_warmup(opt, warmup=10, total=100)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[9] <= 1.0          # montée
    assert lrs[50] < lrs[10]               # décroissance
    assert lrs[-1] < 0.01


def test_predicteur(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path / "m")
    preds, heat = Predictor(str(tmp_path / "m")).predict(random_image(), top_k=3)
    assert len(preds) == 3
    assert preds[0].score >= preds[1].score >= preds[2].score
    assert sum(p.score for p in preds) <= 1.0 + 1e-6
    assert preds[0].label_fr == LABELS_FR[preds[0].label]
    assert heat.size == (64, 64)
