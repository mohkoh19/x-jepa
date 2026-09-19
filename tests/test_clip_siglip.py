from __future__ import annotations

import inspect
import math
from functools import partial
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

import torch.nn as nn
import yaml
from torchvision import transforms

from src.data.VL4M_datamodule import VL4MWebDatasetDatamodule
from src.models.contrastive_baselines import (
    CLIP,
    SigLIP,
    build_global_targets,
    build_siglip_labels,
)
from src.utils.ijepa.transforms import make_train_transform


class TokenBatch(SimpleNamespace):
    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class PairTokenizer:
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        del padding, truncation, return_tensors
        ids = []
        for text in texts:
            value = int(str(text).split()[-1]) + 1
            ids.append([value] + [0] * (max_length - 1))
        input_ids = torch.tensor(ids, dtype=torch.long)
        return TokenBatch(input_ids=input_ids, attention_mask=(input_ids != 0).long())


class PairVision(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.embed_dim = dim
        self.bias = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, images):
        indices = images[:, 0, 0, 0].long()
        features = torch.zeros(images.shape[0], self.embed_dim, device=images.device)
        features[torch.arange(images.shape[0], device=images.device), indices] = 1.0
        return self.norm(features + self.bias)


class SequenceVision(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.embed_dim = dim
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, images):
        batch_size = images.shape[0]
        cls = torch.ones(batch_size, 1, self.embed_dim, device=images.device)
        patches = torch.full((batch_size, 3, self.embed_dim), 100.0, device=images.device)
        return torch.cat([cls + self.bias, patches], dim=1)


class PairText(nn.Module):
    def __init__(self, dim=8, vocab_size=32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim)
        self.embeddings = nn.Embedding(vocab_size, dim)
        self.LayerNorm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.pooler = nn.Linear(dim, dim)
        with torch.no_grad():
            self.embeddings.weight.zero_()
            for idx in range(1, min(vocab_size, dim + 1)):
                self.embeddings.weight[idx, idx - 1] = 1.0

    def forward(self, input_ids, attention_mask):
        del attention_mask
        hidden = self.LayerNorm(self.proj(self.embeddings(input_ids)))
        return SimpleNamespace(last_hidden_state=hidden)


class FlatVision768(nn.Module):
    hidden_dim = 768

    def forward(self, images):
        return torch.ones(images.shape[0], 768, device=images.device)


def _model_kwargs(weight_decay=0.2, model_cls=CLIP):
    peak = 1e-2
    return {
        "optimizer": partial(
            torch.optim.AdamW,
            lr=peak,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=weight_decay,
        ),
        "shared_dim": 8,
        "text_encoder_name": "bert-base-uncased",
        "max_text_len": 4,
        "vision_model": PairVision(),
        "text_encoder": PairText(),
        "tokenizer": PairTokenizer(),
        "logit_scale_init": math.log(10.0) if model_cls is SigLIP else math.log(1 / 0.07),
        "lr_schedule": {"type": "warmup_cosine", "final_lr": 0.0},
        "ipe_scale": 1.0,
        "warmup": 4,
        "lr_groups": {
            "vision_encoder": {"peak_lr": peak},
            "text_encoder": {"peak_lr": peak},
            "projection_heads": {"peak_lr": peak},
            "temperature": {"peak_lr": peak},
            "temperature_bias": {"peak_lr": peak},
        },
        "log_grad_every": 0,
        "log_embedding_stats_every": 0,
    }


def _paired_batch(batch_size=4):
    images = torch.arange(batch_size, dtype=torch.float32).view(batch_size, 1, 1, 1)
    texts = [f"caption {idx}" for idx in range(batch_size)]
    return images, texts


def _assert_loss_decreases(model):
    images, texts = _paired_batch()
    model.train()
    with torch.no_grad():
        initial = model(images, texts)[0].item()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
    final = initial
    for _ in range(10):
        optimizer.zero_grad()
        loss = model(images, texts)[0]
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            final = model(images, texts)[0].item()
        if final < initial:
            break
    assert final < initial


def _param_weight_decay(optimizer, param):
    for group in optimizer.param_groups:
        if any(candidate is param for candidate in group["params"]):
            return group["weight_decay"]
    raise AssertionError("parameter not found in optimizer")


def test_clip_image_encoder_output_shape_is_768():
    model = CLIP(
        optimizer=partial(torch.optim.AdamW, lr=1e-3, weight_decay=0.2),
        vision_model=FlatVision768(),
        text_encoder=PairText(dim=768, vocab_size=1024),
        tokenizer=PairTokenizer(),
        shared_dim=768,
    )
    pooled = model.encode_image_pooled(torch.zeros(2, 3, 224, 224))
    assert tuple(pooled.shape) == (2, 768)


def test_clip_uses_cls_token_not_mean_pooling_for_sequence_vision_outputs():
    model = CLIP(
        **{
            **_model_kwargs(),
            "vision_model": SequenceVision(),
        }
    )
    pooled = model.encode_image_pooled(torch.zeros(2, 3, 4, 4))
    assert torch.allclose(pooled, torch.ones_like(pooled))
    assert not torch.allclose(pooled, torch.full_like(pooled, 75.25))


def test_clip_logit_scale_initializes_to_clip_temperature():
    model = CLIP(**_model_kwargs())
    assert model.logit_scale.item() == pytest.approx(math.log(1 / 0.07))


def test_clip_forward_loss_returns_finite_scalar():
    model = CLIP(**_model_kwargs())
    loss, image_features, text_features = model(*_paired_batch())
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.allclose(image_features.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(text_features.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_clip_tiny_paired_batch_loss_decreases():
    _assert_loss_decreases(CLIP(**_model_kwargs()))


def test_siglip_logit_scale_initializes_to_log_ten():
    model = SigLIP(**_model_kwargs(weight_decay=1e-4, model_cls=SigLIP))
    assert model.logit_scale.item() == pytest.approx(math.log(10.0))


def test_siglip_logit_bias_initializes_to_minus_ten():
    model = SigLIP(**_model_kwargs(weight_decay=1e-4, model_cls=SigLIP))
    assert model.logit_bias.item() == pytest.approx(-10.0)


def test_siglip_forward_loss_returns_finite_scalar():
    model = SigLIP(**_model_kwargs(weight_decay=1e-4, model_cls=SigLIP))
    loss, image_features, text_features = model(*_paired_batch())
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.allclose(image_features.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(text_features.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_siglip_tiny_paired_batch_loss_decreases():
    _assert_loss_decreases(SigLIP(**_model_kwargs(weight_decay=1e-4, model_cls=SigLIP)))


def test_siglip_positive_labels_are_only_at_local_global_targets():
    labels = build_siglip_labels(
        3,
        12,
        rank=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert labels.eq(1).nonzero(as_tuple=False).tolist() == [[0, 6], [1, 7], [2, 8]]
    assert labels.eq(1).sum().item() == 3
    assert labels.eq(-1).sum().item() == 33


def test_optimizer_bias_parameters_have_zero_weight_decay():
    model = CLIP(**_model_kwargs(weight_decay=0.2))
    optimizer = model.hparams.optimizer(params=model._get_param_groups())
    assert _param_weight_decay(optimizer, model.text_encoder.proj.bias) == 0.0


def test_optimizer_layernorm_parameters_have_zero_weight_decay():
    model = CLIP(**_model_kwargs(weight_decay=0.2))
    optimizer = model.hparams.optimizer(params=model._get_param_groups())
    assert _param_weight_decay(optimizer, model.text_encoder.LayerNorm.weight) == 0.0
    assert _param_weight_decay(optimizer, model.vision_encoder.norm.weight) == 0.0


def test_optimizer_logit_scale_and_logit_bias_have_zero_weight_decay():
    siglip = SigLIP(**_model_kwargs(weight_decay=1e-4, model_cls=SigLIP))
    optimizer = siglip.hparams.optimizer(params=siglip._get_param_groups())
    assert _param_weight_decay(optimizer, siglip.logit_scale) == 0.0
    assert _param_weight_decay(optimizer, siglip.logit_bias) == 0.0


def test_optimizer_projection_weights_have_configured_weight_decay():
    model = CLIP(**_model_kwargs(weight_decay=0.2))
    optimizer = model.hparams.optimizer(params=model._get_param_groups())
    assert _param_weight_decay(optimizer, model.visual_projection.weight) == pytest.approx(0.2)
    assert _param_weight_decay(optimizer, model.text_projection.weight) == pytest.approx(0.2)


def test_clip_optimizer_uses_configured_decay_group_weight_decay():
    model = CLIP(**_model_kwargs(weight_decay=0.2))
    optimizer = model.hparams.optimizer(params=model._get_param_groups())
    decay_values = {group["weight_decay"] for group in optimizer.param_groups}
    assert 0.2 in decay_values


def test_siglip_optimizer_uses_configured_decay_group_weight_decay():
    model = SigLIP(**_model_kwargs(weight_decay=1e-4, model_cls=SigLIP))
    optimizer = model.hparams.optimizer(params=model._get_param_groups())
    decay_values = {group["weight_decay"] for group in optimizer.param_groups}
    assert 1e-4 in decay_values


@pytest.mark.parametrize(
    ("model_cls", "weight_decay"),
    [(CLIP, 0.2), (SigLIP, 1e-4)],
)
def test_clip_siglip_scheduler_warmup_uses_optimizer_steps_with_accumulation(
    model_cls,
    weight_decay,
):
    model = model_cls(
        **{
            **_model_kwargs(weight_decay=weight_decay, model_cls=model_cls),
            "warmup": 4,
            "ipe_scale": 1.0,
            "lr_schedule": {"type": "warmup_cosine", "final_lr": 0.0},
        }
    )
    model._trainer = SimpleNamespace(
        num_training_batches=100,
        accumulate_grad_batches=4,
        max_epochs=40,
    )

    optim_config = model.configure_optimizers()
    scheduler = optim_config["lr_scheduler"]["scheduler"]

    assert model.hparams["opt/accumulate_grad_batches"] == 4
    assert model.hparams["opt/steps_per_epoch"] == 25
    assert model.hparams["opt/total_steps"] == 1000
    assert model.hparams["opt/warmup_steps"] == 100
    assert scheduler.lr_lambdas[0](99) == pytest.approx(99 / 100)
    assert scheduler.lr_lambdas[0](100) == pytest.approx(1.0)


@pytest.mark.parametrize("model_cls", [CLIP, SigLIP])
def test_clip_siglip_configure_optimizers_keeps_fixed_weight_decay(model_cls):
    model = model_cls(**_model_kwargs(weight_decay=0.01, model_cls=model_cls))
    model._trainer = SimpleNamespace(
        num_training_batches=100,
        accumulate_grad_batches=4,
        max_epochs=40,
    )

    optim_config = model.configure_optimizers()
    optimizer = optim_config["optimizer"]
    no_decay_groups = [group for group in optimizer.param_groups if group.get("WD_exclude")]
    decay_groups = [group for group in optimizer.param_groups if not group.get("WD_exclude")]

    assert "optimizer" in optim_config
    assert "lr_scheduler" in optim_config
    assert no_decay_groups
    assert decay_groups
    assert all(group["weight_decay"] == pytest.approx(0.0) for group in no_decay_groups)
    assert all(group["weight_decay"] == pytest.approx(0.01) for group in decay_groups)


def test_clip_siglip_configs_use_final_fixed_weight_decay_recipe():
    for config_name in ("clip", "siglip"):
        with open(f"configs/model/{config_name}.yaml", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        assert float(cfg["optimizer"]["weight_decay"]) == pytest.approx(0.01)
        assert cfg["optimizer"]["betas"] == pytest.approx([0.9, 0.98])
        assert float(cfg["optimizer"]["eps"]) == pytest.approx(1e-6)
        assert float(cfg["lr_schedule"]["final_lr"]) == pytest.approx(1e-6)
        assert "wd_scheduler" not in cfg


def test_clip_global_targets_include_rank_offset():
    targets = build_global_targets(4, rank=3, device=torch.device("cpu"))
    assert targets.tolist() == [12, 13, 14, 15]


def test_siglip_global_targets_include_rank_offset():
    labels = build_siglip_labels(
        2,
        8,
        rank=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert labels.eq(1).nonzero(as_tuple=False).tolist() == [[0, 6], [1, 7]]


def test_clip_siglip_train_transform_matches_torchvision_imagenet_vit_preprocessing():
    transform = make_train_transform(crop_size=224)
    random_crop = next(
        t for t in transform.transforms if isinstance(t, transforms.RandomResizedCrop)
    )
    normalize = next(t for t in transform.transforms if isinstance(t, transforms.Normalize))

    assert random_crop.size == (224, 224)
    assert tuple(normalize.mean) == pytest.approx((0.485, 0.456, 0.406))
    assert tuple(normalize.std) == pytest.approx((0.229, 0.224, 0.225))


def test_vl4m_webdataset_train_loader_drops_last_for_equal_ddp_batches():
    source = inspect.getsource(VL4MWebDatasetDatamodule.train_dataloader)
    assert '"drop_last": True' in source
    assert (
        "samples_per_rank = (int(self.train_size) // world_size // batch_size) * batch_size"
        in source
    )
