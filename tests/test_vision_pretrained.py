from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
torchvision_models = pytest.importorskip("torchvision.models")

import torch.nn as nn

from src.models.components import vision_pretrained
from src.models.components.torchvision_vit import torchvision_vit_b_16
from src.models.components.vision_pretrained import (
    load_vision_pretrained,
    _coerce_config,
)


class TinyEncoder(nn.Module):
    """Minimal stand-in for the TorchVision ViT wrapper."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.loaded: dict | None = None

    def load_torchvision_state_dict(self, state_dict, *, strict: bool = True):
        missing, unexpected = self.load_state_dict(dict(state_dict), strict=False)
        if strict and (missing or unexpected):
            raise RuntimeError(f"missing={list(missing)}; unexpected={list(unexpected)}")
        self.loaded = dict(state_dict)
        return tuple(sorted(state_dict)), tuple(missing), tuple(unexpected)


def test_torchvision_weights_load_into_wrapper_and_target(monkeypatch):
    source_state = torchvision_models.vit_b_16(weights=None).state_dict()
    monkeypatch.setattr(
        vision_pretrained,
        "_load_torchvision_state_dict",
        lambda config: source_state,
    )

    encoder = torchvision_vit_b_16(img_size=224)
    target_encoder = torchvision_vit_b_16(img_size=224)
    result = load_vision_pretrained(
        encoder,
        target_encoder=target_encoder,
        cfg={"source": "torchvision", "id": "vit_b_16:IMAGENET1K_V1"},
    )

    assert result is not None
    assert not result.missing_keys
    assert not result.unexpected_keys
    assert len(result.loaded_keys) == len(encoder.model.state_dict())
    torch.testing.assert_close(
        encoder.model.conv_proj.weight, source_state["conv_proj.weight"]
    )
    torch.testing.assert_close(
        target_encoder.model.conv_proj.weight, encoder.model.conv_proj.weight
    )


def test_init_target_none_leaves_target_untouched(monkeypatch):
    source_state = torchvision_models.vit_b_16(weights=None).state_dict()
    monkeypatch.setattr(
        vision_pretrained,
        "_load_torchvision_state_dict",
        lambda config: source_state,
    )

    encoder = torchvision_vit_b_16(img_size=224)
    target_encoder = torchvision_vit_b_16(img_size=224)
    target_before = target_encoder.model.conv_proj.weight.detach().clone()

    load_vision_pretrained(
        encoder,
        target_encoder=target_encoder,
        cfg={
            "source": "torchvision",
            "id": "vit_b_16:IMAGENET1K_V1",
            "init_target": "none",
        },
    )

    torch.testing.assert_close(target_encoder.model.conv_proj.weight, target_before)
    assert not torch.allclose(encoder.model.conv_proj.weight, target_before)


def test_strict_load_reports_missing_keys(monkeypatch):
    source_state = torchvision_models.vit_b_16(weights=None).state_dict()
    truncated = {key: value for key, value in source_state.items() if "conv_proj" not in key}
    monkeypatch.setattr(
        vision_pretrained,
        "_load_torchvision_state_dict",
        lambda config: truncated,
    )

    with pytest.raises(RuntimeError, match="missing="):
        load_vision_pretrained(
            torchvision_vit_b_16(img_size=224),
            cfg={"source": "torchvision", "id": "vit_b_16:IMAGENET1K_V1"},
        )


def test_unsupported_source_raises():
    with pytest.raises(ValueError, match="Unsupported vision_pretrained.source"):
        load_vision_pretrained(TinyEncoder(), cfg={"source": "mystery"})


def test_unsupported_init_target_raises(monkeypatch):
    monkeypatch.setattr(
        vision_pretrained,
        "_load_torchvision_state_dict",
        lambda config: torchvision_models.vit_b_16(weights=None).state_dict(),
    )
    with pytest.raises(ValueError, match="init_target"):
        load_vision_pretrained(
            torchvision_vit_b_16(img_size=224),
            target_encoder=torchvision_vit_b_16(img_size=224),
            cfg={
                "source": "torchvision",
                "id": "vit_b_16:IMAGENET1K_V1",
                "init_target": "load_target_key",
            },
        )


def test_config_coercion_ignores_unknown_keys_and_null_source():
    assert _coerce_config(None) is None
    assert _coerce_config({"source": "null"}) is None

    config = _coerce_config(
        SimpleNamespace(
            source="torchvision",
            id="vit_b_16:IMAGENET1K_V1",
            resize_pos_embed=True,
            load_pos_embed=True,
        )
    )
    assert config is not None
    assert config.source == "torchvision"
    assert config.id == "vit_b_16:IMAGENET1K_V1"
    assert config.init_target == "copy_online"
