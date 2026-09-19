"""ImageNet initialization for the TorchVision-backed ViT image encoder.

The released models initialize the ViT-B/16 image encoder with the official
``vit_b_16:IMAGENET1K_V1`` weights.  Loading is kept separate from the Hydra
factory so that both the online encoder and its EMA target copy are
initialized from exactly one source.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

SUPPORTED_SOURCES = ("torchvision",)
SUPPORTED_INIT_TARGETS = ("copy_online", "none")


@dataclass(frozen=True)
class VisionPretrainedConfig:
    source: str | None = None
    id: str | None = None
    init_target: str = "copy_online"
    strict: bool = True


@dataclass(frozen=True)
class VisionPretrainedLoadResult:
    source: str
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def load_vision_pretrained(
    encoder: nn.Module,
    target_encoder: nn.Module | None = None,
    cfg: Any | None = None,
) -> VisionPretrainedLoadResult | None:
    """Load pretrained ViT weights into the online encoder and its EMA target."""
    config = _coerce_config(cfg)
    if config is None:
        return None

    source = str(config.source).lower()
    if source not in SUPPORTED_SOURCES:
        raise ValueError(
            f"Unsupported vision_pretrained.source `{config.source}`. "
            f"Expected one of: {', '.join(SUPPORTED_SOURCES)}."
        )
    if not hasattr(encoder, "load_torchvision_state_dict"):
        raise TypeError(
            f"`{type(encoder).__name__}` cannot load TorchVision weights. "
            "Use `src.models.components.torchvision_vit.torchvision_vit_b_16`."
        )

    state_dict = _load_torchvision_state_dict(config)
    result = _load_into_wrapper(encoder, state_dict, config)
    _init_target_encoder(encoder, target_encoder, config)
    log.info(
        "Loaded %s pretrained vision weights into %s (%d keys).",
        source,
        type(encoder).__name__,
        len(result.loaded_keys),
    )
    return result


def _load_into_wrapper(
    encoder: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    config: VisionPretrainedConfig,
) -> VisionPretrainedLoadResult:
    loaded_keys, missing_keys, unexpected_keys = encoder.load_torchvision_state_dict(
        state_dict,
        strict=config.strict,
    )
    return VisionPretrainedLoadResult(
        source=config.source or "torchvision",
        loaded_keys=tuple(f"model.{key}" for key in loaded_keys),
        missing_keys=tuple(f"model.{key}" for key in missing_keys),
        unexpected_keys=unexpected_keys,
    )


def _init_target_encoder(
    encoder: nn.Module,
    target_encoder: nn.Module | None,
    config: VisionPretrainedConfig,
) -> None:
    if target_encoder is None:
        return
    if config.init_target == "none":
        return
    if config.init_target != "copy_online":
        raise ValueError(
            f"Unsupported vision_pretrained.init_target `{config.init_target}`. "
            f"Expected one of: {', '.join(SUPPORTED_INIT_TARGETS)}."
        )
    target_encoder.load_state_dict(encoder.state_dict(), strict=True)
    target_encoder.eval()


def _load_torchvision_state_dict(config: VisionPretrainedConfig) -> Mapping[str, torch.Tensor]:
    if config.id is None:
        raise ValueError("`vision_pretrained.id` is required, e.g. `vit_b_16:IMAGENET1K_V1`.")
    try:
        import torchvision.models as models
    except ImportError as exc:  # pragma: no cover - torchvision is a hard dependency
        raise ImportError(
            "Loading `torchvision` vision weights requires the `torchvision` package."
        ) from exc

    model_name, weights_name = _parse_torchvision_id(config.id)
    model_fn = getattr(models, model_name)
    weights_cls = getattr(models, _torchvision_weights_enum_name(model_name))
    weights = getattr(weights_cls, weights_name)
    return model_fn(weights=weights).state_dict()


def _coerce_config(cfg: Any | None) -> VisionPretrainedConfig | None:
    if cfg is None or cfg is False:
        return None
    if isinstance(cfg, VisionPretrainedConfig):
        return cfg

    if isinstance(cfg, Mapping):
        data = dict(cfg)
    else:
        fields = VisionPretrainedConfig.__dataclass_fields__
        data = {key: getattr(cfg, key) for key in fields if hasattr(cfg, key)}

    if data.get("source") in (None, "null"):
        return None
    valid_keys = set(VisionPretrainedConfig.__dataclass_fields__)
    unknown = sorted(set(data) - valid_keys)
    if unknown:
        log.debug("Ignoring unsupported vision_pretrained keys: %s", unknown)
    return VisionPretrainedConfig(
        **{key: value for key, value in data.items() if key in valid_keys}
    )


def _parse_torchvision_id(identifier: str) -> tuple[str, str]:
    if ":" not in identifier:
        raise ValueError(
            "`torchvision` vision_pretrained.id must look like `vit_b_16:IMAGENET1K_V1`."
        )
    model_name, weights_name = identifier.split(":", 1)
    return model_name, weights_name


def _torchvision_weights_enum_name(model_name: str) -> str:
    parts = model_name.split("_")
    if len(parts) < 3 or parts[0] != "vit":
        raise ValueError(f"Unsupported TorchVision ViT model id `{model_name}`.")
    return f"ViT_{parts[1].upper()}_{parts[2]}_Weights"
