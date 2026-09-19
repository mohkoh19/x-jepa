"""TorchVision ViT backbones with explicit CLS and patch-token outputs."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class VisionBackboneOutput:
    cls: torch.Tensor
    patch_tokens: torch.Tensor


class TorchvisionViTWithTokens(nn.Module):
    """Wrap official TorchVision ViT while exposing CLS and patch tokens.

    The factory path is intentionally weight-free. Pretrained weights are loaded
    by ``load_vision_pretrained`` so Hydra training has one source of truth for
    initialization.
    """

    def __init__(
        self,
        *,
        architecture: str = "vit_b_16",
        image_size: int = 224,
        weights: object | None = None,
    ) -> None:
        super().__init__()
        if architecture != "vit_b_16":
            raise ValueError("TorchvisionViTWithTokens currently supports only vit_b_16.")

        from torchvision.models import vit_b_16

        self.architecture = architecture
        self.model = vit_b_16(weights=weights, image_size=image_size)
        self.model.heads = nn.Identity()

        self.embed_dim = int(self.model.hidden_dim)
        self.num_features = self.embed_dim
        self.image_size = int(self.model.image_size)
        self.patch_size = int(self.model.patch_size)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.patch_embed = SimpleNamespace(
            num_patches=self.num_patches,
            patch_size=self.patch_size,
        )

    @property
    def hidden_dim(self) -> int:
        return self.embed_dim

    def _forward_full(self, images: torch.Tensor) -> VisionBackboneOutput:
        patch_tokens = self.model._process_input(images)
        batch_size = patch_tokens.shape[0]
        cls = self.model.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, patch_tokens], dim=1)
        encoded = self.model.encoder(tokens)
        return VisionBackboneOutput(cls=encoded[:, 0], patch_tokens=encoded[:, 1:])

    def forward_features(self, images: torch.Tensor) -> VisionBackboneOutput:
        return self._forward_full(images)

    def encode_global(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_features(images).cls

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_features(images).patch_tokens

    def load_torchvision_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        filtered: dict[str, torch.Tensor] = {}
        ignored: list[str] = []
        for key, value in state_dict.items():
            key = _strip_prefixes(key, ("module.", "model."))
            if key.startswith("heads."):
                ignored.append(key)
                continue
            filtered[key] = value

        target_keys = set(self.model.state_dict())
        loaded_keys = tuple(sorted(key for key in filtered if key in target_keys))
        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        missing = tuple(key for key in missing if not key.startswith("heads."))
        unexpected = tuple(
            key for key in unexpected if not key.startswith("heads.") and key not in ignored
        )
        if strict and (missing or unexpected):
            raise RuntimeError(
                "Could not strictly load torchvision ViT weights: "
                f"missing={list(missing)[:20]}; unexpected={list(unexpected)[:20]}"
            )
        return loaded_keys, tuple(missing), tuple(unexpected)


def torchvision_vit_b_16(
    *,
    img_size: int = 224,
    image_size: int | None = None,
    weights: object | None = None,
    **_: Any,
) -> TorchvisionViTWithTokens:
    """Build the weight-free TorchVision ViT-B/16 wrapper.

    Pretrained weights are loaded separately by
    :func:`src.models.components.vision_pretrained.load_vision_pretrained` so
    that the online encoder and its EMA target share one initialization path.
    Unknown keyword arguments are ignored so that archived run configs keep
    loading.
    """
    if weights is not None:
        raise ValueError(
            "torchvision_vit_b_16 is intentionally weight-free; configure "
            "`vision_pretrained` instead so weights are loaded exactly once."
        )
    return TorchvisionViTWithTokens(
        architecture="vit_b_16",
        image_size=int(image_size if image_size is not None else img_size),
        weights=None,
    )


def _strip_prefixes(key: str, prefixes: tuple[str, ...]) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key
