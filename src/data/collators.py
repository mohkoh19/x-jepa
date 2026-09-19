"""Batch collators for the VL4M pretraining mixture.

The datamodules yield ``(image, (caption, ""), -1, index, source)`` samples.
``XJEPACollator`` stacks the images and tokenizes the captions for the X-JEPA
variants; ``CaptionCollator`` only stacks images and keeps the raw captions
because the CLIP/SigLIP baselines own their tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import default_collate


@dataclass
class XJEPABatch:
    """Tokenized image-caption batch consumed by the X-JEPA variants."""

    images: torch.Tensor
    input_ids: torch.Tensor
    attention_masks: torch.Tensor
    texts: list[str]
    indices: torch.Tensor
    sources: list[str]


@dataclass
class CaptionBatch:
    """Image-caption batch consumed by the CLIP/SigLIP baselines."""

    images: torch.Tensor
    texts: list[str]
    indices: torch.Tensor
    sources: list[str]


def _sample_metadata(batch) -> tuple[torch.Tensor, list[str]]:
    indices = torch.as_tensor([int(sample[3]) for sample in batch], dtype=torch.long)
    sources = [str(sample[4]) for sample in batch]
    return indices, sources


class XJEPACollator:
    """Tokenize captions to a fixed length and stack the paired images."""

    def __init__(self, tokenizer, max_text_length: int = 64) -> None:
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.max_text_length = int(max_text_length)

    def __call__(self, batch) -> XJEPABatch:
        images = default_collate([sample[0] for sample in batch])
        texts = [str(sample[1][0]) for sample in batch]
        indices, sources = _sample_metadata(batch)
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        return XJEPABatch(
            images=images,
            input_ids=tokens.input_ids,
            attention_masks=tokens.attention_mask,
            texts=texts,
            indices=indices,
            sources=sources,
        )


class CaptionCollator:
    """Stack images and keep raw captions for the contrastive baselines."""

    def __call__(self, batch) -> CaptionBatch:
        images = default_collate([sample[0] for sample in batch])
        texts = [str(sample[1][0]) for sample in batch]
        indices, sources = _sample_metadata(batch)
        return CaptionBatch(
            images=images,
            texts=texts,
            indices=indices,
            sources=sources,
        )
