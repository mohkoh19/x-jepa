from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.data.collators import CaptionCollator, XJEPABatch, XJEPACollator


class FakeTokenizer:
    """Minimal tokenizer stub that mimics the Hugging Face call signature."""

    def __init__(self) -> None:
        self.added_special_tokens = []
        self.calls = []

    def add_special_tokens(self, mapping):
        self.added_special_tokens.append(mapping)

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        self.calls.append(
            {
                "texts": list(texts),
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
                "return_tensors": return_tensors,
            }
        )
        batch = len(texts)
        input_ids = torch.arange(batch * max_length).reshape(batch, max_length)
        attention = torch.ones(batch, max_length, dtype=torch.long)
        return SimpleNamespace(input_ids=input_ids, attention_mask=attention)


def _sample(idx: int, caption: str, source: str = "CC3M"):
    image = torch.full((3, 4, 4), float(idx))
    return image, (caption, ""), -1, idx, source


def test_xjepa_collator_tokenizes_captions_and_stacks_images():
    tokenizer = FakeTokenizer()
    collator = XJEPACollator(tokenizer, max_text_length=8)

    batch = collator([_sample(0, "a cat"), _sample(1, "a dog", source="COCO")])

    assert isinstance(batch, XJEPABatch)
    assert tuple(batch.images.shape) == (2, 3, 4, 4)
    assert tuple(batch.input_ids.shape) == (2, 8)
    assert tuple(batch.attention_masks.shape) == (2, 8)
    assert batch.texts == ["a cat", "a dog"]
    assert batch.indices.tolist() == [0, 1]
    assert batch.sources == ["CC3M", "COCO"]
    assert tokenizer.calls == [
        {
            "texts": ["a cat", "a dog"],
            "padding": "max_length",
            "truncation": True,
            "max_length": 8,
            "return_tensors": "pt",
        }
    ]
    assert tokenizer.added_special_tokens == [{"bos_token": "[DEC]"}]


def test_caption_collator_keeps_raw_captions_for_the_baselines():
    batch = CaptionCollator()([_sample(3, "raw caption", source="VG")])

    assert tuple(batch.images.shape) == (1, 3, 4, 4)
    assert batch.texts == ["raw caption"]
    assert batch.indices.tolist() == [3]
    assert batch.sources == ["VG"]
    assert not hasattr(batch, "input_ids")


def test_xjepa_collator_reads_text_from_the_caption_tuple():
    tokenizer = FakeTokenizer()
    collator = XJEPACollator(tokenizer, max_text_length=4)

    batch = collator([(torch.zeros(3, 2, 2), ("first caption", "ignored"), -1, 0, "SBU")])

    assert batch.texts == ["first caption"]
    assert tokenizer.calls[0]["texts"] == ["first caption"]
