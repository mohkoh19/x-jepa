from __future__ import annotations

import re


class CaptionPreprocessingMixin:
    """Shared caption normalization used across caption datasets."""

    max_words: int

    def pre_caption(self, caption: str) -> str:
        caption = re.sub(r"([.!\"()*#:;~])", " ", caption.lower())
        caption = re.sub(r"\s{2,}", " ", caption)
        caption = caption.rstrip("\n").strip(" ")

        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption
