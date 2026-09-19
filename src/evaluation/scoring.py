from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

ZERO_SHOT_SCORE_ALIASES = {
    "itc": "itc",
    "cosine": "itc",
}
SUPPORTED_ZERO_SHOT_SCORES = tuple(ZERO_SHOT_SCORE_ALIASES)


def canonical_zero_shot_score(score: str) -> str:
    key = str(score or "itc").strip().lower()
    if key not in ZERO_SHOT_SCORE_ALIASES:
        valid = ", ".join(sorted(ZERO_SHOT_SCORE_ALIASES))
        raise ValueError(f"Unsupported zero-shot score `{score}`. Valid scores: {valid}.")
    return ZERO_SHOT_SCORE_ALIASES[key]


def normalize_for_zero_shot_score(score: str) -> bool:
    return canonical_zero_shot_score(score) == "itc"


def pair_scores_from_features(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    *,
    score: str = "itc",
) -> torch.Tensor:
    score = canonical_zero_shot_score(score)
    if image_features.shape != text_features.shape:
        raise ValueError(
            "Pair scoring expects image and text features with the same shape; "
            f"got {tuple(image_features.shape)} and {tuple(text_features.shape)}."
        )
    if score == "itc":
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        return (image_features * text_features).sum(dim=-1)
    raise AssertionError(f"Unhandled canonical score: {score}")


def similarity_matrix_from_features(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    *,
    score: str = "itc",
) -> torch.Tensor:
    score = canonical_zero_shot_score(score)
    if score == "itc":
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        return image_features @ text_features.t()
    raise AssertionError(f"Unhandled canonical score: {score}")


def encode_pair_scores(
    adapter,
    images: torch.Tensor,
    texts: Iterable[str],
    *,
    score: str = "itc",
) -> torch.Tensor:
    normalize = normalize_for_zero_shot_score(score)
    image_features = adapter.encode_images(images, normalize=normalize)
    text_features = adapter.encode_texts(list(texts), normalize=normalize)
    return pair_scores_from_features(image_features, text_features, score=score)
