import pytest

torch = pytest.importorskip("torch")

from src.evaluation.scoring import (
    normalize_for_zero_shot_score,
    pair_scores_from_features,
    similarity_matrix_from_features,
)


def test_itc_and_cosine_scores_are_normalized():
    assert normalize_for_zero_shot_score("itc") is True
    assert normalize_for_zero_shot_score("cosine") is True


def test_pair_scores_support_itc_aliases():
    image_features = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    text_features = torch.tensor([[1.0, 0.0], [0.0, 3.0]])

    scores = pair_scores_from_features(image_features, text_features, score="cosine")

    assert scores.tolist() == pytest.approx([1.0, 1.0])


def test_similarity_matrix_supports_itc_aliases():
    image_features = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    text_features = torch.tensor([[1.0, 0.0], [4.0, 0.0]])

    sims = similarity_matrix_from_features(
        image_features,
        text_features,
        score="cosine",
    )

    assert torch.allclose(sims, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
