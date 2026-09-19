from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import torch
from lightning import LightningModule

from src.evaluation.adapters import PretrainedModelAdapter
from src.evaluation.scoring import (
    SUPPORTED_ZERO_SHOT_SCORES,
    normalize_for_zero_shot_score,
    pair_scores_from_features,
)

SVO_PROBES_KEY_METRICS = ("acc", "margin_mean", "n_examples")


def compute_svo_probes_metrics(records: Iterable[Mapping]) -> dict[str, float]:
    all_correct: list[float] = []
    all_margins: list[float] = []
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"correct": [], "margin": []})

    for record in records:
        negative_type = str(record["negative_type"])
        correct = float(record["correct"])
        margin = float(record["margin"])
        all_correct.append(correct)
        all_margins.append(margin)
        grouped[negative_type]["correct"].append(correct)
        grouped[negative_type]["margin"].append(margin)

    def mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    metrics = {
        "acc": mean(all_correct),
        "margin_mean": mean(all_margins),
        "n_examples": float(len(all_correct)),
    }
    for negative_type, values in sorted(grouped.items()):
        metrics[f"{negative_type}_acc"] = mean(values["correct"])
        metrics[f"{negative_type}_margin_mean"] = mean(values["margin"])
        metrics[f"{negative_type}_n_examples"] = float(len(values["correct"]))
    return metrics


def _metadata_value(batch: Mapping[str, Any], key: str, idx: int, default: Any = None) -> Any:
    if key not in batch:
        return default
    value = batch[key]
    try:
        item = value[idx]
    except Exception:
        item = value
    if hasattr(item, "item"):
        try:
            return item.item()
        except Exception:
            pass
    return item


def _negative_type_group(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text if text in {"subject", "verb", "object"} else "multi"


def _score_svo_probes_batch(
    adapter: PretrainedModelAdapter,
    pos_images: torch.Tensor,
    neg_images: torch.Tensor,
    sentences: list[str],
    *,
    score: str = "itc",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = torch.cat([pos_images, neg_images], dim=0)
    normalize = normalize_for_zero_shot_score(score)
    image_features = adapter.encode_images(images, normalize=normalize)
    text_features = adapter.encode_texts(sentences, normalize=normalize)
    batch_size = pos_images.shape[0]
    pos_features = image_features[:batch_size]
    neg_features = image_features[batch_size:]

    pos_scores = pair_scores_from_features(pos_features, text_features, score=score)
    neg_scores = pair_scores_from_features(neg_features, text_features, score=score)
    margins = pos_scores - neg_scores
    return pos_scores, neg_scores, margins


def evaluate_svo_probes_task(
    adapter: PretrainedModelAdapter,
    datamodule,
    *,
    score: str = "itc",
    setup_stage: str = "validate",
    include_type_metrics: bool = False,
) -> dict[str, float]:
    """Run SVO-Probes zero-shot positive-vs-negative image scoring."""
    if score not in SUPPORTED_ZERO_SHOT_SCORES:
        raise ValueError("SVO-Probes score must be `itc` or `cosine`.")

    datamodule.setup(stage=setup_stage)
    records = []
    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            pos_images = batch["pos_image"]
            neg_images = batch["neg_image"]
            sentences = list(batch["sentence"])
            negative_types = list(batch["negative_type"])

            _, _, margins = _score_svo_probes_batch(
                adapter,
                pos_images,
                neg_images,
                sentences,
                score=score,
            )
            correct = margins > 0
            for idx, negative_type in enumerate(negative_types):
                records.append(
                    {
                        "negative_type": negative_type,
                        "correct": bool(correct[idx].detach().cpu()),
                        "margin": float(margins[idx].detach().cpu()),
                    }
                )

    metrics = compute_svo_probes_metrics(records)
    if include_type_metrics:
        return metrics
    return {key: metrics[key] for key in SVO_PROBES_KEY_METRICS if key in metrics}


def score_svo_probes_examples(
    adapter: PretrainedModelAdapter,
    datamodule,
    *,
    score: str = "itc",
    setup_stage: str = "validate",
    split: str | None = None,
) -> list[dict[str, Any]]:
    """Return per-example SVO-Probes scores with native metadata preserved."""
    if score not in SUPPORTED_ZERO_SHOT_SCORES:
        raise ValueError("SVO-Probes score must be `itc` or `cosine`.")

    datamodule.setup(stage=setup_stage)
    split_name = split or str(getattr(getattr(datamodule, "hparams", None), "split", "train"))
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            pos_scores, neg_scores, margins = _score_svo_probes_batch(
                adapter,
                batch["pos_image"],
                batch["neg_image"],
                list(batch["sentence"]),
                score=score,
            )
            correct = margins > 0
            for idx, sentence in enumerate(list(batch["sentence"])):
                native = str(
                    _metadata_value(
                        batch,
                        "negative_type_native",
                        idx,
                        _metadata_value(batch, "negative_type", idx, "unknown"),
                    )
                )
                rows.append(
                    {
                        "dataset": "svo",
                        "split": split_name,
                        "example_id": str(
                            _metadata_value(batch, "example_id", idx, f"{split_name}:{len(rows)}")
                        ),
                        "category": _negative_type_group(native),
                        "sub_category": native,
                        "negative_type_native": native,
                        "negative_type": native,
                        "negative_type_group": _negative_type_group(native),
                        "sentence": str(sentence),
                        "caption": str(sentence),
                        "image_id": str(_metadata_value(batch, "pos_image_id", idx, "")),
                        "positive_image_id": _metadata_value(batch, "pos_image_id", idx, ""),
                        "negative_image_id": _metadata_value(batch, "neg_image_id", idx, ""),
                        "positive_text_id": "",
                        "negative_text_id": "",
                        "positive_triplet": str(_metadata_value(batch, "pos_triplet", idx, "")),
                        "negative_triplet": str(_metadata_value(batch, "neg_triplet", idx, "")),
                        "positive_score": float(pos_scores[idx].detach().cpu()),
                        "negative_score": float(neg_scores[idx].detach().cpu()),
                        "margin": float(margins[idx].detach().cpu()),
                        "correct": bool(correct[idx].detach().cpu()),
                    }
                )
    return rows


class SVOProbesEvaluator(LightningModule):
    """Zero-shot SVO-Probes sentence-to-image pair evaluator."""

    def __init__(
        self,
        ckpt_path: str,
        bert_size: str = "auto",
        max_text_len: int = 64,
        score: str = "itc",
        include_type_metrics: bool = True,
        adapter: PretrainedModelAdapter | None = None,
    ) -> None:
        super().__init__()
        if score not in SUPPORTED_ZERO_SHOT_SCORES:
            raise ValueError("SVO-Probes score must be `itc` or `cosine`.")
        self.save_hyperparameters(ignore=["adapter"], logger=False)
        self.adapter = adapter or PretrainedModelAdapter(
            ckpt_path=ckpt_path,
            bert_size=bert_size,
            max_text_len=max_text_len,
        )
        self.records = []

    def on_validation_epoch_start(self) -> None:
        self.records.clear()

    def validation_step(self, batch, batch_idx):
        del batch_idx
        pos_images = batch["pos_image"]
        neg_images = batch["neg_image"]
        sentences = list(batch["sentence"])
        negative_types = list(batch["negative_type"])

        _, _, margins = _score_svo_probes_batch(
            self.adapter,
            pos_images,
            neg_images,
            sentences,
            score=self.hparams.score,
        )
        correct = margins > 0
        for idx, negative_type in enumerate(negative_types):
            self.records.append(
                {
                    "negative_type": negative_type,
                    "correct": bool(correct[idx].detach().cpu()),
                    "margin": float(margins[idx].detach().cpu()),
                }
            )

    def on_validation_epoch_end(self) -> None:
        metrics = compute_svo_probes_metrics(self.records)
        if not self.hparams.include_type_metrics:
            metrics = {key: metrics[key] for key in SVO_PROBES_KEY_METRICS if key in metrics}
        self.log_dict(metrics, sync_dist=False, logger=False)
