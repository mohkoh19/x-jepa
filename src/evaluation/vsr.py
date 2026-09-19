from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, Mapping

import torch
from lightning import LightningModule

from src.evaluation.adapters import PretrainedModelAdapter
from src.evaluation.scoring import SUPPORTED_ZERO_SHOT_SCORES, encode_pair_scores

try:
    import torch.distributed as dist  # type: ignore[attr-defined]
except Exception:
    dist = None


VSR_KEY_METRICS = (
    "acc",
    "auroc",
    "average_precision",
    "label_margin_mean",
    "score_gap_mean",
    "n_examples",
)


def _metric_safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower())
    return value.strip("_") or "relation"


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


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


def _relation_family(value: Any) -> str:
    relation = re.sub(r"\s+", " ", str(value or "").strip().lower())
    groups = {
        "Orientation": {"facing", "facing away from", "parallel to", "perpendicular to"},
        "Projective": {
            "above",
            "behind",
            "below",
            "beneath",
            "beside",
            "in front of",
            "in the middle of",
            "left of",
            "on top of",
            "over",
            "right of",
            "under",
        },
        "Topological": {
            "among",
            "at",
            "between",
            "connected to",
            "consists of",
            "contains",
            "detached from",
            "enclosed by",
            "has as a part",
            "in",
            "inside",
            "on",
            "out of",
            "outside",
            "part of",
            "surrounding",
            "touching",
            "with",
            "within",
        },
        "Proximity": {"by", "close to", "far away from", "far from", "near"},
        "Adjacency": {
            "adjacent to",
            "ahead of",
            "alongside",
            "at the back of",
            "at the edge of",
            "at the left side of",
            "at the right side of",
            "at the side of",
            "attached to",
            "against",
            "next to",
        },
        "Directional": {
            "across",
            "across from",
            "along",
            "around",
            "away from",
            "deep down",
            "down",
            "down from",
            "from",
            "into",
            "off",
            "opposite to",
            "past",
            "through",
            "to",
            "toward",
            "up",
        },
    }
    for family, relations in groups.items():
        if relation in relations:
            return family
    return "other"


def _binary_auroc(scores: list[float], labels: list[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if int(label) == 1]
    negatives = [score for score, label in zip(scores, labels) if int(label) == 0]
    if not positives or not negatives:
        return 0.0

    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return float(wins / (len(positives) * len(negatives)))


def _binary_average_precision(scores: list[float], labels: list[int]) -> float:
    total_positives = sum(1 for label in labels if int(label) == 1)
    if total_positives == 0:
        return 0.0

    true_positives = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(
        sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True),
        start=1,
    ):
        if int(label) == 1:
            true_positives += 1
            precision_sum += true_positives / rank
    return float(precision_sum / total_positives)


def _binary_metrics(records: list[Mapping], threshold: float) -> dict[str, float]:
    scores = [float(record["score"]) for record in records]
    labels = [int(record["label"]) for record in records]
    correct = [float((score > threshold) == bool(label)) for score, label in zip(scores, labels)]
    label_margins = [
        score - threshold if label else threshold - score for score, label in zip(scores, labels)
    ]
    positive_scores = [score for score, label in zip(scores, labels) if label == 1]
    negative_scores = [score for score, label in zip(scores, labels) if label == 0]

    return {
        "acc": _mean(correct),
        "auroc": _binary_auroc(scores, labels),
        "average_precision": _binary_average_precision(scores, labels),
        "label_margin_mean": _mean(label_margins),
        "score_gap_mean": _mean(positive_scores) - _mean(negative_scores),
        "n_examples": float(len(records)),
    }


def compute_vsr_metrics(
    records: Iterable[Mapping],
    *,
    threshold: float = 0.0,
    include_relation_metrics: bool = True,
) -> dict[str, float]:
    records = list(records)
    metrics = _binary_metrics(records, threshold)
    if not include_relation_metrics:
        return metrics

    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for record in records:
        grouped[str(record["relation"])].append(record)

    for relation, relation_records in sorted(grouped.items()):
        relation_name = _metric_safe_name(relation)
        relation_metrics = _binary_metrics(relation_records, threshold)
        for key, value in relation_metrics.items():
            metrics[f"relation/{relation_name}_{key}"] = value
    return metrics


def _labels_to_list(labels) -> list[int]:
    if isinstance(labels, torch.Tensor):
        return [int(label) for label in labels.detach().cpu().tolist()]
    return [int(label) for label in labels]


def _score_vsr_batch(
    adapter: PretrainedModelAdapter,
    images: torch.Tensor,
    captions: list[str],
    *,
    score: str = "itc",
) -> torch.Tensor:
    return encode_pair_scores(adapter, images, captions, score=score)


def evaluate_vsr_task(
    adapter: PretrainedModelAdapter,
    datamodule,
    *,
    score: str = "itc",
    threshold: float = 0.0,
    setup_stage: str = "validate",
    include_relation_metrics: bool = False,
) -> dict[str, float]:
    """Run VSR zero-shot image-caption scoring without a Lightning Trainer."""
    if score not in SUPPORTED_ZERO_SHOT_SCORES:
        raise ValueError("VSR score must be `itc` or `cosine`.")

    datamodule.setup(stage=setup_stage)
    records = []
    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            images = batch["image"]
            captions = list(batch["caption"])
            labels = _labels_to_list(batch["label"])
            relations = list(batch["relation"])

            scores = _score_vsr_batch(adapter, images, captions, score=score)
            for idx, relation in enumerate(relations):
                records.append(
                    {
                        "relation": relation,
                        "label": labels[idx],
                        "score": float(scores[idx].detach().cpu()),
                    }
                )

    metrics = compute_vsr_metrics(
        records,
        threshold=threshold,
        include_relation_metrics=include_relation_metrics,
    )
    if include_relation_metrics:
        return metrics
    return {key: metrics[key] for key in VSR_KEY_METRICS if key in metrics}


def score_vsr_examples(
    adapter: PretrainedModelAdapter,
    datamodule,
    *,
    score: str = "itc",
    threshold: float = 0.0,
    setup_stage: str = "validate",
    split: str | None = None,
) -> list[dict[str, Any]]:
    """Return per-example VSR zero-shot scores.

    ``signed_margin`` mirrors the existing thresholded VSR evaluator semantics:
    it is positive when the score is on the correct side of the decision
    threshold and negative otherwise.
    """
    if score not in SUPPORTED_ZERO_SHOT_SCORES:
        raise ValueError("VSR score must be `itc` or `cosine`.")

    datamodule.setup(stage=setup_stage)
    split_name = split or str(getattr(getattr(datamodule, "hparams", None), "split", "test"))
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            captions = list(batch["caption"])
            labels = _labels_to_list(batch["label"])
            relations = list(batch["relation"])
            scores = _score_vsr_batch(adapter, batch["image"], captions, score=score)
            for idx, relation in enumerate(relations):
                score_value = float(scores[idx].detach().cpu())
                label = int(labels[idx])
                predicted = int(score_value > threshold)
                signed_margin = score_value - threshold if label else threshold - score_value
                rows.append(
                    {
                        "dataset": "vsr",
                        "split": split_name,
                        "example_id": str(
                            _metadata_value(batch, "example_id", idx, f"{split_name}:{len(rows)}")
                        ),
                        "category": _relation_family(relation),
                        "sub_category": str(relation),
                        "caption": str(captions[idx]),
                        "image_id": str(_metadata_value(batch, "filename", idx, "")),
                        "caption_id": "",
                        "label": label,
                        "relation": str(relation),
                        "relation_family": _relation_family(relation),
                        "subject": str(_metadata_value(batch, "subj", idx, "")),
                        "object": str(_metadata_value(batch, "obj", idx, "")),
                        "subj": str(_metadata_value(batch, "subj", idx, "")),
                        "obj": str(_metadata_value(batch, "obj", idx, "")),
                        "score": score_value,
                        "positive_score": score_value,
                        "predicted_label": predicted,
                        "correct": bool(predicted == label),
                        "signed_margin": float(signed_margin),
                    }
                )
    return rows


class VSREvaluator(LightningModule):
    """Zero-shot Visual Spatial Reasoning image-caption evaluator."""

    def __init__(
        self,
        ckpt_path: str,
        bert_size: str = "auto",
        max_text_len: int = 64,
        score: str = "itc",
        threshold: float = 0.0,
        include_relation_metrics: bool = True,
        adapter: PretrainedModelAdapter | None = None,
    ) -> None:
        super().__init__()
        if score not in SUPPORTED_ZERO_SHOT_SCORES:
            raise ValueError("VSR score must be `itc` or `cosine`.")
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
        images = batch["image"]
        captions = list(batch["caption"])
        labels = _labels_to_list(batch["label"])
        relations = list(batch["relation"])

        scores = _score_vsr_batch(self.adapter, images, captions, score=self.hparams.score)
        for idx, relation in enumerate(relations):
            self.records.append(
                {
                    "relation": relation,
                    "label": labels[idx],
                    "score": float(scores[idx].detach().cpu()),
                }
            )

    def on_validation_epoch_end(self) -> None:
        metrics = compute_vsr_metrics(
            self.records,
            threshold=float(self.hparams.threshold),
            include_relation_metrics=bool(self.hparams.include_relation_metrics),
        )
        if not self.hparams.include_relation_metrics:
            metrics = {key: metrics[key] for key in VSR_KEY_METRICS if key in metrics}
        self.log_dict(metrics, sync_dist=False, logger=False)
