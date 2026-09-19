from __future__ import annotations

import re
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

SUGARCREPE_KEY_METRICS = ("itt_acc", "p1_acc", "p2_acc")


def compute_sugarcrepe_metrics(records: Iterable[Mapping]) -> dict[str, float]:
    grouped = defaultdict(lambda: {"itt": [], "p1": [], "p2": []})
    all_records = {"itt": [], "p1": [], "p2": []}

    for record in records:
        category = str(record["category"])
        for key in ("itt", "p1", "p2"):
            value = float(record[key])
            all_records[key].append(value)
            grouped[category][key].append(value)

    def mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def metric_safe_name(value: str) -> str:
        value = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
        return value.strip("_") or "category"

    metrics = {
        "itt_acc": mean(all_records["itt"]),
        "p1_acc": mean(all_records["p1"]),
        "p2_acc": mean(all_records["p2"]),
    }
    for category, values in sorted(grouped.items()):
        category_name = metric_safe_name(category)
        metrics[f"{category_name}_itt_acc"] = mean(values["itt"])
        metrics[f"{category_name}_p1_acc"] = mean(values["p1"])
        metrics[f"{category_name}_p2_acc"] = mean(values["p2"])
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


def score_sugarcrepe_examples(
    adapter: PretrainedModelAdapter,
    datamodule,
    *,
    score: str = "itc",
    setup_stage: str = "validate",
    split: str = "sugarcrepe_pp",
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Return per-example SugarCrepe++ image-text scores.

    This helper is additive to the aggregate evaluator path and intentionally
    preserves benchmark-native metadata for paper analysis.
    """
    if score not in {*SUPPORTED_ZERO_SHOT_SCORES, "itm"}:
        raise ValueError("SugarCrepe score must be `itc`, `cosine`, or `itm`.")

    datamodule.setup(stage=setup_stage)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            images = batch["image"]
            captions = list(batch["caption"])
            captions2 = list(batch["caption2"])
            negatives = list(batch["negative_caption"])
            categories = list(batch["category"])

            if score == "itm":
                p1_scores = adapter.score_image_text_pairs(images, captions, score="itm")
                p2_scores = adapter.score_image_text_pairs(images, captions2, score="itm")
                negative_scores = adapter.score_image_text_pairs(images, negatives, score="itm")
            else:
                normalize = normalize_for_zero_shot_score(score)
                image_features = adapter.encode_images(images, normalize=normalize)
                text_features = adapter.encode_texts(
                    captions + captions2 + negatives,
                    normalize=normalize,
                )
                batch_size = image_features.shape[0]
                p1_features = text_features[:batch_size]
                p2_features = text_features[batch_size : 2 * batch_size]
                negative_features = text_features[2 * batch_size :]
                p1_scores = pair_scores_from_features(image_features, p1_features, score=score)
                p2_scores = pair_scores_from_features(image_features, p2_features, score=score)
                negative_scores = pair_scores_from_features(
                    image_features,
                    negative_features,
                    score=score,
                )

            for idx, category in enumerate(categories):
                p1_score = float(p1_scores[idx].detach().cpu())
                p2_score = float(p2_scores[idx].detach().cpu())
                negative_score = float(negative_scores[idx].detach().cpu())
                margin_p1 = p1_score - negative_score
                margin_p2 = p2_score - negative_score
                rows.append(
                    {
                        "dataset": "sugarcrepe_pp",
                        "split": split,
                        "example_id": str(
                            _metadata_value(
                                batch,
                                "example_id",
                                idx,
                                f"{category}:{_metadata_value(batch, 'id', idx, idx)}",
                            )
                        ),
                        "category": str(category),
                        "sub_category": "",
                        "image_id": str(_metadata_value(batch, "filename", idx, "")),
                        "caption_id": "",
                        "positive_caption_1": str(captions[idx]),
                        "positive_caption_2": str(captions2[idx]),
                        "negative_caption": str(negatives[idx]),
                        "positive_caption": str(captions[idx]),
                        "p1_score": p1_score,
                        "p2_score": p2_score,
                        "negative_score": negative_score,
                        "positive_score": min(p1_score, p2_score),
                        "margin_p1": margin_p1,
                        "margin_p2": margin_p2,
                        "min_positive_margin": min(margin_p1, margin_p2),
                        "margin": min(margin_p1, margin_p2),
                        "correct": bool(margin_p1 > 0 and margin_p2 > 0),
                        "source_file": str(_metadata_value(batch, "source_file", idx, "")),
                        "line_number": _metadata_value(batch, "line_number", idx, ""),
                    }
                )
                if max_samples is not None and len(rows) >= max_samples:
                    return rows
    return rows


def evaluate_sugarcrepe_task(
    adapter: PretrainedModelAdapter,
    datamodule,
    *,
    score: str = "itc",
    setup_stage: str = "validate",
    include_category_metrics: bool = False,
) -> dict[str, float]:
    """Run SugarCrepe++ zero-shot scoring without a Lightning Trainer."""
    if score not in {*SUPPORTED_ZERO_SHOT_SCORES, "itm"}:
        raise ValueError("SugarCrepe score must be `itc`, `cosine`, or `itm`.")

    datamodule.setup(stage=setup_stage)
    records = []
    for batch in datamodule.val_dataloader():
        images = batch["image"]
        captions = list(batch["caption"])
        captions2 = list(batch["caption2"])
        negatives = list(batch["negative_caption"])
        categories = list(batch["category"])

        if score == "itm":
            p1_scores = adapter.score_image_text_pairs(images, captions, score="itm")
            p2_scores = adapter.score_image_text_pairs(images, captions2, score="itm")
            negative_scores = adapter.score_image_text_pairs(images, negatives, score="itm")
        else:
            normalize = normalize_for_zero_shot_score(score)
            image_features = adapter.encode_images(images, normalize=normalize)
            text_features = adapter.encode_texts(
                captions + captions2 + negatives,
                normalize=normalize,
            )
            batch_size = image_features.shape[0]
            p1_features = text_features[:batch_size]
            p2_features = text_features[batch_size : 2 * batch_size]
            negative_features = text_features[2 * batch_size :]

            p1_scores = pair_scores_from_features(image_features, p1_features, score=score)
            p2_scores = pair_scores_from_features(image_features, p2_features, score=score)
            negative_scores = pair_scores_from_features(
                image_features,
                negative_features,
                score=score,
            )

        p1_correct = p1_scores > negative_scores
        p2_correct = p2_scores > negative_scores
        itt_correct = p1_correct & p2_correct
        for idx, category in enumerate(categories):
            records.append(
                {
                    "category": category,
                    "itt": bool(itt_correct[idx].detach().cpu()),
                    "p1": bool(p1_correct[idx].detach().cpu()),
                    "p2": bool(p2_correct[idx].detach().cpu()),
                }
            )

    metrics = compute_sugarcrepe_metrics(records)
    if include_category_metrics:
        return metrics
    return {key: metrics[key] for key in SUGARCREPE_KEY_METRICS if key in metrics}


class SugarCrepeEvaluator(LightningModule):
    """Zero-shot SugarCrepe++ image-to-text evaluator."""

    def __init__(
        self,
        ckpt_path: str,
        bert_size: str = "auto",
        max_text_len: int = 64,
        score: str = "itc",
        adapter: PretrainedModelAdapter | None = None,
    ) -> None:
        super().__init__()
        if score not in {*SUPPORTED_ZERO_SHOT_SCORES, "itm"}:
            raise ValueError("SugarCrepe score must be `itc`, `cosine`, or `itm`.")
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
        images = batch["image"]
        captions = list(batch["caption"])
        captions2 = list(batch["caption2"])
        negatives = list(batch["negative_caption"])
        categories = list(batch["category"])

        if self.hparams.score == "itm":
            p1_scores = self.adapter.score_image_text_pairs(images, captions, score="itm")
            p2_scores = self.adapter.score_image_text_pairs(images, captions2, score="itm")
            negative_scores = self.adapter.score_image_text_pairs(images, negatives, score="itm")
        else:
            normalize = normalize_for_zero_shot_score(self.hparams.score)
            image_features = self.adapter.encode_images(images, normalize=normalize)
            text_features = self.adapter.encode_texts(
                captions + captions2 + negatives,
                normalize=normalize,
            )
            batch_size = image_features.shape[0]
            p1_features = text_features[:batch_size]
            p2_features = text_features[batch_size : 2 * batch_size]
            negative_features = text_features[2 * batch_size :]

            p1_scores = pair_scores_from_features(
                image_features,
                p1_features,
                score=self.hparams.score,
            )
            p2_scores = pair_scores_from_features(
                image_features,
                p2_features,
                score=self.hparams.score,
            )
            negative_scores = pair_scores_from_features(
                image_features,
                negative_features,
                score=self.hparams.score,
            )

        p1_correct = p1_scores > negative_scores
        p2_correct = p2_scores > negative_scores
        itt_correct = p1_correct & p2_correct

        for idx, category in enumerate(categories):
            self.records.append(
                {
                    "category": category,
                    "itt": bool(itt_correct[idx].detach().cpu()),
                    "p1": bool(p1_correct[idx].detach().cpu()),
                    "p2": bool(p2_correct[idx].detach().cpu()),
                }
            )

    def on_validation_epoch_end(self) -> None:
        metrics = compute_sugarcrepe_metrics(self.records)
        self.log_dict(metrics, sync_dist=False, logger=False)
