from __future__ import annotations

import logging
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F
from lightning import LightningModule

from src.evaluation.adapters import PretrainedModelAdapter
from src.evaluation.scoring import (
    SUPPORTED_ZERO_SHOT_SCORES,
    normalize_for_zero_shot_score,
    similarity_matrix_from_features,
)

log = logging.getLogger(__name__)


def _recall_at_k(sims: torch.Tensor, ground_truth: Mapping[int, int | list[int]], k: int) -> float:
    if sims.numel() == 0:
        return 0.0

    k = min(k, sims.shape[1])
    topk = sims.topk(k, dim=1).indices.detach().cpu()
    correct = 0
    for row_idx in range(topk.shape[0]):
        target = ground_truth[row_idx]
        if isinstance(target, list):
            target_set = set(target)
            if any(int(pred) in target_set for pred in topk[row_idx]):
                correct += 1
        elif int(target) in topk[row_idx].tolist():
            correct += 1
    return correct / sims.shape[0]


def compute_coco_karpathy_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    txt2img: Mapping[int, int],
    img2txt: Mapping[int, list[int]],
    ks: Iterable[int] = (1, 5, 10),
    score: str = "itc",
) -> dict[str, float]:
    normalized_image_features = F.normalize(image_features, dim=-1)
    normalized_text_features = F.normalize(text_features, dim=-1)
    sims_i2t = similarity_matrix_from_features(image_features, text_features, score=score)
    sims_t2i = sims_i2t.t()

    metrics = {}
    for k in ks:
        metrics[f"i2t_r{k}"] = _recall_at_k(sims_i2t, img2txt, k)
        metrics[f"t2i_r{k}"] = _recall_at_k(sims_t2i, txt2img, k)

    recall_values = [value for key, value in metrics.items() if "_r" in key]
    metrics["mean_recall"] = (
        float(sum(recall_values) / len(recall_values)) if recall_values else 0.0
    )
    metrics["modality_gap"] = float(
        torch.linalg.vector_norm(
            normalized_image_features.mean(dim=0) - normalized_text_features.mean(dim=0)
        )
        .detach()
        .cpu()
    )
    metrics["modality_gap_cosine"] = float(
        (
            1
            - F.cosine_similarity(
                normalized_image_features.mean(dim=0),
                normalized_text_features.mean(dim=0),
                dim=0,
            )
        )
        .detach()
        .cpu()
    )
    return metrics


class CocoKarpathyEvaluator(LightningModule):
    """Zero-shot COCO Karpathy retrieval evaluator."""

    def __init__(
        self,
        ckpt_path: str,
        bert_size: str = "auto",
        max_text_len: int = 64,
        text_batch_size: int = 256,
        score: str = "itc",
        adapter: PretrainedModelAdapter | None = None,
    ) -> None:
        super().__init__()
        if score not in SUPPORTED_ZERO_SHOT_SCORES:
            raise ValueError("COCO retrieval score must be `itc` or `cosine`.")
        self.save_hyperparameters(ignore=["adapter"], logger=False)
        self.adapter = adapter or PretrainedModelAdapter(
            ckpt_path=ckpt_path,
            bert_size=bert_size,
            max_text_len=max_text_len,
        )
        self.text_features = []
        self.image_features = []

    def on_validation_start(self) -> None:
        self.text_features.clear()
        self.image_features.clear()

        texts = self.trainer.datamodule.val_set.text
        normalize = normalize_for_zero_shot_score(self.hparams.score)
        for start in range(0, len(texts), self.hparams.text_batch_size):
            text_batch = texts[start : start + self.hparams.text_batch_size]
            self.text_features.append(self.adapter.encode_texts(text_batch, normalize=normalize))

    def validation_step(self, batch, batch_idx):
        normalize = normalize_for_zero_shot_score(self.hparams.score)
        self.image_features.append(self.adapter.encode_images(batch["image"], normalize=normalize))

    def on_validation_epoch_end(self) -> None:
        image_features = torch.cat(self.image_features, dim=0)
        text_features = torch.cat(self.text_features, dim=0)
        self.image_features.clear()
        self.text_features.clear()

        dataset = self.trainer.datamodule.val_set
        metrics = compute_coco_karpathy_metrics(
            image_features=image_features.detach().float().cpu(),
            text_features=text_features.detach().float().cpu(),
            txt2img=dataset.txt2img,
            img2txt=dataset.img2txt,
            score=self.hparams.score,
        )
        self.log_dict(metrics, sync_dist=False, logger=False, on_epoch=True)
        for key, value in metrics.items():
            log.info("%s: %.4f", key, value)
