from __future__ import annotations

import logging
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn

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


def compute_flickr30k_metrics(
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


class RetrievalProjectionHead(nn.Module):
    def __init__(
        self, feature_dim: int = 768, projection_dim: int = 768, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), dim=-1)


class Flickr30KRetrievalEvaluator(LightningModule):
    """Flickr30K retrieval evaluator with zero-shot and lightweight fit modes."""

    def __init__(
        self,
        ckpt_path: str,
        mode: str = "zeroshot",
        bert_size: str = "auto",
        max_text_len: int = 64,
        text_batch_size: int = 256,
        score: str = "itc",
        feature_dim: int = 768,
        projection_dim: int = 768,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        logit_scale_init: float = 2.6592,
        train_metric_prefix: str = "train",
        adapter: PretrainedModelAdapter | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"zeroshot", "retrieval_adapter_tune"}:
            raise ValueError(
                "Flickr30K retrieval mode must be `zeroshot` or `retrieval_adapter_tune`."
            )
        if score not in SUPPORTED_ZERO_SHOT_SCORES:
            raise ValueError("Flickr30K retrieval score must be `itc` or `cosine`.")

        self.save_hyperparameters(ignore=["adapter"], logger=False)
        self.adapter = adapter or PretrainedModelAdapter(
            ckpt_path=ckpt_path,
            bert_size=bert_size,
            max_text_len=max_text_len,
        )
        self.image_head = RetrievalProjectionHead(feature_dim, projection_dim, dropout=dropout)
        self.text_head = RetrievalProjectionHead(feature_dim, projection_dim, dropout=dropout)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))
        self.text_features = []
        self.image_features = []

    @property
    def is_adapter_tune(self) -> bool:
        return self.hparams.mode == "retrieval_adapter_tune"

    def _project_images(self, images: torch.Tensor) -> torch.Tensor:
        normalize = (
            False if self.is_adapter_tune else normalize_for_zero_shot_score(self.hparams.score)
        )
        features = self.adapter.encode_images(images, normalize=normalize)
        return self.image_head(features) if self.is_adapter_tune else features

    def _project_texts(self, texts: Iterable[str]) -> torch.Tensor:
        normalize = (
            False if self.is_adapter_tune else normalize_for_zero_shot_score(self.hparams.score)
        )
        features = self.adapter.encode_texts(texts, normalize=normalize)
        return self.text_head(features) if self.is_adapter_tune else features

    def training_step(self, batch, batch_idx):
        image_features = self._project_images(batch["image"])
        text_features = self._project_texts(list(batch["caption"]))
        logits = self.logit_scale.exp().clamp(max=100.0) * image_features @ text_features.t()
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_i2t + loss_t2i)
        train_prefix = self.hparams.train_metric_prefix.strip("/") or "train"
        self.log(
            f"{train_prefix}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=False,
        )
        return loss

    def on_validation_start(self) -> None:
        self.text_features.clear()
        self.image_features.clear()

        texts = self.trainer.datamodule.val_set.text
        for start in range(0, len(texts), self.hparams.text_batch_size):
            text_batch = texts[start : start + self.hparams.text_batch_size]
            self.text_features.append(self._project_texts(text_batch))

    def validation_step(self, batch, batch_idx):
        self.image_features.append(self._project_images(batch["image"]))

    def on_validation_epoch_end(self) -> None:
        image_features = torch.cat(self.image_features, dim=0)
        text_features = torch.cat(self.text_features, dim=0)
        self.image_features.clear()
        self.text_features.clear()

        dataset = self.trainer.datamodule.val_set
        metrics = compute_flickr30k_metrics(
            image_features=image_features.detach().float().cpu(),
            text_features=text_features.detach().float().cpu(),
            txt2img=dataset.txt2img,
            img2txt=dataset.img2txt,
            score=self.hparams.score,
        )
        self.log_dict(metrics, sync_dist=False, logger=False, on_epoch=True)
        for key, value in metrics.items():
            log.info("%s: %.4f", key, value)

    def configure_optimizers(self):
        params = (
            list(self.image_head.parameters())
            + list(self.text_head.parameters())
            + [self.logit_scale]
        )
        optimizer = torch.optim.AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.trainer.max_epochs)
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
