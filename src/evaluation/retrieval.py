from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn

from src.evaluation.adapters import PretrainedModelAdapter
from src.evaluation.coco_karpathy import compute_coco_karpathy_metrics
from src.evaluation.flickr30k import compute_flickr30k_metrics
from src.evaluation.scoring import (
    SUPPORTED_ZERO_SHOT_SCORES,
    normalize_for_zero_shot_score,
    similarity_matrix_from_features,
)

try:
    import torch.distributed as dist  # type: ignore[attr-defined]
except Exception:
    dist = None  # fallback if torch.distributed is unavailable

log = logging.getLogger(__name__)

RETRIEVAL_METRICS = {
    "coco_karpathy": compute_coco_karpathy_metrics,
    "flickr30k": compute_flickr30k_metrics,
}


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


def compute_retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    txt2img: Mapping[int, int],
    img2txt: Mapping[int, list[int]],
    ks: Iterable[int] = (1, 5, 10),
    score: str = "itc",
) -> dict[str, float]:
    sims_i2t = similarity_matrix_from_features(image_features, text_features, score=score)
    sims_t2i = sims_i2t.t()

    metrics = {}
    for k in ks:
        metrics[f"i2t_r{k}"] = _recall_at_k(sims_i2t, img2txt, k)
        metrics[f"t2i_r{k}"] = _recall_at_k(sims_t2i, txt2img, k)
    metrics["mean_recall"] = float(np.mean(list(metrics.values()))) if metrics else 0.0
    return metrics


class RetrievalEvaluator(LightningModule):
    """COCO retrieval evaluator backed by the shared pretrained-model adapter."""

    def __init__(
        self,
        ckpt_path: str,
        bert_size: str = "auto",
        max_text_len: int = 64,
        text_batch_size: int = 256,
        score: str = "itc",
        adapter: PretrainedModelAdapter | None = None,
    ):
        super().__init__()
        if score not in SUPPORTED_ZERO_SHOT_SCORES:
            raise ValueError("Retrieval score must be `itc` or `cosine`.")
        self.save_hyperparameters(ignore=["adapter"], logger=False)
        self.adapter = adapter or PretrainedModelAdapter(
            ckpt_path=ckpt_path,
            bert_size=bert_size,
            max_text_len=max_text_len,
        )
        self.text_features: list[torch.Tensor] = []
        self.visual_features: list[torch.Tensor] = []

    def on_validation_start(self) -> None:
        self.text_features.clear()
        self.visual_features.clear()

        texts = self.trainer.datamodule.val_set.text
        normalize = normalize_for_zero_shot_score(self.hparams.score)
        for start in range(0, len(texts), self.hparams.text_batch_size):
            text_batch = texts[start : start + self.hparams.text_batch_size]
            self.text_features.append(self.adapter.encode_texts(text_batch, normalize=normalize))

    def validation_step(self, batch, batch_idx):
        images = batch["image"] if isinstance(batch, dict) else batch[0]
        normalize = normalize_for_zero_shot_score(self.hparams.score)
        self.visual_features.append(self.adapter.encode_images(images, normalize=normalize))

    def on_validation_end(self) -> None:
        """
        Gather features across all distributed ranks and compute the retrieval
        metrics once on rank zero.  When running in distributed mode the
        validation dataloader may be sharded across devices, so naive
        concatenation on each rank would produce incomplete (and
        inconsistent) feature sets.  We gather the lists of features
        using `dist.all_gather_object`, which can handle tensors of
        different shapes on each rank, then concatenate them on rank
        zero.  The metrics are logged with `sync_dist=True` so that
        other ranks receive the same values.
        """
        # Concatenate features on this rank
        local_text = torch.cat(self.text_features, dim=0)
        local_visual = torch.cat(self.visual_features, dim=0)
        # Clear the buffers for the next epoch
        self.text_features.clear()
        self.visual_features.clear()

        # If torch.distributed is available and initialised, gather the
        # feature tensors across ranks.  Otherwise, operate on the local
        # features directly.
        all_text_features = [local_text]
        all_visual_features = [local_visual]
        world_size = 1
        rank = 0
        if dist is not None and dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            # Gather lists of tensors from each rank.  We use
            # `all_gather_object` because the number of samples per rank
            # may differ.  The gathered lists will contain tensors from
            # all ranks in order of rank.
            gathered_text: list[torch.Tensor] = [None for _ in range(world_size)]  # type: ignore[var-annotated]
            gathered_visual: list[torch.Tensor] = [None for _ in range(world_size)]  # type: ignore[var-annotated]
            dist.all_gather_object(gathered_text, local_text.detach().cpu())  # type: ignore[call-arg]
            dist.all_gather_object(gathered_visual, local_visual.detach().cpu())  # type: ignore[call-arg]
            all_text_features = gathered_text
            all_visual_features = gathered_visual

        # Compute metrics only on rank zero
        if rank == 0:
            text_features = torch.cat(all_text_features, dim=0).float()
            visual_features = torch.cat(all_visual_features, dim=0).float()
            dataset = self.trainer.datamodule.val_set
            metrics = compute_retrieval_metrics(
                image_features=visual_features,
                text_features=text_features,
                txt2img=dataset.txt2img,
                img2txt=dataset.img2txt,
                score=self.hparams.score,
            )
        else:
            metrics = {
                k: 0.0
                for k in [
                    "i2t_r1",
                    "i2t_r5",
                    "i2t_r10",
                    "t2i_r1",
                    "t2i_r5",
                    "t2i_r10",
                    "mean_recall",
                ]
            }
        # Log metrics with sync_dist=True to broadcast the values to all ranks
        self.log_dict(metrics, sync_dist=True, logger=False)
        if rank == 0:
            for key, value in metrics.items():
                log.info("%s: %.4f", key, value)


class RetrievalAdapterHead(nn.Module):
    """Projection head used for retrieval adapter tuning."""

    def __init__(
        self,
        input_dim: int = 768,
        projection_dim: int = 768,
        residual: bool = True,
        residual_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.projection_dim = projection_dim
        self.use_residual = residual and input_dim == projection_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, projection_dim),
        )
        self.residual_scale = (
            nn.Parameter(torch.tensor(float(residual_scale_init))) if self.use_residual else None
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.net(features)
        if self.use_residual:
            projected = F.normalize(features, dim=-1) + self.residual_scale * projected
        return F.normalize(projected, dim=-1)


class RetrievalAdapterTuneEvaluator(LightningModule):
    """Zero-shot and retrieval-adapter-tuning evaluator for COCO/Flickr retrieval."""

    def __init__(
        self,
        ckpt_path: str,
        metric: str,
        mode: str = "zeroshot",
        protocol: str | None = None,
        freeze_backbone: bool = True,
        bert_size: str = "auto",
        max_text_len: int = 64,
        text_batch_size: int = 256,
        feature_dim: int = 768,
        projection_dim: int = 768,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        logit_scale_init: float = 2.6592,
        encoder_source: str | None = "target",
        validation_metric_prefix: str | None = None,
        test_metric_prefix: str | None = None,
        load_best_val_before_test: bool = True,
        adapter_head: str = "residual",
        residual_scale_init: float = 0.0,
        log_epoch0_metrics: bool = True,
        score: str = "itc",
        adapter: PretrainedModelAdapter | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"zeroshot", "retrieval_adapter_tune"}:
            raise ValueError(
                "Retrieval evaluator mode must be `zeroshot` or `retrieval_adapter_tune`."
            )
        if metric not in RETRIEVAL_METRICS:
            raise ValueError(f"Unknown retrieval metric `{metric}`.")
        if adapter_head not in {"linear", "residual"}:
            raise ValueError("Retrieval adapter_head must be `linear` or `residual`.")
        if score not in SUPPORTED_ZERO_SHOT_SCORES:
            raise ValueError("Retrieval score must be `itc` or `cosine`.")

        self.save_hyperparameters(ignore=["adapter"], logger=False)
        self.adapter = adapter or PretrainedModelAdapter(
            ckpt_path=ckpt_path,
            bert_size=bert_size,
            max_text_len=max_text_len,
            encoder_source=encoder_source,
        )
        if freeze_backbone:
            for param in self.adapter.parameters():
                param.requires_grad = False

        self.image_adapter = (
            RetrievalAdapterHead(
                feature_dim,
                projection_dim,
                residual=adapter_head == "residual",
                residual_scale_init=residual_scale_init,
            )
            if self.is_adapter_tune
            else None
        )
        self.text_adapter = (
            RetrievalAdapterHead(
                feature_dim,
                projection_dim,
                residual=adapter_head == "residual",
                residual_scale_init=residual_scale_init,
            )
            if self.is_adapter_tune
            else None
        )
        self.logit_scale = (
            nn.Parameter(torch.tensor(float(logit_scale_init))) if self.is_adapter_tune else None
        )
        self._split_features: dict[str, dict[str, list[torch.Tensor]]] = {}
        self._best_val_score = float("-inf")
        self._best_adapter_state: dict[str, Any] | None = None

    @property
    def is_adapter_tune(self) -> bool:
        return self.hparams.mode == "retrieval_adapter_tune"

    def _project_images(self, images: torch.Tensor) -> torch.Tensor:
        normalize = (
            False if self.is_adapter_tune else normalize_for_zero_shot_score(self.hparams.score)
        )
        features = self.adapter.encode_images(images, normalize=normalize)
        if self.image_adapter is not None:
            return self.image_adapter(features)
        return features

    def _project_base_images(self, images: torch.Tensor) -> torch.Tensor:
        normalize = normalize_for_zero_shot_score(self.hparams.score)
        return self.adapter.encode_images(images, normalize=normalize)

    def _project_base_texts(self, texts: Iterable[str]) -> torch.Tensor:
        normalize = normalize_for_zero_shot_score(self.hparams.score)
        return self.adapter.encode_texts(texts, normalize=normalize)

    def _project_texts(self, texts: Iterable[str]) -> torch.Tensor:
        normalize = (
            False if self.is_adapter_tune else normalize_for_zero_shot_score(self.hparams.score)
        )
        features = self.adapter.encode_texts(texts, normalize=normalize)
        if self.text_adapter is not None:
            return self.text_adapter(features)
        return features

    @staticmethod
    def _images(batch: Any) -> torch.Tensor:
        if isinstance(batch, Mapping):
            return batch["image"]
        return batch[0]

    @staticmethod
    def _captions(batch: Any) -> list[str]:
        if isinstance(batch, Mapping):
            return list(batch["caption"])
        return list(batch[-1])

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if not self.is_adapter_tune:
            raise RuntimeError("Zero-shot retrieval evaluator does not support training.")
        image_features = self._project_images(self._images(batch))
        text_features = self._project_texts(self._captions(batch))
        logits = self.logit_scale.exp().clamp(max=100.0) * image_features @ text_features.t()
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=False,
        )
        return loss

    def _reset_split(self, split: str) -> None:
        self._split_features[split] = {"image": [], "text": []}
        if self._collect_epoch0_metrics(split):
            self._split_features[split]["image_epoch0"] = []
            self._split_features[split]["text_epoch0"] = []

    def _collect_epoch0_metrics(self, split: str) -> bool:
        return bool(self.is_adapter_tune and self.hparams.log_epoch0_metrics and split == "test")

    def _encode_split_texts(self, split: str) -> None:
        dataset = self._split_dataset(split)
        texts = getattr(dataset, "text")
        for start in range(0, len(texts), self.hparams.text_batch_size):
            text_batch = texts[start : start + self.hparams.text_batch_size]
            self._split_features[split]["text"].append(self._project_texts(text_batch))
            if self._collect_epoch0_metrics(split):
                self._split_features[split]["text_epoch0"].append(
                    self._project_base_texts(text_batch)
                )

    def _split_dataset(self, split: str):
        datamodule = self.trainer.datamodule
        return datamodule.test_set if split == "test" else datamodule.val_set

    def _eval_step(self, batch: Any, split: str) -> None:
        images = self._images(batch)
        self._split_features[split]["image"].append(self._project_images(images))
        if self._collect_epoch0_metrics(split):
            self._split_features[split]["image_epoch0"].append(self._project_base_images(images))

    def _metric_key(self, split: str, metric: str) -> str:
        if split == "test":
            prefix = self.hparams.test_metric_prefix
        else:
            prefix = self.hparams.validation_metric_prefix
        prefix = str(prefix or "").strip("/")
        return f"{prefix}/{metric}" if prefix else metric

    def _finish_split(self, split: str) -> dict[str, float]:
        chunks = self._split_features[split]
        image_features = torch.cat(chunks["image"], dim=0)
        text_features = torch.cat(chunks["text"], dim=0)
        self._reset_split(split)

        dataset = self._split_dataset(split)
        metric_fn = RETRIEVAL_METRICS[self.hparams.metric]
        raw_metrics = metric_fn(
            image_features=image_features.detach().float().cpu(),
            text_features=text_features.detach().float().cpu(),
            txt2img=dataset.txt2img,
            img2txt=dataset.img2txt,
            score=self.hparams.score,
        )
        metrics = {self._metric_key(split, key): value for key, value in raw_metrics.items()}
        if self._collect_epoch0_metrics(split):
            epoch0_image = torch.cat(chunks["image_epoch0"], dim=0)
            epoch0_text = torch.cat(chunks["text_epoch0"], dim=0)
            epoch0_metrics = metric_fn(
                image_features=epoch0_image.detach().float().cpu(),
                text_features=epoch0_text.detach().float().cpu(),
                txt2img=dataset.txt2img,
                img2txt=dataset.img2txt,
                score=self.hparams.score,
            )
            for key, value in epoch0_metrics.items():
                metrics[self._metric_key(split, f"epoch0_{key}")] = value
            if "mean_recall" in raw_metrics and "mean_recall" in epoch0_metrics:
                metrics[self._metric_key(split, "fit_delta_mean_recall")] = (
                    raw_metrics["mean_recall"] - epoch0_metrics["mean_recall"]
                )
        if split == "val" and self.is_adapter_tune:
            score = float(raw_metrics.get("mean_recall", float("-inf")))
            if score > self._best_val_score:
                self._best_val_score = score
                self._best_adapter_state = {
                    "image_adapter": deepcopy(self.image_adapter.state_dict()),
                    "text_adapter": deepcopy(self.text_adapter.state_dict()),
                    "logit_scale": self.logit_scale.detach().cpu().clone(),
                }
        self.log_dict(metrics, sync_dist=False, logger=False, on_epoch=True)
        for key, value in metrics.items():
            log.info("%s: %.4f", key, value)
        return metrics

    def on_validation_epoch_start(self) -> None:
        self._reset_split("val")
        self._encode_split_texts("val")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._eval_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        self._finish_split("val")

    def on_test_epoch_start(self) -> None:
        if (
            self.is_adapter_tune
            and self.hparams.load_best_val_before_test
            and self._best_adapter_state is not None
        ):
            self.image_adapter.load_state_dict(self._best_adapter_state["image_adapter"])
            self.text_adapter.load_state_dict(self._best_adapter_state["text_adapter"])
            self.logit_scale.data.copy_(self._best_adapter_state["logit_scale"].to(self.device))
        self._reset_split("test")
        self._encode_split_texts("test")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._eval_step(batch, "test")

    def on_test_epoch_end(self) -> None:
        self._finish_split("test")

    def configure_optimizers(self):
        if not self.is_adapter_tune:
            return None
        params = (
            list(self.image_adapter.parameters())
            + list(self.text_adapter.parameters())
            + [self.logit_scale]
        )
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.trainer.max_epochs),
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
