"""Canonical CLIP and SigLIP baselines for controlled pretraining runs."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models import ViT_B_16_Weights, vit_b_16
from transformers import BertModel, BertTokenizer

try:
    from torch.distributed.nn.functional import all_gather as all_gather_with_grad
except ImportError:  # pragma: no cover - older torch fallback
    all_gather_with_grad = None

from src.models.base import BaseModule, compute_optimizer_step_schedule

logger = logging.getLogger(__name__)


def build_global_targets(
    local_batch_size: int,
    *,
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    """Return global positive column indices for local rows on one rank."""
    return torch.arange(local_batch_size, device=device, dtype=torch.long) + (
        int(rank) * int(local_batch_size)
    )


def build_siglip_labels(
    local_batch_size: int,
    global_batch_size: int,
    *,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return SigLIP pairwise labels with positives at local/global matches."""
    labels = -torch.ones((local_batch_size, global_batch_size), device=device, dtype=dtype)
    targets = build_global_targets(local_batch_size, rank=rank, device=device)
    labels[torch.arange(local_batch_size, device=device), targets] = 1.0
    return labels


def is_no_decay_parameter(name: str, param: nn.Parameter) -> bool:
    """Name-based AdamW exclusion rules for CLIP/SigLIP baselines."""
    lname = name.lower()
    if not param.requires_grad:
        return True
    if lname.endswith(".bias"):
        return True
    if "norm" in lname or "layernorm" in lname or "ln_" in lname:
        return True
    if "logit_scale" in lname or "logit_bias" in lname:
        return True
    return False


class _ContrastiveBaseline(BaseModule):
    loss_name = "clip"

    def __init__(
        self,
        optimizer: object,
        name: str | None = None,
        shared_dim: int = 768,
        text_encoder_name: str = "bert-base-uncased",
        max_text_len: int = 64,
        vision_encoder: object | None = None,
        logit_scale_init: float | None = None,
        logit_scale_max: float = 100.0,
        lr_schedule: object | None = None,
        lr_groups: object | None = None,
        ipe_scale: float = 1.0,
        warmup: int | float = 4,
        lr_scheduler_vis: object | None = None,
        lr_scheduler_text: object | None = None,
        vision_model: nn.Module | None = None,
        vis_encoder: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
        tokenizer: object | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["vision_model", "vis_encoder", "text_encoder", "tokenizer"],
            logger=False,
        )
        self.shared_dim = int(shared_dim)
        self.max_text_len = int(max_text_len)
        self.logit_scale_max = float(logit_scale_max)

        injected_vision = vision_model or vis_encoder
        if isinstance(vision_encoder, nn.Module):
            injected_vision = vision_encoder
            vision_encoder = None
        self.vision_encoder = injected_vision or self._build_torchvision_vit(vision_encoder)
        self.vis_encoder = self.vision_encoder

        self.text_encoder = text_encoder or BertModel.from_pretrained(text_encoder_name)
        self._freeze_unused_text_pooler()
        if any(param.requires_grad for param in self.text_encoder.parameters()):
            self.text_encoder.train()
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(
            text_encoder_name,
            truncation_side="right",
        )

        vision_dim = self._vision_output_dim()
        text_dim = int(self.text_encoder.config.hidden_size)
        self.visual_projection = nn.Linear(vision_dim, self.shared_dim, bias=False)
        self.text_projection = nn.Linear(text_dim, self.shared_dim, bias=False)
        self.logit_scale = nn.Parameter(
            torch.tensor(
                float(logit_scale_init if logit_scale_init is not None else math.log(1 / 0.07))
            )
        )
        self.world_size: int | None = None
        self._lr_group_indices: dict[str, list[int]] = {}

    @staticmethod
    def _cfg_get(cfg: object | None, key: str, default: Any | None = None) -> Any:
        if cfg is None:
            return default
        if isinstance(cfg, Mapping):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _build_torchvision_vit(self, cfg: object | None) -> nn.Module:
        source = self._cfg_get(cfg, "source", "torchvision")
        architecture = self._cfg_get(cfg, "architecture", None)
        if architecture is None:
            legacy_id = str(self._cfg_get(cfg, "id", "vit_b_16:IMAGENET1K_V1"))
            architecture, _, weights_name = legacy_id.partition(":")
        else:
            weights_name = self._cfg_get(cfg, "weights", "IMAGENET1K_V1")
        if source != "torchvision" or architecture != "vit_b_16":
            raise ValueError("CLIP/SigLIP baselines require torchvision vit_b_16.")
        weights = getattr(ViT_B_16_Weights, str(weights_name))
        model = vit_b_16(weights=weights)
        model.heads = nn.Identity()
        return model

    def _vision_output_dim(self) -> int:
        for attr in ("hidden_dim", "embed_dim"):
            value = getattr(self.vision_encoder, attr, None)
            if value is not None:
                return int(value)
        return self.shared_dim

    def _freeze_unused_text_pooler(self) -> None:
        pooler = getattr(self.text_encoder, "pooler", None)
        if pooler is None:
            return
        for param in pooler.parameters():
            param.requires_grad = False

    @staticmethod
    def _caption_from_collated_pair(caption_pair: object) -> list[str]:
        if isinstance(caption_pair, tuple) and caption_pair and isinstance(caption_pair[0], str):
            return list(caption_pair)
        if isinstance(caption_pair, list) and caption_pair and isinstance(caption_pair[0], str):
            return caption_pair
        if isinstance(caption_pair, (list, tuple)) and caption_pair:
            return list(caption_pair[0])
        return list(caption_pair)  # type: ignore[arg-type]

    def _extract_batch(self, batch: Any) -> tuple[torch.Tensor, list[str]]:
        if hasattr(batch, "images") and hasattr(batch, "texts"):
            return batch.images, list(batch.texts)
        if isinstance(batch, Mapping):
            images = batch.get("images", batch.get("image"))
            texts = batch.get("texts", batch.get("caption", batch.get("text")))
            return images, list(texts)  # type: ignore[arg-type]
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise TypeError("CLIP/SigLIP training expects a batch with images and captions.")
        return batch[0], self._caption_from_collated_pair(batch[1])

    @staticmethod
    def _module_parameter_device(module: nn.Module) -> torch.device:
        """Return the device of ``module`` parameters.

        LightningModule.device can remain ``cpu`` when a checkpoint is loaded and
        moved manually outside a Trainer, which is exactly what the evaluation
        suite and checkpoint-selection code do.  Using the actual parameter
        device keeps tokenized text tensors on the same device as BERT.
        """
        try:
            return next(module.parameters()).device
        except StopIteration:  # pragma: no cover - defensive for parameter-free modules
            return torch.device("cpu")

    def _tokenize(self, texts: Iterable[str], device: torch.device) -> object:
        return self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        ).to(device)

    @staticmethod
    def _pool_vision_output(vision_outputs: torch.Tensor) -> torch.Tensor:
        if vision_outputs.ndim == 2:
            return vision_outputs
        if vision_outputs.ndim == 3:
            return vision_outputs[:, 0, :]
        raise ValueError(f"Unsupported vision output shape: {tuple(vision_outputs.shape)}")

    def encode_image_pooled(self, images: torch.Tensor) -> torch.Tensor:
        return self._pool_vision_output(self.vision_encoder(images))

    def encode_image_features(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        image_device = self._module_parameter_device(self.vision_encoder)
        if images.device != image_device:
            images = images.to(image_device, non_blocking=True)
        image_pooled = self.encode_image_pooled(images)
        image_features = self.visual_projection(image_pooled)
        return F.normalize(image_features, dim=-1) if normalize else image_features

    def encode_text_features(self, texts: Iterable[str], normalize: bool = True) -> torch.Tensor:
        text_device = self._module_parameter_device(self.text_encoder)
        tokens = self._tokenize(texts, text_device)
        text_outputs = self.text_encoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
        )
        text_pooled = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_pooled)
        return F.normalize(text_features, dim=-1) if normalize else text_features

    def _world_size(self) -> int:
        if self.world_size is not None:
            return self.world_size
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        try:
            return int(self.trainer.world_size)
        except Exception:
            return 1

    def _rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    def _gather_features(self, features: torch.Tensor) -> torch.Tensor:
        if self._world_size() <= 1 or not (dist.is_available() and dist.is_initialized()):
            return features
        if all_gather_with_grad is None:
            raise RuntimeError(
                "CLIP/SigLIP DDP training requires "
                "`torch.distributed.nn.functional.all_gather` so gathered negatives keep "
                "gradient flow. Upgrade PyTorch or run single-process training."
            )
        gathered = all_gather_with_grad(features)
        return torch.cat(list(gathered), dim=0)

    def _logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=self.logit_scale_max)

    def _compute_loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        images: torch.Tensor,
        texts: Iterable[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features = self.encode_image_features(images, normalize=True)
        text_features = self.encode_text_features(texts, normalize=True)
        loss = self._compute_loss(image_features, text_features)
        return loss, image_features, text_features

    def on_fit_start(self) -> None:
        try:
            super().on_fit_start()  # type: ignore[misc]
        except Exception:
            pass
        self.world_size = self._world_size()
        if any(param.requires_grad for param in self.text_encoder.parameters()):
            self.text_encoder.train()
        else:
            self.text_encoder.eval()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        del batch_idx
        images, texts = self._extract_batch(batch)
        loss, image_features, text_features = self(images, texts)
        self.log_dict(
            {
                "loss": loss.detach(),
                f"loss_{self.loss_name}": loss.detach(),
            },
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            prog_bar=False,
        )

        optimizer_metrics = {
            "lr_logit_scale": self._logit_scale().detach(),
        }
        logit_bias = getattr(self, "logit_bias", None)
        if logit_bias is not None:
            optimizer_metrics["logit_bias"] = logit_bias.detach()
        scheduler = self._current_scheduler()
        if scheduler is not None and hasattr(scheduler, "get_last_lr"):
            lrs = scheduler.get_last_lr()
            for key, indices in self._lr_group_indices.items():
                if indices:
                    optimizer_metrics[f"lr_{key}"] = lrs[indices[0]]
        self.log_dict(
            {f"optim/{key}": value for key, value in optimizer_metrics.items()},
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        return loss

    def _named_parameters_for_group(
        self,
        prefix: str,
        module: nn.Module,
    ) -> list[tuple[str, nn.Parameter]]:
        return [
            (f"{prefix}.{name}", param)
            for name, param in module.named_parameters()
            if param.requires_grad
        ]

    def _make_decay_groups(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        *,
        lr_schedule: str,
    ) -> list[dict[str, Any]]:
        decay = [param for name, param in named_params if not is_no_decay_parameter(name, param)]
        no_decay = [param for name, param in named_params if is_no_decay_parameter(name, param)]
        groups: list[dict[str, Any]] = []
        peak_lr = self._peak_lr(lr_schedule)
        if decay:
            groups.append({"params": decay, "lr_schedule": lr_schedule, "lr": peak_lr})
        if no_decay:
            groups.append(
                {
                    "params": no_decay,
                    "lr_schedule": lr_schedule,
                    "lr": peak_lr,
                    "WD_exclude": True,
                    "weight_decay": 0.0,
                }
            )
        return groups

    def _get_param_groups(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        groups.extend(
            self._make_decay_groups(
                self._named_parameters_for_group("vision_encoder", self.vision_encoder),
                lr_schedule="vision_encoder",
            )
        )
        groups.extend(
            self._make_decay_groups(
                self._named_parameters_for_group("text_encoder", self.text_encoder),
                lr_schedule="text_encoder",
            )
        )
        groups.extend(
            self._make_decay_groups(
                self._named_parameters_for_group("visual_projection", self.visual_projection)
                + self._named_parameters_for_group("text_projection", self.text_projection),
                lr_schedule="projection_heads",
            )
        )
        temperature_name = "temperature_bias" if self.loss_name == "siglip" else "temperature"
        groups.append(
            {
                "params": [self.logit_scale],
                "lr_schedule": temperature_name,
                "lr": self._peak_lr(temperature_name),
                "WD_exclude": True,
                "weight_decay": 0.0,
            }
        )
        logit_bias = getattr(self, "logit_bias", None)
        if logit_bias is not None:
            groups.append(
                {
                    "params": [logit_bias],
                    "lr_schedule": temperature_name,
                    "lr": self._peak_lr(temperature_name),
                    "WD_exclude": True,
                    "weight_decay": 0.0,
                }
            )
        return groups

    def _peak_lr(self, group_name: str) -> float:
        group_cfg = self._cfg_get(getattr(self.hparams, "lr_groups", None), group_name)
        if group_cfg is not None:
            return float(self._cfg_get(group_cfg, "peak_lr"))
        optimizer_lr = float(getattr(self.hparams.optimizer, "keywords", {}).get("lr", 0.0))
        lr_scheduler_vis = getattr(self.hparams, "lr_scheduler_vis", None)
        lr_scheduler_text = getattr(self.hparams, "lr_scheduler_text", None)
        if group_name == "vision_encoder" and lr_scheduler_vis is not None:
            return float(
                self._cfg_get(
                    getattr(lr_scheduler_vis, "keywords", {}),
                    "ref_lr",
                    optimizer_lr,
                )
            )
        if group_name == "text_encoder" and lr_scheduler_text is not None:
            return float(
                self._cfg_get(
                    getattr(lr_scheduler_text, "keywords", {}),
                    "ref_lr",
                    optimizer_lr,
                )
            )
        return optimizer_lr

    def _optimizer_step_schedule(self):
        schedule = compute_optimizer_step_schedule(
            trainer=self.trainer,
            ipe_scale=self.hparams.ipe_scale,
            warmup=self.hparams.warmup,
        )
        self.hparams.opt_num_training_batches = schedule.num_training_batches
        self.hparams.opt_accumulate_grad_batches = schedule.accumulate_grad_batches
        self.hparams.opt_steps_per_epoch = schedule.optimizer_steps_per_epoch
        self.hparams.opt_total_steps = schedule.total_optimizer_steps
        self.hparams.opt_warmup_steps = schedule.warmup_optimizer_steps
        self.hparams["opt/accumulate_grad_batches"] = schedule.accumulate_grad_batches
        self.hparams["opt/steps_per_epoch"] = schedule.optimizer_steps_per_epoch
        self.hparams["opt/total_steps"] = schedule.total_optimizer_steps
        self.hparams["opt/warmup_steps"] = schedule.warmup_optimizer_steps
        logger.info(
            "CLIP/SigLIP optimizer-step schedule: batches=%s accumulate_grad_batches=%s "
            "steps_per_epoch=%s total_steps=%s warmup_steps=%s warmup_epochs=%s",
            schedule.num_training_batches,
            schedule.accumulate_grad_batches,
            schedule.optimizer_steps_per_epoch,
            schedule.total_optimizer_steps,
            schedule.warmup_optimizer_steps,
            self.hparams.warmup,
        )
        return schedule

    def _final_lr(self) -> float:
        cfg = getattr(self.hparams, "lr_schedule", None)
        if cfg is not None:
            return float(self._cfg_get(cfg, "final_lr", 0.0))
        return 0.0

    @staticmethod
    def _warmup_cosine_lambda(
        *,
        warmup_steps: int,
        total_steps: int,
        final_lr: float,
        peak_lr: float,
    ):
        cosine_steps = max(1, total_steps - warmup_steps)
        final_factor = final_lr / peak_lr if peak_lr > 0 else 0.0

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(cosine_steps)
            progress = min(max(progress, 0.0), 1.0)
            return final_factor + (1.0 - final_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return lr_lambda

    def _make_grouped_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
    ) -> LambdaLR:
        lr_lambdas = []
        self._lr_group_indices = {}
        for idx, group in enumerate(optimizer.param_groups):
            schedule_name = group["lr_schedule"]
            self._lr_group_indices.setdefault(schedule_name, []).append(idx)
            lr_lambdas.append(
                self._warmup_cosine_lambda(
                    warmup_steps=warmup_steps,
                    total_steps=total_steps,
                    final_lr=self._final_lr(),
                    peak_lr=float(group["lr"]),
                )
            )
        return LambdaLR(optimizer, lr_lambdas)

    def configure_optimizers(self) -> dict[str, Any]:
        schedule = self._optimizer_step_schedule()
        total_steps = schedule.total_optimizer_steps
        warmup_steps = schedule.warmup_optimizer_steps
        optimizer = self.hparams.optimizer(params=self._get_param_groups())
        lr_scheduler = self._make_grouped_lr_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }

    def _current_scheduler(self) -> LambdaLR | None:
        try:
            scheduler = self.lr_schedulers()
        except RuntimeError:
            return None
        if isinstance(scheduler, (list, tuple)):
            return scheduler[0] if scheduler else None
        return scheduler


class CLIP(_ContrastiveBaseline):
    """Dual-encoder CLIP with symmetric softmax contrastive loss."""

    loss_name = "clip"

    def _compute_loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        all_image_features = self._gather_features(image_features)
        all_text_features = self._gather_features(text_features)
        targets = build_global_targets(
            image_features.shape[0],
            rank=self._rank(),
            device=image_features.device,
        )
        logit_scale = self._logit_scale()
        logits_i2t = logit_scale * image_features @ all_text_features.T
        logits_t2i = logit_scale * text_features @ all_image_features.T
        loss_i2t = F.cross_entropy(logits_i2t, targets)
        loss_t2i = F.cross_entropy(logits_t2i, targets)
        return 0.5 * (loss_i2t + loss_t2i)


class SigLIP(_ContrastiveBaseline):
    """Dual-encoder SigLIP with pairwise sigmoid image-text loss."""

    loss_name = "siglip"

    def __init__(
        self,
        *args: Any,
        logit_scale_init: float | None = None,
        logit_bias_init: float = -10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            logit_scale_init=(
                math.log(10.0) if logit_scale_init is None else float(logit_scale_init)
            ),
            **kwargs,
        )
        self.logit_bias = nn.Parameter(torch.tensor(float(logit_bias_init)))
        self.hparams.logit_bias_init = float(logit_bias_init)

    def _compute_loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        all_text_features = self._gather_features(text_features)
        logits = self._logit_scale() * image_features @ all_text_features.T + self.logit_bias
        labels = build_siglip_labels(
            image_features.shape[0],
            all_text_features.shape[0],
            rank=self._rank(),
            device=logits.device,
            dtype=logits.dtype,
        )
        return -F.logsigmoid(labels * logits).sum(dim=1).mean()
