from __future__ import annotations

import logging
import math
from collections import defaultdict
from copy import deepcopy
from typing import Iterable

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn

from src.evaluation.adapters import PretrainedModelAdapter

try:
    import torch.distributed as dist  # type: ignore[attr-defined]
except Exception:
    dist = None  # fallback if torch.distributed is unavailable

log = logging.getLogger(__name__)


def _official_consistency_key(identifier: str, sentence: str) -> str:
    """Match the official NLVR2 grouping by hiding the pair id."""
    parts = str(identifier).split("-")
    if len(parts) >= 4:
        parts[2] = ""
        return "-".join(parts)
    return sentence


def compute_nlvr2_metrics(
    predictions: Iterable[int],
    labels: Iterable[int],
    identifiers: Iterable[str],
    sentences: Iterable[str],
) -> dict[str, float]:
    predictions = [int(prediction) for prediction in predictions]
    labels = [int(label) for label in labels]
    identifiers = [str(identifier) for identifier in identifiers]
    sentences = [str(sentence) for sentence in sentences]

    if not labels:
        return {"acc": 0.0, "consistency": 0.0}

    correct = [prediction == label for prediction, label in zip(predictions, labels)]
    groups: dict[str, bool] = defaultdict(lambda: True)
    for is_correct, identifier, sentence in zip(correct, identifiers, sentences):
        key = _official_consistency_key(identifier, sentence)
        groups[key] = groups[key] and is_correct

    return {
        "acc": sum(correct) / len(correct),
        "consistency": sum(groups.values()) / len(groups) if groups else 0.0,
    }


class NLVR2InteractionHead(nn.Module):
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 1536,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = feature_dim * 8
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [
                left,
                right,
                text,
                left * text,
                right * text,
                (left - text).abs(),
                (right - text).abs(),
                (left - right).abs(),
            ],
            dim=-1,
        )
        return self.net(features)


class NLVR2TokenInteractionHead(nn.Module):
    """Lightweight token-level interaction probe for frozen NLVR2 evaluation."""

    def __init__(
        self,
        feature_dim: int = 768,
        fusion_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        classifier_hidden_dim: int = 512,
        image_token_pool: int = 2,
        max_position_embeddings: int = 512,
    ) -> None:
        super().__init__()
        self.image_token_pool = max(1, int(image_token_pool))
        self.max_position_embeddings = int(max_position_embeddings)
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_projection = nn.Linear(feature_dim, fusion_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        self.type_embeddings = nn.Embedding(4, fusion_dim)
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, fusion_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=max(fusion_dim, int(fusion_dim * float(mlp_ratio))),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 2),
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.type_embeddings.weight, std=0.02)
        nn.init.trunc_normal_(self.position_embeddings.weight, std=0.02)

    def _pool_image_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.image_token_pool <= 1:
            return tokens
        batch_size, seq_len, dim = tokens.shape
        side = int(math.sqrt(seq_len))
        if side * side == seq_len:
            grid = tokens.reshape(batch_size, side, side, dim).permute(0, 3, 1, 2)
            pooled = F.avg_pool2d(
                grid,
                kernel_size=self.image_token_pool,
                stride=self.image_token_pool,
            )
            return pooled.permute(0, 2, 3, 1).reshape(batch_size, -1, dim)
        return tokens[:, :: self.image_token_pool, :]

    def _embed_tokens(self, tokens: torch.Tensor, type_id: int) -> torch.Tensor:
        if tokens.shape[1] > self.max_position_embeddings:
            raise ValueError(
                "NLVR2 token-interaction sequence segment has "
                f"{tokens.shape[1]} tokens, but max_position_embeddings="
                f"{self.max_position_embeddings}."
            )
        batch_size, seq_len = tokens.shape[:2]
        projected = self.input_projection(self.input_norm(tokens))
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.long)
        positions = self.position_embeddings(positions).unsqueeze(0)
        types = torch.full(
            (batch_size, seq_len),
            int(type_id),
            device=tokens.device,
            dtype=torch.long,
        )
        return projected + positions + self.type_embeddings(types)

    def forward(
        self,
        left_tokens: torch.Tensor,
        right_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        left_tokens = self._pool_image_tokens(left_tokens)
        right_tokens = self._pool_image_tokens(right_tokens)
        left = self._embed_tokens(left_tokens, type_id=1)
        right = self._embed_tokens(right_tokens, type_id=2)
        text = self._embed_tokens(text_tokens, type_id=3)

        batch_size = left.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        cls = cls + self.type_embeddings.weight[0].view(1, 1, -1)
        cls = cls + self.position_embeddings.weight[0].view(1, 1, -1)
        tokens = torch.cat([cls, left, right, text], dim=1)

        valid = torch.cat(
            [
                torch.ones(batch_size, 1, device=tokens.device, dtype=torch.bool),
                torch.ones(batch_size, left.shape[1], device=tokens.device, dtype=torch.bool),
                torch.ones(batch_size, right.shape[1], device=tokens.device, dtype=torch.bool),
                text_mask.to(device=tokens.device, dtype=torch.bool),
            ],
            dim=1,
        )
        fused = self.encoder(tokens, src_key_padding_mask=~valid)
        return self.classifier(fused[:, 0])


class NLVR2Evaluator(LightningModule):
    """Frozen-probe evaluator for NLVR2 binary visual reasoning."""

    _MODE_ALIASES = {
        "frozen_probe": "global_projected",
        "global_projected": "global_projected",
        "global_raw": "global_raw",
        "token_interaction": "token_interaction",
        "token_interaction_probe": "token_interaction",
    }

    def __init__(
        self,
        ckpt_path: str,
        mode: str = "frozen_probe",
        freeze_backbone: bool = True,
        bert_size: str = "auto",
        max_text_len: int = 64,
        feature_dim: int = 768,
        hidden_dim: int = 1536,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        metric_prefix: str = "",
        load_best_val_before_test: bool = True,
        encoder_source: str | None = None,
        feature_projection: str | None = None,
        token_interaction_dim: int = 256,
        token_interaction_layers: int = 2,
        token_interaction_heads: int = 4,
        token_interaction_mlp_ratio: float = 2.0,
        token_interaction_classifier_hidden_dim: int = 512,
        token_interaction_image_pool: int = 2,
        token_interaction_max_position_embeddings: int = 512,
        adapter: PretrainedModelAdapter | None = None,
    ) -> None:
        super().__init__()
        self.probe_mode = self._resolve_mode(mode)
        if feature_projection is None:
            feature_projection = (
                "raw" if self.probe_mode in {"global_raw", "token_interaction"} else "projected"
            )
        feature_projection = PretrainedModelAdapter._validate_feature_projection(
            feature_projection
        )
        if self.probe_mode == "global_raw" and feature_projection != "raw":
            raise ValueError("`mode='global_raw'` requires `feature_projection='raw'`.")
        self.save_hyperparameters(ignore=["adapter"], logger=False)
        self.adapter = adapter or PretrainedModelAdapter(
            ckpt_path=ckpt_path,
            bert_size=bert_size,
            max_text_len=max_text_len,
            encoder_source=encoder_source,
            feature_projection=feature_projection,
        )
        if freeze_backbone:
            parameters = getattr(self.adapter, "parameters", lambda: [])()
            for param in parameters:
                param.requires_grad = False
        if self.probe_mode == "token_interaction":
            self.classifier = NLVR2TokenInteractionHead(
                feature_dim=feature_dim,
                fusion_dim=token_interaction_dim,
                num_layers=token_interaction_layers,
                num_heads=token_interaction_heads,
                mlp_ratio=token_interaction_mlp_ratio,
                dropout=dropout,
                classifier_hidden_dim=token_interaction_classifier_hidden_dim,
                image_token_pool=token_interaction_image_pool,
                max_position_embeddings=token_interaction_max_position_embeddings,
            )
        else:
            self.classifier = NLVR2InteractionHead(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        self._split_outputs: dict[str, dict[str, list | float | int]] = {}
        self._best_val_acc = float("-inf")
        self._best_classifier_state: dict[str, torch.Tensor] | None = None

    @classmethod
    def _resolve_mode(cls, mode: str) -> str:
        resolved = cls._MODE_ALIASES.get(str(mode))
        if resolved is None:
            supported = ", ".join(sorted(cls._MODE_ALIASES))
            raise ValueError(
                f"Unsupported NLVR2 evaluation mode {mode!r}. Use one of: {supported}."
            )
        return resolved

    def _metric_key(self, split: str, metric: str) -> str:
        if split == "train":
            return f"train/{metric}"
        key = f"{split}/{metric}"
        prefix = str(self.hparams.metric_prefix or "").strip("/")
        return f"{prefix}/{key}" if prefix else key

    def _encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left = self.adapter.encode_images(batch["image_left"], normalize=True)
        right = self.adapter.encode_images(batch["image_right"], normalize=True)
        text = self.adapter.encode_texts(list(batch["sentence"]), normalize=True)
        return left, right, text

    def _token_logits(self, batch: dict) -> torch.Tensor:
        project = str(self.hparams.feature_projection or "raw").lower() == "projected"
        left = self.adapter.encode_image_sequence(
            batch["image_left"],
            normalize=False,
            project=project,
        )
        right = self.adapter.encode_image_sequence(
            batch["image_right"],
            normalize=False,
            project=project,
        )
        text, text_mask = self.adapter.encode_text_sequence(
            list(batch["sentence"]),
            normalize=False,
            project=project,
        )
        return self.classifier(left, right, text, text_mask)

    def _logits(self, batch: dict) -> torch.Tensor:
        if self.probe_mode == "token_interaction":
            return self._token_logits(batch)
        left, right, text = self._encode(batch)
        return self.classifier(left, right, text)

    def _shared_step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = batch["label"].long().to(self.device)
        logits = self._logits(batch)
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=-1)
        return loss, predictions, labels

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, predictions, labels = self._shared_step(batch)
        acc = (predictions == labels).float().mean()
        self.log(
            self._metric_key("train", "loss"),
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=False,
        )
        self.log(
            self._metric_key("train", "acc"),
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=False,
        )
        return loss

    def _reset_split(self, split: str) -> None:
        self._split_outputs[split] = {
            "predictions": [],
            "labels": [],
            "identifiers": [],
            "sentences": [],
            "loss_sum": 0.0,
            "count": 0,
        }

    def _eval_step(self, batch: dict, split: str) -> None:
        loss, predictions, labels = self._shared_step(batch)
        output = self._split_outputs[split]
        count = int(labels.numel())
        output["loss_sum"] = float(output["loss_sum"]) + float(loss.detach().cpu()) * count
        output["count"] = int(output["count"]) + count
        output["predictions"].extend(predictions.detach().cpu().tolist())
        output["labels"].extend(labels.detach().cpu().tolist())
        output["identifiers"].extend(str(identifier) for identifier in batch["identifier"])
        output["sentences"].extend(str(sentence) for sentence in batch["sentence"])

    def _finish_split(self, split: str) -> dict[str, float]:
        """
        Finalise the accumulated metrics for the given split.  In
        distributed mode the outputs are gathered across ranks before
        computing the final accuracy and consistency metrics.  This
        ensures that results are based on the full evaluation set
        rather than each rank's partial shard.  Metrics are computed
        and logged on rank zero and then broadcast to other ranks via
        `sync_dist=True`.
        """
        output = self._split_outputs.get(split)
        if output is None:
            return {}
        # Gather outputs across ranks if distributed is initialised
        gathered_outputs = [output]
        world_size = 1
        rank = 0
        if dist is not None and dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            gathered_list: list[dict] = [None for _ in range(world_size)]  # type: ignore[var-annotated]
            dist.all_gather_object(gathered_list, output)  # type: ignore[call-arg]
            gathered_outputs = gathered_list
        # Merge the gathered outputs on rank zero
        if rank == 0:
            merged = {
                "predictions": [],
                "labels": [],
                "identifiers": [],
                "sentences": [],
                "loss_sum": 0.0,
                "count": 0,
            }
            for out in gathered_outputs:
                merged["predictions"].extend(out["predictions"])
                merged["labels"].extend(out["labels"])
                merged["identifiers"].extend(out["identifiers"])
                merged["sentences"].extend(out["sentences"])
                merged["loss_sum"] += float(out["loss_sum"])
                merged["count"] += int(out["count"])
            count = merged["count"]
            metrics = compute_nlvr2_metrics(
                predictions=merged["predictions"],
                labels=merged["labels"],
                identifiers=merged["identifiers"],
                sentences=merged["sentences"],
            )
            metrics["loss"] = float(merged["loss_sum"]) / count if count else 0.0
            if split == "val":
                if metrics["acc"] > self._best_val_acc:
                    self._best_val_acc = metrics["acc"]
                    self._best_classifier_state = deepcopy(
                        {
                            key: value.detach().cpu()
                            for key, value in self.classifier.state_dict().items()
                        }
                    )
                metrics["best_acc"] = self._best_val_acc
        else:
            # Create a dummy metrics dict to be synced; the values will be overwritten
            metrics = {"acc": 0.0, "consistency": 0.0, "loss": 0.0}
            if split == "val":
                metrics["best_acc"] = 0.0
        prefixed = {self._metric_key(split, key): value for key, value in metrics.items()}
        # Log with sync_dist=True to broadcast metrics to all ranks and avoid double logging
        self.log_dict(prefixed, sync_dist=True, logger=False, on_epoch=True)
        if rank == 0:
            for key, value in prefixed.items():
                log.info("%s: %.4f", key, value)
        self._reset_split(split)
        return prefixed

    def on_validation_epoch_start(self) -> None:
        self._reset_split("val")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._eval_step(batch, split="val")

    def on_validation_epoch_end(self) -> None:
        self._finish_split("val")

    def on_test_epoch_start(self) -> None:
        if self.hparams.load_best_val_before_test and self._best_classifier_state is not None:
            self.classifier.load_state_dict(self._best_classifier_state)
        self._reset_split("test")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self._eval_step(batch, split="test")

    def on_test_epoch_end(self) -> None:
        self._finish_split("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.trainer.max_epochs),
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
