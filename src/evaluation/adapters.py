from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from src.evaluation.checkpoint import load_model_from_run
from src.models.xjepa import _image_global, _image_patch_tokens

log = logging.getLogger(__name__)


class UnsupportedEvaluationModelError(TypeError):
    """Raised when an evaluator cannot infer an embedding interface."""


class PretrainedModelAdapter(nn.Module):
    """Expose a stable embedding interface for retained checkpoints."""

    def __init__(
        self,
        ckpt_path: str,
        bert_size: str = "bert-base-uncased",
        max_text_len: int = 64,
        strict: bool = True,
        encoder_source: str | None = None,
        feature_projection: str = "projected",
    ) -> None:
        super().__init__()
        if not ckpt_path:
            raise ValueError(
                "Evaluation requires `ckpt_path` to point to a pretraining run or checkpoint."
            )

        self.model, self.run_cfg, self.resolved_ckpt = load_model_from_run(
            ckpt_path, strict=strict
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.bert_size = self._resolve_bert_size(bert_size)
        self.requested_max_text_len = int(max_text_len)
        self.max_text_len = self.requested_max_text_len
        self.encoder_source = encoder_source
        self.feature_projection = self._validate_feature_projection(feature_projection)
        self._bert_tokenizer = None
        self.adapter = self._infer_adapter()
        self.max_text_len = self._effective_max_text_len(self.requested_max_text_len)
        if self.max_text_len < self.requested_max_text_len:
            log.warning(
                "Capping %s adapter max_text_len from %s to %s to fit the checkpoint "
                "joint encoder positional capacity.",
                self.adapter,
                self.requested_max_text_len,
                self.max_text_len,
            )
        # This wrapper is used as a frozen feature extractor.  Keep it in eval
        # mode even when an outer LightningModule is training a lightweight
        # adapter/probe head; otherwise BERT/ViT dropout would be active during
        # frozen-backbone evaluation.
        self.train(False)
        log.info("Using %s evaluation adapter for %s", self.adapter, self.resolved_ckpt)

    def train(self, mode: bool = True):
        """Keep the wrapped pretrained model deterministic during probe training.

        The evaluation adapter is a feature extractor, not the trainable part of
        adapter/probe protocols.  PyTorch/Lightning recursively calls
        ``train(True)`` on child modules during ``fit``; without this override,
        the frozen pretrained checkpoint would switch to train mode and enable
        dropout even though all parameters have ``requires_grad=False``.
        """

        super().train(False)
        self.model.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _infer_adapter(self) -> str:
        if (
            hasattr(self.model, "prediction_proj")
            and hasattr(self.model, "target_proj")
            and hasattr(self.model, "logit_scale")
        ):
            return "target_contrastive"
        if hasattr(self.model, "target_vis_encoder") and hasattr(
            self.model, "target_text_encoder"
        ):
            return "target_dual"
        if (
            hasattr(self.model, "vis_encoder")
            and hasattr(self.model, "text_encoder")
            and hasattr(self.model, "global_proj_vis")
            and hasattr(self.model, "global_proj_text")
        ):
            return "online_dual"
        if (
            hasattr(self.model, "vis_encoder")
            and hasattr(self.model, "text_encoder")
            and hasattr(self.model, "visual_projection")
            and hasattr(self.model, "text_projection")
        ):
            loss_name = getattr(getattr(self.model, "hparams", None), "loss_name", None)
            loss_name = loss_name or getattr(self.model, "loss_name", None)
            if str(loss_name).lower() == "siglip" or type(self.model).__name__.lower() == "siglip":
                return "siglip"
            return "clip"
        raise UnsupportedEvaluationModelError(
            f"Unsupported checkpoint model type for evaluation: {type(self.model).__name__}"
        )

    @staticmethod
    def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
        if config is None:
            return default
        if isinstance(config, Mapping):
            return config.get(key, default)
        try:
            if key in config:
                return config[key]
        except (AttributeError, KeyError, TypeError):
            pass
        return getattr(config, key, default)

    @staticmethod
    def _torchvision_vit_patch_tokens(
        encoder: nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor | None:
        if not all(
            hasattr(encoder, name) for name in ("_process_input", "class_token", "encoder")
        ):
            return None
        batch_size = images.shape[0]
        tokens = encoder._process_input(images)
        class_token = encoder.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat([class_token, tokens], dim=1)
        tokens = encoder.encoder(tokens)
        if tokens.ndim != 3:
            raise ValueError(
                f"Expected torchvision ViT token output with 3 dims, got {tokens.shape}."
            )
        return tokens[:, 1:]

    def _resolve_text_encoder_name(self) -> str | None:
        for name in ("target_text_encoder", "text_encoder"):
            text_encoder = getattr(self.model, name, None)
            config = getattr(text_encoder, "config", None)
            if config is None:
                continue
            model_name = (
                getattr(config, "name_or_path", None)
                or getattr(config, "_name_or_path", None)
                or self._cfg_get(config, "name_or_path")
                or self._cfg_get(config, "_name_or_path")
            )
            if model_name:
                return model_name
        return None

    def _resolve_bert_size(self, bert_size: str | None) -> str:
        if bert_size and bert_size != "auto":
            return bert_size

        model_cfg = self._cfg_get(self.run_cfg, "model", {})
        text_encoder_cfg = self._cfg_get(model_cfg, "text_encoder", {})
        text_encoder_size = self._cfg_get(text_encoder_cfg, "size")
        if text_encoder_size:
            return text_encoder_size

        data_cfg = self._cfg_get(self.run_cfg, "data", {})
        collate_cfg = self._cfg_get(data_cfg, "collate_fn", {})
        tokenizer_cfg = self._cfg_get(collate_cfg, "tokenizer", {})
        tokenizer_name = self._cfg_get(tokenizer_cfg, "pretrained_model_name_or_path")
        if tokenizer_name:
            return tokenizer_name

        model_name = self._resolve_text_encoder_name()
        if model_name:
            return model_name

        return "bert-base-uncased"

    @staticmethod
    def _validate_feature_projection(value: str | None) -> str:
        projection = str(value or "projected").lower()
        supported = {"projected", "raw"}
        if projection not in supported:
            raise ValueError(
                f"Unsupported feature_projection={value!r}. Use one of: {sorted(supported)}."
            )
        return projection

    def _use_feature_projection(self) -> bool:
        return self.feature_projection == "projected"

    def _effective_max_text_len(self, requested_max_text_len: int) -> int:
        return int(requested_max_text_len)

    def _ensure_bert_tokenizer(self):
        if self._bert_tokenizer is None:
            from transformers import BertTokenizer

            tokenizer = BertTokenizer.from_pretrained(self.bert_size, truncation_side="right")
            tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            self._bert_tokenizer = tokenizer
        return self._bert_tokenizer

    def _tokenize_bert(self, texts: Iterable[str]):
        tokenizer = self._ensure_bert_tokenizer()
        return tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        ).to(self.device)

    def _pool_sequence(self, output, pool: str = "mean") -> torch.Tensor:
        if hasattr(output, "last_hidden_state"):
            output = output.last_hidden_state
        if output.ndim == 2:
            return output
        if output.ndim == 4:
            output = output.flatten(2).transpose(1, 2)
        if pool == "cls":
            return output[:, 0, :]
        return output.mean(dim=1)

    def _maybe_normalize(self, features: torch.Tensor, normalize: bool) -> torch.Tensor:
        return F.normalize(features, dim=-1) if normalize else features

    def _project(self, features: torch.Tensor, *names: str) -> torch.Tensor:
        for name in names:
            projector = getattr(self.model, name, None)
            if projector is not None:
                return projector(features)
        return features

    def _project_sequence(self, features: torch.Tensor, *names: str) -> torch.Tensor:
        for name in names:
            projector = getattr(self.model, name, None)
            if projector is None:
                continue
            batch_size, seq_len, dim = features.shape
            projected = projector(features.reshape(batch_size * seq_len, dim))
            return projected.reshape(batch_size, seq_len, -1)
        return features

    def _target_or_online_image_encoder(self) -> nn.Module:
        encoder_source = getattr(self, "encoder_source", None)
        if self.adapter == "target_dual":
            if encoder_source == "online" and hasattr(self.model, "vis_encoder"):
                return self.model.vis_encoder
            return self.model.target_vis_encoder
        if self.adapter == "online_dual":
            if encoder_source == "target" and hasattr(self.model, "target_vis_encoder"):
                return self.model.target_vis_encoder
            return self.model.vis_encoder
        if self.adapter == "target_contrastive":
            return self.model.vis_encoder
        raise UnsupportedEvaluationModelError(f"No dual image encoder for adapter {self.adapter}")

    def _target_or_online_text_encoder(self) -> nn.Module:
        encoder_source = getattr(self, "encoder_source", None)
        if self.adapter == "target_dual":
            if encoder_source == "online" and hasattr(self.model, "text_encoder"):
                return self.model.text_encoder
            return self.model.target_text_encoder
        if self.adapter == "online_dual":
            if encoder_source == "target" and hasattr(self.model, "target_text_encoder"):
                return self.model.target_text_encoder
            return self.model.text_encoder
        if self.adapter == "target_contrastive":
            return self.model.text_encoder
        raise UnsupportedEvaluationModelError(f"No dual text encoder for adapter {self.adapter}")

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        images = images.to(self.device)

        if self.adapter == "target_dual":
            features = _image_global(self._target_or_online_image_encoder(), images)
            if self._use_feature_projection():
                features = self._project(features, "global_proj_vis", "target_vis_proj")
            return self._maybe_normalize(features, normalize)

        if self.adapter == "online_dual":
            features = _image_global(self._target_or_online_image_encoder(), images)
            if self._use_feature_projection():
                features = self.model.global_proj_vis(features)
            return self._maybe_normalize(features, normalize)

        if self.adapter in {"clip", "siglip"}:
            if self._use_feature_projection() and hasattr(self.model, "encode_image_features"):
                return self.model.encode_image_features(images, normalize=normalize)
            features = self._pool_sequence(self.model.vis_encoder(images))
            if self._use_feature_projection():
                features = self.model.visual_projection(features)
            return self._maybe_normalize(features, normalize)

        if self.adapter == "target_contrastive":
            if self._use_feature_projection() and hasattr(self.model, "encode_image_features"):
                return self.model.encode_image_features(
                    images,
                    normalize=normalize,
                    target_seq_len=self.max_text_len,
                )
            features = _image_global(self.model.vis_encoder, images)
            return self._maybe_normalize(features, normalize)

        raise UnsupportedEvaluationModelError(f"No image encoder for adapter {self.adapter}")

    @torch.no_grad()
    def encode_texts(self, texts: Iterable[str], normalize: bool = True) -> torch.Tensor:
        text_list = list(texts)
        if (
            self.adapter in {"clip", "siglip"}
            and self._use_feature_projection()
            and hasattr(self.model, "encode_text_features")
        ):
            return self.model.encode_text_features(text_list, normalize=normalize)

        if (
            self.adapter == "target_contrastive"
            and self._use_feature_projection()
            and hasattr(self.model, "encode_text_features_from_tokens")
        ):
            tokens = self._tokenize_bert(text_list)
            return self.model.encode_text_features_from_tokens(
                tokens.input_ids, tokens.attention_mask, normalize=normalize
            )

        if self.adapter not in {"target_dual", "online_dual", "clip", "siglip", "target_contrastive"}:
            raise UnsupportedEvaluationModelError(f"No text encoder for adapter {self.adapter}")

        tokens = self._tokenize_bert(text_list)
        text_encoder = (
            self._target_or_online_text_encoder()
            if self.adapter in {"target_dual", "online_dual", "target_contrastive"}
            else self.model.text_encoder
        )
        encoded = text_encoder(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)
        features = self._pool_sequence(encoded, pool="cls")
        if self.adapter == "target_dual":
            if self._use_feature_projection():
                features = self._project(features, "global_proj_text", "target_text_proj")
        elif self.adapter == "online_dual":
            if self._use_feature_projection():
                features = self.model.global_proj_text(features)
        elif self.adapter in {"clip", "siglip"}:
            if self._use_feature_projection():
                features = self.model.text_projection(features)
        elif self._use_feature_projection():
            features = self.model.target_proj(features)
        return self._maybe_normalize(features, normalize)

    @torch.no_grad()
    def encode_text_sequence(
        self,
        texts: Iterable[str],
        normalize: bool = True,
        project: bool = True,
        return_tokens: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[list[str]]]:
        """Encode text as projected token-level embeddings.

        This is intended for qualitative token-patch affinity maps.  It uses
        the same BERT-like text path as ``encode_texts`` and returns the
        attention mask so callers can ignore padding/special tokens.
        """

        text_list = list(texts)
        if self.adapter not in {
            "target_dual",
            "online_dual",
            "clip",
            "siglip",
            "target_contrastive",
        }:
            raise UnsupportedEvaluationModelError(
                f"Token-level text encoding is not supported for adapter {self.adapter}."
            )

        tokens = self._tokenize_bert(text_list)
        text_encoder = (
            self._target_or_online_text_encoder()
            if self.adapter in {"target_dual", "online_dual", "target_contrastive"}
            else getattr(self.model, "text_encoder", None)
        )
        if text_encoder is None:
            raise UnsupportedEvaluationModelError(
                f"Model {type(self.model).__name__} does not expose a token-level text encoder."
            )

        encoded = text_encoder(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)
        if not hasattr(encoded, "last_hidden_state"):
            raise UnsupportedEvaluationModelError(
                f"Text encoder for {type(self.model).__name__} did not return token states."
            )

        features = encoded.last_hidden_state
        if project:
            if self.adapter == "target_dual":
                features = self._project_sequence(features, "global_proj_text", "target_text_proj")
            elif self.adapter == "online_dual":
                features = self._project_sequence(features, "global_proj_text")
            elif self.adapter in {"clip", "siglip"}:
                features = self._project_sequence(features, "text_projection")
            else:
                features = self._project_sequence(features, "target_proj")
        features = self._maybe_normalize(features, normalize)
        attention_mask = tokens.attention_mask.bool()

        if return_tokens:
            tokenizer = self._ensure_bert_tokenizer()
            decoded_tokens = [
                tokenizer.convert_ids_to_tokens(row.detach().cpu().tolist())
                for row in tokens.input_ids
            ]
            return features, attention_mask, decoded_tokens
        return features, attention_mask

    @torch.no_grad()
    def encode_image_sequence(
        self,
        images: torch.Tensor,
        normalize: bool = True,
        project: bool = True,
    ) -> torch.Tensor:
        images = images.to(self.device)

        if self.adapter == "target_dual":
            features = _image_patch_tokens(self._target_or_online_image_encoder(), images)
            if project:
                features = self._project_sequence(features, "global_proj_vis", "target_vis_proj")
            return self._maybe_normalize(features, normalize)

        if self.adapter == "online_dual":
            features = _image_patch_tokens(self._target_or_online_image_encoder(), images)
            if project:
                features = self._project_sequence(features, "global_proj_vis")
            return self._maybe_normalize(features, normalize)

        if self.adapter in {"clip", "siglip"}:
            try:
                features = _image_patch_tokens(self.model.vis_encoder, images)
            except (AttributeError, TypeError, ValueError):
                features = self._torchvision_vit_patch_tokens(self.model.vis_encoder, images)
            if features is None:
                features = self.model.vis_encoder(images)
                if features.ndim == 2:
                    raise UnsupportedEvaluationModelError(
                        "CLIP/SigLIP token-sequence evaluation requires a vision encoder "
                        "that exposes patch tokens or torchvision ViT internals."
                    )
            if project:
                features = self.model.visual_projection(features)
            return self._maybe_normalize(features, normalize)

        if self.adapter == "target_contrastive":
            features = _image_patch_tokens(self.model.vis_encoder, images)
            return self._maybe_normalize(features, normalize)

        raise UnsupportedEvaluationModelError(
            f"No image sequence encoder for adapter {self.adapter}"
        )

    @torch.no_grad()
    def score_image_text_pairs(
        self,
        images: torch.Tensor,
        texts: Iterable[str],
        score: str = "itc",
    ) -> torch.Tensor:
        image_features = self.encode_images(images, normalize=True)
        text_features = self.encode_texts(texts, normalize=True)
        return (image_features * text_features).sum(dim=-1)
