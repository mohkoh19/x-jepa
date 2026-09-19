"""X-JEPA: joint-embedding predictive vision-language models.

The module implements the three variants that are compared in the paper:

``XJEPA_P``
    ``[P]`` - bidirectional latent cross-modal prediction with exponential
    moving-average target encoders and an MSE prediction loss.
``XJEPA_PA``
    ``[P,A]`` - the prediction objective plus direct global image-text
    alignment, ``L = L_[P] + lambda * L_[A]``.
``XJEPA_TC``
    ``[TC]`` - image-to-text predictor/target InfoNCE without EMA targets.

``XJEPA`` holds the shared encoder envelope (ViT-B/16 image encoder,
BERT-base text encoder and a shared cross-modal predictor) together with the
``[P]`` objective.  The paper-named subclasses pin the paper configuration and
add the variant-specific objective term.

Checkpoint compatibility
------------------------
The released checkpoints were trained with earlier attribute names.  Their
state dictionaries are migrated on load (see ``LEGACY_STATE_DICT_KEY_MAP``),
so the public class/attribute names can stay descriptive without invalidating
the published weights.
"""

from __future__ import annotations

import copy
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from src.models.base import BaseModule
from src.models.components.projection import GlobalProjectionHead
from src.models.components.vision_pretrained import load_vision_pretrained
from src.models.components.vision_transformer import (
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    make_transformer_block,
)
from src.models.losses import SymmetricInfoNCELoss
from src.utils.dist import FullGatherLayer
from src.utils.ijepa.tensors import init_weights, trunc_normal_

logger = logging.getLogger(__name__)

VALID_PREDICTION_DIRECTIONS = {"both", "i2t", "t2i"}

#: BERT-base-uncased ``[PAD]``, ``[UNK]``, ``[CLS]`` and ``[SEP]`` token ids.
DEFAULT_TEXT_SPECIAL_TOKEN_IDS = (0, 100, 101, 102)

#: Attributes that were renamed before the public release.  The mapping is
#: applied to state dictionaries so that published checkpoints keep loading.
LEGACY_STATE_DICT_KEY_MAP = {
    "vicreg_proj_vis": "global_proj_vis",
    "vicreg_proj_text": "global_proj_text",
    "vljepa_pred_proj": "prediction_proj",
    "vljepa_target_proj": "target_proj",
    "vljepa_logit_scale": "logit_scale",
    "vljepa_info_nce_loss": "info_nce_loss",
}


def _validate_prediction_directions(prediction_directions: str) -> str:
    directions = str(prediction_directions).lower()
    if directions not in VALID_PREDICTION_DIRECTIONS:
        supported = ", ".join(sorted(VALID_PREDICTION_DIRECTIONS))
        raise ValueError(
            f"Unsupported prediction_directions={prediction_directions!r}. Use one of: {supported}."
        )
    return directions



def _normalize_text_special_token_ids(token_ids) -> tuple[int, ...]:
    if token_ids is None:
        return ()
    return tuple(int(token_id) for token_id in token_ids)


def _text_target_valid_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_token_ids: tuple[int, ...],
) -> torch.Tensor:
    valid = attention_mask.to(dtype=torch.bool, device=input_ids.device)
    for token_id in special_token_ids:
        valid = valid & (input_ids != token_id)
    return valid


def _all_valid_mask(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)


def _masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error over valid target tokens and hidden dimensions."""
    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes must match, got {pred.shape} and {target.shape}."
        )
    if pred.numel() == 0:
        return pred.sum() * 0.0

    squared_error = (pred - target).pow(2)
    if valid_mask is None:
        return squared_error.mean()

    valid_mask = valid_mask.to(device=pred.device, dtype=torch.bool)
    while valid_mask.ndim < squared_error.ndim:
        valid_mask = valid_mask.unsqueeze(-1)
    valid_mask = valid_mask.expand_as(squared_error)

    if not valid_mask.any():
        return squared_error.sum() * 0.0
    return squared_error.masked_select(valid_mask).mean()


def _masked_mean_pool(
    tokens: torch.Tensor, valid_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Masked mean pooling over a token sequence."""
    if valid_mask is None:
        return tokens.mean(dim=1)

    mask_expanded = valid_mask.unsqueeze(-1).float()
    weighted_sum = (tokens * mask_expanded).sum(dim=1)
    valid_count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return weighted_sum / valid_count


def _directional_prediction_loss(
    module: nn.Module,
    pred_text_from_image: torch.Tensor,
    pred_image_from_text: torch.Tensor,
    target_image: torch.Tensor,
    target_text: torch.Tensor,
    text_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average the active cross-modal prediction directions.

    Returns ``(loss_image_to_text, loss_text_to_image, loss)``.  Directions that
    are inactive contribute a detached zero so that every returned value stays
    safe to log.
    """
    raw_loss_i2t = _masked_mse(pred_text_from_image, target_text, valid_mask=text_valid_mask)
    raw_loss_t2i = _masked_mse(pred_image_from_text, target_image)

    directions = getattr(module, "prediction_directions", "both")
    active_i2t = directions in {"both", "i2t"}
    active_t2i = directions in {"both", "t2i"}
    loss_i2t = raw_loss_i2t if active_i2t else raw_loss_i2t.detach() * 0.0
    loss_t2i = raw_loss_t2i if active_t2i else raw_loss_t2i.detach() * 0.0
    denominator = float(active_i2t) + float(active_t2i)
    return loss_i2t, loss_t2i, (loss_i2t + loss_t2i) / denominator


def _image_features(encoder: nn.Module, images: torch.Tensor):
    """Run a vision encoder and return its raw output object."""
    if hasattr(encoder, "forward_features"):
        return encoder.forward_features(images)
    return encoder(images)


def _image_patch_tokens_from_output(output) -> torch.Tensor:
    if hasattr(output, "patch_tokens"):
        return output.patch_tokens
    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    if output.ndim == 4:
        return output.flatten(2).transpose(1, 2)
    if output.ndim != 3:
        raise ValueError(f"Expected image patch-token output with 3 dims, got {output.shape}.")
    return output


def _image_global_from_output(
    encoder: nn.Module,
    output,
    *,
    images: torch.Tensor | None = None,
) -> torch.Tensor:
    if hasattr(output, "cls"):
        return output.cls
    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    if output.ndim == 2:
        return output
    if output.ndim == 4:
        output = output.flatten(2).transpose(1, 2)
    if output.ndim == 3:
        if hasattr(encoder, "encode_global"):
            if images is None:
                raise ValueError(
                    "Torchvision-backed image globals must be produced by encode_global "
                    "or forward_features, not by mean-pooling patch tokens."
                )
            return encoder.encode_global(images)
        return output.mean(dim=1)
    raise ValueError(f"Unsupported image output shape for global pooling: {output.shape}.")


def _image_patch_tokens(encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
    return _image_patch_tokens_from_output(_image_features(encoder, images))


def _image_global(
    encoder: nn.Module,
    images: torch.Tensor | None = None,
    *,
    encoded=None,
) -> torch.Tensor:
    if encoded is not None:
        return _image_global_from_output(encoder, encoded, images=images)
    if images is None:
        raise ValueError("images are required when encoded image features are not provided.")
    if hasattr(encoder, "encode_global"):
        return encoder.encode_global(images)
    return _image_global_from_output(encoder, encoder(images), images=images)


class SharedCrossModalPredictor(nn.Module):
    """Shared bidirectional predictor with modality-specific adapters.

    The same predictor weights are used for image-to-text and text-to-image
    prediction; only the input/output adapters and the position/modality
    embeddings depend on the direction.
    """

    def __init__(
        self,
        *,
        num_patches: int,
        vis_dim: int,
        text_dim: int,
        predictor_embed_dim: int,
        depth: int,
        num_heads: int,
        max_text_len: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        init_std: float = 0.02,
        block_impl: str = "sdpa",
    ):
        super().__init__()
        self.predictor_dim = predictor_embed_dim
        self.num_patches = int(num_patches)
        self.max_text_len = int(max_text_len)
        self.depth = depth
        self.num_heads = num_heads
        self.block_impl = block_impl
        self.input_proj_vis = nn.Linear(vis_dim, predictor_embed_dim)
        self.input_proj_text = nn.Linear(text_dim, predictor_embed_dim)
        self.output_proj_vis = nn.Linear(predictor_embed_dim, vis_dim)
        self.output_proj_text = nn.Linear(predictor_embed_dim, text_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.modality_embed = nn.ParameterDict(
            {
                "vis": nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)),
                "text": nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)),
            }
        )
        self.pos_embeds = nn.ParameterDict(
            {
                "vis": nn.Parameter(
                    torch.zeros(1, num_patches, predictor_embed_dim),
                    requires_grad=False,
                ),
                "text": nn.Parameter(
                    torch.zeros(1, max_text_len, predictor_embed_dim),
                    requires_grad=False,
                ),
            }
        )
        vis_pos = get_2d_sincos_pos_embed(
            predictor_embed_dim,
            int(num_patches**0.5),
            cls_token=False,
        )
        text_pos = get_1d_sincos_pos_embed(predictor_embed_dim, max_text_len, cls_token=False)
        self.pos_embeds["vis"].data.copy_(torch.from_numpy(vis_pos).float().unsqueeze(0))
        self.pos_embeds["text"].data.copy_(torch.from_numpy(text_pos).float().unsqueeze(0))

        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList(
            [
                make_transformer_block(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rates[i],
                    norm_layer=norm_layer,
                    impl=block_impl,
                )
                for i in range(depth)
            ]
        )
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=init_std)
        for emb in self.modality_embed.values():
            trunc_normal_(emb, std=init_std)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _input_proj(self, modality: str):
        return self.input_proj_vis if modality == "vis" else self.input_proj_text

    def _output_proj(self, modality: str):
        return self.output_proj_vis if modality == "vis" else self.output_proj_text

    def _pos(self, modality: str, seq_len: int, batch_size: int):
        positions = self.pos_embeds[modality][:, :seq_len, :]
        return positions.repeat(batch_size, 1, 1)

    def forward(
        self,
        source_tokens: torch.Tensor,
        source_modality: str,
        target_modality: str,
        source_valid_mask: torch.Tensor | None = None,
        target_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict all target-modality tokens from the full source sequence."""
        batch_size = source_tokens.shape[0]
        source_len = source_tokens.shape[1]
        target_len = (
            self.num_patches if target_modality == "vis" else self.max_text_len
        )

        source = self._input_proj(source_modality)(source_tokens)
        source = source + self._pos(source_modality, source_len, batch_size)
        source = source + self.modality_embed[source_modality]
        if source_valid_mask is None:
            source_valid_mask = _all_valid_mask(batch_size, source_len, source.device)
        else:
            source_valid_mask = source_valid_mask.to(device=source.device, dtype=torch.bool)

        target = self.mask_token.repeat(batch_size, target_len, 1)
        target = target + self._pos(target_modality, target_len, batch_size)
        target = target + self.modality_embed[target_modality]
        if target_valid_mask is None:
            target_valid_mask = _all_valid_mask(batch_size, target_len, target.device)
        else:
            target_valid_mask = target_valid_mask.to(device=target.device, dtype=torch.bool)

        tokens = torch.cat([source, target], dim=1)
        key_padding_mask = ~torch.cat([source_valid_mask, target_valid_mask], dim=1)
        for block in self.predictor_blocks:
            tokens = block(tokens, key_padding_mask=key_padding_mask)
        tokens = self.predictor_norm(tokens)
        tokens = tokens.masked_fill(key_padding_mask[..., None], 0.0)
        return self._output_proj(target_modality)(tokens[:, source_len:])


def _optimizer_group_setting(
    module: nn.Module, group_name: str, key: str, default: float
) -> float:
    cfg = getattr(getattr(module, "hparams", None), "optimizer_groups", None)
    group_cfg = getattr(cfg, group_name, None) if cfg is not None else None
    return float(getattr(group_cfg, key, default)) if group_cfg is not None else float(default)


def _split_decay_params(modules, *, trainable_only: bool = True):
    decay, no_decay = [], []
    seen = set()
    for module in modules:
        if module is None:
            continue
        for name, param in module.named_parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            if trainable_only and not param.requires_grad:
                continue
            if ("bias" in name) or len(param.shape) == 1:
                no_decay.append(param)
            else:
                decay.append(param)
    return decay, no_decay


def _make_param_groups(
    module: nn.Module,
    modules,
    *,
    optimizer_group: str,
    lr_schedule: str,
    lr_mult_default: float,
    wd_mult_default: float,
):
    lr_mult = _optimizer_group_setting(module, optimizer_group, "lr_mult", lr_mult_default)
    wd_mult = _optimizer_group_setting(module, optimizer_group, "wd_mult", wd_mult_default)
    decay, no_decay = _split_decay_params(modules)
    groups = []
    if decay:
        groups.append(
            {
                "params": decay,
                "optimizer_group": optimizer_group,
                "lr_schedule": lr_schedule,
                "lr_mult": lr_mult,
                "wd_mult": wd_mult,
            }
        )
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "optimizer_group": optimizer_group,
                "lr_schedule": lr_schedule,
                "lr_mult": lr_mult,
                "wd_mult": 0.0,
                "WD_exclude": True,
                "weight_decay": 0.0,
            }
        )
    return groups


def _task_weight_decay(module: nn.Module) -> float:
    optimizer = getattr(getattr(module, "hparams", None), "optimizer", None)
    keywords = getattr(optimizer, "keywords", {})
    if isinstance(keywords, dict):
        return float(keywords.get("weight_decay", 0.0))
    return 0.0


def _logit_scale_parameter(module: nn.Module) -> dict:
    return {
        "params": [module.logit_scale],
        "optimizer_group": "temperature",
        "lr_schedule": "vision",
        "lr_mult": _optimizer_group_setting(module, "temperature", "lr_mult", 1.0),
        "wd_mult": _optimizer_group_setting(module, "temperature", "wd_mult", 0.0),
        "WD_exclude": True,
        "weight_decay": 0.0,
    }


def _xjepa_param_groups(module: nn.Module):
    """Build the parameter groups for the reported optimizer configuration."""
    groups = []
    groups += _make_param_groups(
        module,
        [module.vis_encoder],
        optimizer_group="encoder",
        lr_schedule="vision",
        lr_mult_default=1.0,
        wd_mult_default=1.0,
    )
    if any(param.requires_grad for param in module.text_encoder.parameters()):
        groups += _make_param_groups(
            module,
            [module.text_encoder],
            optimizer_group="encoder",
            lr_schedule="text",
            lr_mult_default=1.0,
            wd_mult_default=1.0,
        )

    prediction_modules = [
        module.shared_predictor,
        getattr(module, "prediction_proj", None),
        getattr(module, "target_proj", None),
    ]
    groups += _make_param_groups(
        module,
        prediction_modules,
        optimizer_group="prediction_head",
        lr_schedule="vision",
        lr_mult_default=1.0,
        wd_mult_default=1.0,
    )
    projection_modules = [
        getattr(module, "global_proj_vis", None),
        getattr(module, "global_proj_text", None),
    ]
    if any(module_ is not None for module_ in projection_modules):
        groups += _make_param_groups(
            module,
            projection_modules,
            optimizer_group="projection_head",
            lr_schedule="vision",
            lr_mult_default=1.0,
            wd_mult_default=1.0,
        )
    if getattr(module, "alignment_enabled", False) or hasattr(module, "logit_scale"):
        groups.append(_logit_scale_parameter(module))
    return groups


class XJEPA(BaseModule):
    """Shared X-JEPA encoder envelope and the ``[P]`` prediction objective."""

    def __init__(
        self,
        *,
        vis_encoder: nn.Module,
        text_encoder: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler_vis: object,
        lr_scheduler_text: object,
        momentum_scheduler: object | None = None,
        ipe_scale: float = 2.0,
        warmup: float = 4.0,
        unfreeze_last_n: int = -1,
        max_text_len: int = 64,
        alignment_lambda: float = 0.0,
        projection_dim: int = 768,
        projection_hidden_dim: int = 1024,
        prediction_directions: str = "both",
        predictor_embed_dim: int = 384,
        predictor_depth: int = 4,
        predictor_num_heads: int = 2,
        predictor_block_impl: str = "sdpa",
        text_special_token_ids: tuple[int, ...]
        | list[int]
        | None = DEFAULT_TEXT_SPECIAL_TOKEN_IDS,
        optimizer_groups: object | None = None,
        vision_pretrained: object | None = None,
    ):
        super().__init__()
        prediction_directions = _validate_prediction_directions(prediction_directions)
        alignment_lambda = float(alignment_lambda)
        if alignment_lambda < 0.0:
            raise ValueError("alignment_lambda must be non-negative.")
        text_special_token_ids = _normalize_text_special_token_ids(text_special_token_ids)

        self.save_hyperparameters(
            ignore=["vis_encoder", "text_encoder", "vision_pretrained"],
            logger=False,
        )
        self.prediction_directions = prediction_directions
        self.alignment_lambda = alignment_lambda
        self.alignment_enabled = alignment_lambda > 0.0
        self.predictor_embed_dim = int(predictor_embed_dim)
        self.predictor_depth = int(predictor_depth)
        self.predictor_num_heads = int(predictor_num_heads)
        self.predictor_block_impl = predictor_block_impl
        self.text_special_token_ids = text_special_token_ids
        self.hparams.predictor_block_impl = predictor_block_impl

        self.vis_encoder = vis_encoder
        self.num_patches = self.vis_encoder.patch_embed.num_patches
        self.target_vis_encoder = copy.deepcopy(vis_encoder)
        for p in self.target_vis_encoder.parameters():
            p.requires_grad = False
        self.target_vis_encoder.eval()

        self.text_encoder = text_encoder.bert
        self.unfreeze_bert_layers(unfreeze_last_n=unfreeze_last_n)
        self.target_text_encoder = copy.deepcopy(self.text_encoder)
        for p in self.target_text_encoder.parameters():
            p.requires_grad = False
        self.target_text_encoder.eval()

        self.shared_predictor = SharedCrossModalPredictor(
            num_patches=self.num_patches,
            vis_dim=self.vis_encoder.embed_dim,
            text_dim=self.text_encoder.config.hidden_size,
            predictor_embed_dim=self.predictor_embed_dim,
            depth=self.predictor_depth,
            num_heads=self.predictor_num_heads,
            max_text_len=max_text_len,
            block_impl=predictor_block_impl,
        )

        self.global_proj_vis = GlobalProjectionHead(
            in_dim=self.vis_encoder.embed_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_dim,
        )
        self.global_proj_text = GlobalProjectionHead(
            in_dim=self.text_encoder.config.hidden_size,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_dim,
        )

        for name, module in self.named_modules():
            if "text_encoder" in name or "target_text_encoder" in name:
                continue
            init_weights(module)

        self.vis_embeddings = []
        self.text_embeddings = []
        self._lr_group_indices = {"vision": [], "text": []}

        load_vision_pretrained(
            self.vis_encoder,
            target_encoder=self.target_vis_encoder,
            cfg=vision_pretrained,
        )

        if self.alignment_enabled:
            self.info_nce_loss = SymmetricInfoNCELoss()
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    # ------------------------------------------------------------------
    # Checkpoint compatibility
    # ------------------------------------------------------------------
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        _remap_legacy_state_dict_keys(state_dict, prefix)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        self.target_vis_encoder.eval()
        self.target_text_encoder.eval()
        return self

    def _encode_target_streams(self, images, input_ids, attention_mask):
        """Encode image and text streams with the frozen EMA target encoders."""
        self.target_vis_encoder.eval()
        self.target_text_encoder.eval()
        with torch.no_grad():
            target_vis_features = _image_features(self.target_vis_encoder, images)
            target_vis_tokens = _image_patch_tokens_from_output(target_vis_features).detach()
            target_vis_global = _image_global(
                self.target_vis_encoder,
                encoded=target_vis_features,
            ).detach()
            target_text_tokens = self.target_text_encoder(
                input_ids, attention_mask
            ).last_hidden_state.detach()
        return target_vis_tokens, target_text_tokens, target_vis_global

    def _text_target_valid_mask(self, input_ids, attention_mask):
        return _text_target_valid_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_token_ids=self.text_special_token_ids,
        )

    def _layernorm_targets(self, target_vis_tokens, target_text_tokens):
        h_i = F.layer_norm(target_vis_tokens, (target_vis_tokens.size(-1),))
        h_t = F.layer_norm(target_text_tokens, (target_text_tokens.size(-1),))
        return h_i, h_t

    def _gather_if_needed(self, *tensors):
        if self.world_size <= 1:
            return tensors
        return tuple(torch.cat(FullGatherLayer.apply(tensor), dim=0) for tensor in tensors)

    def _predict_targets(self, vis_tokens, text_tokens, text_valid_mask):
        """Run the shared predictor in the active cross-modal directions."""
        directions = self.prediction_directions
        if directions in {"both", "i2t"}:
            pred_text_from_image = self.shared_predictor(
                vis_tokens,
                source_modality="vis",
                target_modality="text",
                target_valid_mask=text_valid_mask,
            )
        else:
            pred_text_from_image = None
        if directions in {"both", "t2i"}:
            pred_image_from_text = self.shared_predictor(
                text_tokens,
                source_modality="text",
                target_modality="vis",
                source_valid_mask=text_valid_mask,
            )
        else:
            pred_image_from_text = None
        return pred_text_from_image, pred_image_from_text

    def _compute_prediction_loss(self, z_i, z_t, h_i, h_t, text_valid_mask=None):
        if z_i is None:
            z_i = h_t.new_zeros(h_t.shape)
        if z_t is None:
            z_t = h_i.new_zeros(h_i.shape)
        return _directional_prediction_loss(
            self,
            z_i,
            z_t,
            h_i,
            h_t,
            text_valid_mask=text_valid_mask,
        )

    def _compute_alignment_loss(self, vis_global, text_global):
        """Direct global image-text alignment term ``L_[A]``."""
        if not self.alignment_enabled:
            zero = vis_global.detach().sum() * 0.0
            return zero, zero, zero

        image_features = F.normalize(self.global_proj_vis(vis_global), dim=-1)
        text_features = F.normalize(self.global_proj_text(text_global), dim=-1)
        image_features, text_features = self._gather_if_needed(image_features, text_features)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        loss = self.info_nce_loss(image_features, text_features, logit_scale)
        return loss, image_features, text_features

    def _forward_impl(self, batch):
        vis_features = _image_features(self.vis_encoder, batch.images)
        vis_tokens = _image_patch_tokens_from_output(vis_features)
        vis_global = _image_global(self.vis_encoder, batch.images, encoded=vis_features)
        text_tokens = self.text_encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_masks,
        ).last_hidden_state
        text_valid_mask = self._text_target_valid_mask(
            batch.input_ids,
            batch.attention_masks,
        )

        target_vis_tokens, target_text_tokens, _ = self._encode_target_streams(
            images=batch.images,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_masks,
        )
        h_i, h_t = self._layernorm_targets(target_vis_tokens, target_text_tokens)
        z_i, z_t = self._predict_targets(vis_tokens, text_tokens, text_valid_mask)
        return {
            "pred_text_from_image": z_i,
            "pred_image_from_text": z_t,
            "target_image": h_i,
            "target_text": h_t,
            "text_valid_mask": text_valid_mask,
            "vis_global": vis_global,
            "text_global": text_tokens[:, 0, :],
        }

    def forward(self, batch):
        """Return the untrained prediction and alignment terms for ``batch``."""
        outputs = self._forward_impl(batch)
        loss_i2t, loss_t2i, loss_prediction = self._compute_prediction_loss(
            outputs["pred_text_from_image"],
            outputs["pred_image_from_text"],
            outputs["target_image"],
            outputs["target_text"],
            text_valid_mask=outputs["text_valid_mask"],
        )
        loss_alignment, _, _ = self._compute_alignment_loss(
            outputs["vis_global"],
            outputs["text_global"],
        )
        return loss_prediction + self.alignment_lambda * loss_alignment

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        outputs = self._forward_impl(batch)
        loss_i2t, loss_t2i, loss_prediction = self._compute_prediction_loss(
            outputs["pred_text_from_image"],
            outputs["pred_image_from_text"],
            outputs["target_image"],
            outputs["target_text"],
            text_valid_mask=outputs["text_valid_mask"],
        )
        loss_alignment, _, _ = self._compute_alignment_loss(
            outputs["vis_global"],
            outputs["text_global"],
        )
        loss = loss_prediction + self.alignment_lambda * loss_alignment

        metrics = {
            "loss": loss.detach(),
            "loss_prediction": loss_prediction.detach(),
            "loss_alignment": loss_alignment.detach(),
            "loss_i2t": loss_i2t.detach(),
            "loss_t2i": loss_t2i.detach(),
        }
        if self.alignment_enabled:
            metrics["logit_scale"] = self.logit_scale.exp().clamp(max=100.0).detach()
        self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=False, prog_bar=False)

        scheduler = self.lr_schedulers()
        optimizer_metrics = {
            "wd": _task_weight_decay(self),
            "ema_vis": self.hparams.momentum_scheduler_vis.get_momentum(),
            "ema_text": self.hparams.momentum_scheduler_text.get_momentum(),
        }
        if scheduler is not None and hasattr(scheduler, "get_last_lr"):
            lrs = scheduler.get_last_lr()
            optimizer_metrics["lr_vis"] = lrs[self._lr_group_indices["vision"][0]]
            optimizer_metrics["lr_text"] = lrs[self._lr_group_indices["text"][0]]
        self.log_dict(
            {f"optim/{key}": value for key, value in optimizer_metrics.items() if value is not None},
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        with torch.no_grad():
            vis_emb = _image_global(self.target_vis_encoder, batch.images)
            self.vis_embeddings.append(vis_emb.detach().cpu())

            text_emb = self.target_text_encoder(
                batch.input_ids,
                batch.attention_masks,
            ).last_hidden_state[:, 0, :]
            self.text_embeddings.append(text_emb.detach().cpu())

    def on_validation_epoch_end(self):
        self.vis_embeddings.clear()
        self.text_embeddings.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.update_weight_decay_and_ema()

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        schedule = self._optimizer_step_schedule()

        optimizer = self.hparams.optimizer(params=self._get_param_groups())
        lr_scheduler = self._make_grouped_lr_scheduler(
            optimizer=optimizer,
            warmup_steps=schedule.warmup_optimizer_steps,
            T_max=schedule.total_optimizer_steps,
        )
        if self.hparams.get("momentum_scheduler") is None:
            raise ValueError("XJEPA requires a `momentum_scheduler` for the EMA target encoders.")
        self.hparams.momentum_scheduler_vis = self.hparams.momentum_scheduler(
            self.vis_encoder,
            self.target_vis_encoder,
            total_steps=schedule.total_optimizer_steps,
        )
        self.hparams.momentum_scheduler_text = self.hparams.momentum_scheduler(
            self.text_encoder,
            self.target_text_encoder,
            total_steps=schedule.total_optimizer_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def _get_param_groups(self, modality=None):
        if modality is None:
            return _xjepa_param_groups(self)

        if modality == "vision":
            extra_groups = [
                self._decay_group(self.shared_predictor),
                self._no_decay_group(self.shared_predictor),
            ]
            return self._tag_param_groups(extra_groups, "vision")

        if modality == "text":
            params = []
            if any(p.requires_grad for p in self.text_encoder.parameters()):
                params += [
                    self._decay_group(self.text_encoder, trainable_only=True),
                    self._no_decay_group(self.text_encoder, trainable_only=True),
                ]
            return self._tag_param_groups(params, "text")

        raise ValueError(f"Unknown modality: {modality}")

    def _make_grouped_lr_scheduler(self, optimizer, warmup_steps, T_max):
        default_lr = optimizer.defaults.get("lr", 0.0)
        schedule_configs = {
            "vision": self._scheduler_config(self.hparams.lr_scheduler_vis, default_lr),
            "text": self._scheduler_config(self.hparams.lr_scheduler_text, default_lr),
        }

        lr_lambdas = []
        self._lr_group_indices = {"vision": [], "text": []}
        for idx, group in enumerate(optimizer.param_groups):
            schedule_name = group["lr_schedule"]
            lr_mult = float(group.get("lr_mult", 1.0))
            base_config = schedule_configs[schedule_name]
            config = {
                "start_lr": base_config["start_lr"] * lr_mult,
                "ref_lr": base_config["ref_lr"] * lr_mult,
                "final_lr": base_config["final_lr"] * lr_mult,
            }
            group["lr"] = config["ref_lr"]
            lr_lambdas.append(self._warmup_cosine_lambda(config, warmup_steps, T_max))
            self._lr_group_indices[schedule_name].append(idx)
            optimizer_group = group.get("optimizer_group")
            if optimizer_group is not None:
                self._lr_group_indices.setdefault(optimizer_group, []).append(idx)

        return LambdaLR(optimizer, lr_lambdas)

    @staticmethod
    def _scheduler_config(scheduler_factory, default_lr):
        kwargs = dict(getattr(scheduler_factory, "keywords", {}) or {})
        ref_lr = float(kwargs.get("ref_lr", default_lr))
        return {
            "start_lr": float(kwargs.get("start_lr", ref_lr)),
            "ref_lr": ref_lr,
            "final_lr": float(kwargs.get("final_lr", 0.0)),
        }

    @staticmethod
    def _warmup_cosine_lambda(config, warmup_steps, T_max):
        start_lr = config["start_lr"]
        ref_lr = config["ref_lr"]
        final_lr = config["final_lr"]
        cosine_steps = T_max - warmup_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                progress = float(current_step) / float(max(1, warmup_steps))
                return (start_lr + progress * (ref_lr - start_lr)) / ref_lr

            progress = float(current_step - warmup_steps) / float(max(1, cosine_steps))
            cosine_lr = max(
                final_lr,
                final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
            )
            return cosine_lr / ref_lr

        return lr_lambda

    @staticmethod
    def _tag_param_groups(param_groups, lr_schedule):
        for group in param_groups:
            group["lr_schedule"] = lr_schedule
        return param_groups

    @staticmethod
    def _decay_group(module, trainable_only=False):
        return {
            "params": (
                p
                for name, p in module.named_parameters()
                if ("bias" not in name)
                and (len(p.shape) != 1)
                and ((not trainable_only) or p.requires_grad)
            )
        }

    @staticmethod
    def _no_decay_group(module, trainable_only=False):
        return {
            "params": (
                p
                for name, p in module.named_parameters()
                if (("bias" in name) or (len(p.shape) == 1))
                and ((not trainable_only) or p.requires_grad)
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        }


class XJEPA_P(XJEPA):
    """X-JEPA [P]: bidirectional latent prediction without global alignment."""

    def __init__(self, *args, **kwargs):
        if float(kwargs.get("alignment_lambda", 0.0)) != 0.0:
            raise ValueError("XJEPA_P requires alignment_lambda=0.0.")
        kwargs.update(prediction_directions="both", alignment_lambda=0.0)
        super().__init__(*args, **kwargs)
        self.paper_model_name = "X-JEPA [P]"


class XJEPA_PA(XJEPA):
    """X-JEPA [P,A]: latent prediction with direct global alignment."""

    def __init__(self, *args, alignment_lambda: float = 0.1, **kwargs):
        alignment_lambda = float(alignment_lambda)
        if alignment_lambda <= 0.0:
            raise ValueError("XJEPA_PA requires a positive alignment_lambda.")
        kwargs.update(prediction_directions="both", alignment_lambda=alignment_lambda)
        super().__init__(*args, **kwargs)
        self.paper_model_name = f"X-JEPA [P,A] lambda={alignment_lambda:g}"


class XJEPA_TC(XJEPA):
    """X-JEPA [TC]: image-to-text predictor/target InfoNCE.

    The variant keeps the shared encoder envelope but replaces the MSE
    prediction objective with an InfoNCE loss between predicted text-space
    features and encoded text-target features.  Its objective uses neither the
    EMA target encoders nor direct global image-text alignment, and it only
    trains the image-to-text direction.
    """

    def __init__(
        self,
        *args,
        target_contrastive_dim: int = 768,
        logit_scale_init: float = float(np.log(1 / 0.07)),
        logit_scale_max: float = 100.0,
        **kwargs,
    ):
        if float(kwargs.get("alignment_lambda", 0.0)) != 0.0:
            raise ValueError("XJEPA_TC requires alignment_lambda=0.0.")
        kwargs.update(prediction_directions="i2t", alignment_lambda=0.0)
        super().__init__(*args, **kwargs)

        text_dim = int(self.text_encoder.config.hidden_size)
        target_contrastive_dim = int(target_contrastive_dim)
        self.prediction_proj = nn.Linear(text_dim, target_contrastive_dim, bias=False)
        self.target_proj = nn.Linear(text_dim, target_contrastive_dim, bias=False)
        self.info_nce_loss = SymmetricInfoNCELoss()
        self.logit_scale = nn.Parameter(
            torch.tensor(float(logit_scale_init), dtype=torch.float32)
        )
        self.logit_scale_max = float(logit_scale_max)
        init_weights(self.prediction_proj)
        init_weights(self.target_proj)

        self.hparams.target_contrastive_dim = target_contrastive_dim
        self.hparams.logit_scale_init = float(logit_scale_init)
        self.hparams.logit_scale_max = self.logit_scale_max
        self.paper_model_name = "X-JEPA [TC]"

    def _clamped_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=self.logit_scale_max)

    def _project_predicted_text(self, tokens, valid_mask, *, normalize: bool):
        pooled = _masked_mean_pool(tokens, valid_mask)
        projected = self.prediction_proj(pooled)
        return F.normalize(projected, dim=-1) if normalize else projected

    def _project_target_text(self, tokens, valid_mask, *, normalize: bool):
        pooled = _masked_mean_pool(tokens, valid_mask)
        projected = self.target_proj(pooled)
        return F.normalize(projected, dim=-1) if normalize else projected

    def _predict_text_tokens(self, vis_tokens, text_len: int, text_valid_mask=None):
        batch_size = int(vis_tokens.shape[0])
        if text_valid_mask is None:
            text_valid_mask = _all_valid_mask(batch_size, int(text_len), vis_tokens.device)
        return self.shared_predictor(
            vis_tokens,
            source_modality="vis",
            target_modality="text",
            target_valid_mask=text_valid_mask,
        )

    def _compute_target_contrastive_loss(self, pred_text_tokens, text_tokens, text_valid_mask):
        pred_features = self._project_predicted_text(
            pred_text_tokens,
            text_valid_mask,
            normalize=True,
        )
        target_features = self._project_target_text(
            text_tokens,
            text_valid_mask,
            normalize=True,
        )
        if getattr(self, "world_size", 1) > 1:
            pred_features, target_features = self._gather_if_needed(
                pred_features,
                target_features,
            )
        loss = self.info_nce_loss(
            pred_features,
            target_features,
            self._clamped_logit_scale(),
        )
        return loss, pred_features, target_features

    @torch.no_grad()
    def encode_image_features(
        self,
        images: torch.Tensor,
        normalize: bool = True,
        target_seq_len: int | None = None,
    ) -> torch.Tensor:
        """Image-side retrieval feature produced by the image-to-text predictor."""
        target_seq_len = int(target_seq_len or getattr(self.hparams, "max_text_len", 64))
        vis_tokens = _image_patch_tokens(self.vis_encoder, images)
        text_valid_mask = _all_valid_mask(images.shape[0], target_seq_len, vis_tokens.device)
        pred_text_tokens = self._predict_text_tokens(
            vis_tokens,
            text_len=target_seq_len,
            text_valid_mask=text_valid_mask,
        )
        return self._project_predicted_text(
            pred_text_tokens,
            text_valid_mask,
            normalize=normalize,
        )

    @torch.no_grad()
    def encode_text_features_from_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Text-side retrieval feature produced by the text-target head."""
        text_tokens = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        text_valid_mask = self._text_target_valid_mask(input_ids, attention_mask)
        return self._project_target_text(
            text_tokens,
            text_valid_mask,
            normalize=normalize,
        )

    def training_step(self, batch, batch_idx):
        del batch_idx
        vis_tokens = _image_patch_tokens(self.vis_encoder, batch.images)
        text_tokens = self.text_encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_masks,
        ).last_hidden_state
        text_valid_mask = self._text_target_valid_mask(
            batch.input_ids,
            batch.attention_masks,
        )
        pred_text_tokens = self._predict_text_tokens(
            vis_tokens,
            text_len=text_tokens.shape[1],
            text_valid_mask=text_valid_mask,
        )
        loss, _, _ = self._compute_target_contrastive_loss(
            pred_text_tokens,
            text_tokens,
            text_valid_mask,
        )

        self.log_dict(
            {
                "loss": loss.detach(),
                "loss_target_contrastive": loss.detach(),
                "logit_scale": self._clamped_logit_scale().detach(),
            },
            on_step=True,
            on_epoch=True,
            sync_dist=False,
            prog_bar=False,
        )

        scheduler = self.lr_schedulers()
        optimizer_metrics = {"wd": _task_weight_decay(self)}
        if scheduler is not None and hasattr(scheduler, "get_last_lr"):
            lrs = scheduler.get_last_lr()
            optimizer_metrics["lr_vis"] = lrs[self._lr_group_indices["vision"][0]]
            optimizer_metrics["lr_text"] = lrs[self._lr_group_indices["text"][0]]
        self.log_dict(
            {f"optim/{key}": value for key, value in optimizer_metrics.items() if value is not None},
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        with torch.no_grad():
            self.vis_embeddings.append(
                self.encode_image_features(batch.images, normalize=False).detach().cpu()
            )
            self.text_embeddings.append(
                self.encode_text_features_from_tokens(
                    batch.input_ids,
                    batch.attention_masks,
                    normalize=False,
                )
                .detach()
                .cpu()
            )


def _remap_legacy_state_dict_keys(state_dict: dict, prefix: str) -> None:
    """Rename pre-release checkpoint keys in place."""
    for legacy_root, current_root in LEGACY_STATE_DICT_KEY_MAP.items():
        legacy_prefix = f"{prefix}{legacy_root}"
        for key in [key for key in state_dict if key.startswith(legacy_prefix)]:
            suffix = key[len(legacy_prefix) :]
            if suffix and not suffix.startswith("."):
                continue
            state_dict[f"{prefix}{current_root}{suffix}"] = state_dict.pop(key)
