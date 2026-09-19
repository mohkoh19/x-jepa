"""Resolve and load pretrained checkpoints for evaluation.

Two checkpoint layouts are supported:

1. A released checkpoint file, e.g. ``checkpoints/xjepa_pa_lam01.ckpt``.  Its
   architecture is read from ``configs/checkpoints/<name>.yaml`` (or from a
   sibling ``<name>.yaml`` next to the checkpoint).
2. A Hydra training run directory containing ``.hydra/config.yaml``, which is
   how the internal training runs store their resolved configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import torch
import yaml

log = logging.getLogger(__name__)

#: Model targets used before the public release.
LEGACY_MODEL_TARGETS = {
    "src.models.crossjepa.XJEPA": "src.models.xjepa.XJEPA",
    "src.models.crossjepa.CXJEPA": "src.models.xjepa.XJEPA",
    "src.models.new_models.XJEPA_new": "src.models.xjepa.XJEPA",
    "src.models.paper.XJEPA_P": "src.models.xjepa.XJEPA_P",
    "src.models.paper.XJEPA_PA": "src.models.xjepa.XJEPA_PA",
    "src.models.paper.XJEPA_TC": "src.models.xjepa.XJEPA_TC",
    "src.models.xjepa.PredictionXJEPA": "src.models.xjepa.XJEPA",
    "src.models.xjepa.TargetContrastiveXJEPA": "src.models.xjepa.XJEPA_TC",
}

#: Config keys that used to be part of the model constructor.
LEGACY_MODEL_KEYS = (
    "clip_loss",
    "contrastive_loss",
    "contrastive_weight",
    "diagnostics",
    "direction_loss_weights",
    "jepa_weight",
    "log_embedding_stats_every",
    "log_grad_every",
    "mlm",
    "mlp_vicreg",
    "multimodal",
    "objective_mode",
    "objective_weighting",
    "predictor_sharing",
    "regularization_projectors_enabled",
    "regularization_weight",
    "regularizer_mode",
    "scale_sigreg_by_regularization_weight",
    "sigreg",
    "sigreg_loss_weight",
    "sigreg_reference_batch_size",
    "sigreg_weight",
    "use_sigreg",
    "use_vicreg",
    "vicreg",
    "vicreg_dim",
    "vicreg_weight",
    "vis_predictor",
    "vision_backbone_impl",
    "wd_scheduler",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_run_dir(path: str | Path) -> Path | None:
    """Return the Hydra run directory containing ``path``, if there is one."""
    resolved = Path(path).expanduser().resolve()
    start = resolved.parent if resolved.is_file() else resolved
    for candidate in (start, *start.parents):
        if (candidate / ".hydra" / "config.yaml").exists():
            return candidate
    return None


def load_run_config(path: str | Path) -> Dict[str, Any]:
    """Load the resolved Hydra config stored with a training run."""
    run_dir = find_run_dir(path)
    if run_dir is None:
        raise FileNotFoundError(f"Expected a Hydra run directory for {path}")
    with (run_dir / ".hydra" / "config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_checkpoint(path: str | Path) -> Path:
    """Resolve a checkpoint file from a path or a training run directory."""
    requested = Path(path).expanduser().resolve()
    if requested.suffix in {".ckpt", ".pth"} and requested.exists():
        return requested

    run_dir = find_run_dir(requested)
    search_roots = []
    if run_dir is not None:
        checkpoint_dir = run_dir / "checkpoints"
        search_roots = [checkpoint_dir] if checkpoint_dir.exists() else [run_dir]
    elif requested.is_dir():
        search_roots = [requested]

    for root in search_roots:
        for pattern in ("*.ckpt", "*.pth", "**/*.ckpt", "**/*.pth"):
            matches = sorted(root.glob(pattern))
            if matches:
                preferred = [match for match in matches if match.name != "last.ckpt"]
                checkpoint_path = preferred[0] if preferred else matches[0]
                log.info("Resolved checkpoint %s from %s", checkpoint_path, root)
                return checkpoint_path
    raise FileNotFoundError(f"Could not find a checkpoint under {requested}")


def migrate_model_config(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map pre-release model configs onto the current implementation."""
    cfg = dict(model_cfg)
    for key in LEGACY_MODEL_KEYS:
        cfg.pop(key, None)

    target = cfg.get("_target_")
    migrated = LEGACY_MODEL_TARGETS.get(target)
    if migrated is not None:
        log.warning("Migrating archived checkpoint target %s -> %s", target, migrated)
        cfg["_target_"] = migrated

    if "mlp_vicreg" in model_cfg:
        cfg["projection_mlp"] = bool(model_cfg["mlp_vicreg"])
    if "vicreg_dim" in model_cfg:
        cfg["projection_dim"] = int(model_cfg["vicreg_dim"])

    optimizer_groups = cfg.get("optimizer_groups")
    if isinstance(optimizer_groups, dict) and "regularization_head" in optimizer_groups:
        optimizer_groups = dict(optimizer_groups)
        optimizer_groups["projection_head"] = optimizer_groups.pop("regularization_head")
        optimizer_groups.setdefault(
            "temperature",
            optimizer_groups.pop("logit_scale", {"lr_mult": 1.0, "wd_mult": 0.0}),
        )
        cfg["optimizer_groups"] = optimizer_groups
    return cfg


def load_checkpoint_config(name: str) -> Dict[str, Any] | None:
    """Load the shipped architecture of a released checkpoint, if present."""
    path = repo_root() / "configs" / "checkpoints" / f"{name}.yaml"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_model_config(path: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(model_config, run_config)`` for a checkpoint or run directory."""
    resolved = Path(path).expanduser().resolve()
    if find_run_dir(resolved) is not None:
        run_cfg = load_run_config(resolved)
        return migrate_model_config(run_cfg["model"]), run_cfg

    candidates = [resolved.parent / f"{resolved.stem}.yaml", resolved.with_suffix(".yaml")]
    for candidate in candidates:
        if candidate.exists() and candidate.suffix == ".yaml":
            with candidate.open("r", encoding="utf-8") as handle:
                model_cfg = yaml.safe_load(handle) or {}
            return migrate_model_config(model_cfg), {"model": model_cfg}

    model_cfg = load_checkpoint_config(resolved.stem)
    if model_cfg is not None:
        return migrate_model_config(model_cfg), {"model": model_cfg}

    raise FileNotFoundError(
        f"Could not resolve a model config for `{path}`. Provide a Hydra run "
        f"directory, a checkpoint created by this project, or a sibling "
        f"`{resolved.stem}.yaml` model config."
    )


def instantiate_checkpoint_model(path: str | Path) -> Tuple[torch.nn.Module, Dict[str, Any], Path]:
    model_cfg, run_cfg = load_model_config(path)
    model = hydra.utils.instantiate(model_cfg)
    return model, run_cfg, find_checkpoint(path)


def load_model_from_run(
    path: str | Path, strict: bool = True
) -> Tuple[torch.nn.Module, Dict[str, Any], Path]:
    """Instantiate the model of ``path`` and load its released weights."""
    model, run_cfg, checkpoint_path = instantiate_checkpoint_model(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if checkpoint_path.suffix == ".ckpt" else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning("Missing keys when loading %s: %s", checkpoint_path, missing[:10])
    if unexpected:
        log.warning("Unexpected keys when loading %s: %s", checkpoint_path, unexpected[:10])
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Could not load {checkpoint_path} strictly; "
            f"missing={list(missing)[:10]}, unexpected={list(unexpected)[:10]}"
        )
    model.eval()
    return model, run_cfg, checkpoint_path


def load_module_from_run(path: str | Path, module: str = "target_vis_encoder") -> torch.nn.Module:
    model, _, checkpoint_path = load_model_from_run(path, strict=False)
    if not hasattr(model, module):
        raise AttributeError(f"Checkpoint at {checkpoint_path} does not expose module `{module}`")
    submodule = getattr(model, module)
    submodule.eval()
    return submodule
