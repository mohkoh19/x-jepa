from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import DictConfig, OmegaConf


def _to_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def normalize_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    normalized = {}
    for key, value in metrics.items():
        scalar = _to_float(value)
        if scalar is not None:
            normalized[str(key)] = scalar
    return normalized


def prefix_metrics(metrics: Mapping[str, float], prefix: str | None) -> dict[str, float]:
    clean_prefix = (prefix or "").strip("/")
    if not clean_prefix:
        return dict(metrics)
    return {f"{clean_prefix}/{key}": value for key, value in metrics.items()}


def _cfg_payload(cfg: DictConfig, key: str) -> dict | None:
    value = cfg.evaluation.get(key) if isinstance(cfg.get("evaluation"), DictConfig) else None
    if value is None:
        return None
    return OmegaConf.to_container(value, resolve=True)


def log_evaluation_metrics(
    metrics: Mapping[str, Any], cfg: DictConfig, trainer: Any
) -> dict[str, float]:
    normalized = normalize_metrics(metrics)
    logged_metrics = prefix_metrics(normalized, cfg.evaluation.get("metric_prefix"))

    if getattr(trainer, "is_global_zero", True):
        output_dir = Path(cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "ckpt_path": cfg.get("ckpt_path"),
            "metrics": normalized,
            "logged_metrics": logged_metrics,
            "pretrain_run": _cfg_payload(cfg, "pretrain_run"),
        }
        (output_dir / "evaluation_metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if trainer.logger and logged_metrics:
        for logger in trainer.loggers:
            logger.log_metrics(logged_metrics)

    return logged_metrics
