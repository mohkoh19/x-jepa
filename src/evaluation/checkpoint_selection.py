from __future__ import annotations

import csv
import json
import logging
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Mapping

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from src.evaluation.adapters import PretrainedModelAdapter
from src.evaluation.coco_karpathy import compute_coco_karpathy_metrics
from src.evaluation.flickr30k import compute_flickr30k_metrics
from src.evaluation.sugarcrepe import evaluate_sugarcrepe_task

log = logging.getLogger(__name__)

RETRIEVAL_METRICS = {
    "coco_karpathy": compute_coco_karpathy_metrics,
    "flickr30k": compute_flickr30k_metrics,
}

REQUIRED_RETRIEVAL_METRICS = ("i2t_r1", "t2i_r1", "mean_recall")


@dataclass(frozen=True)
class CheckpointCandidate:
    path: str
    name: str
    epoch: int | None
    step: int | None


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    try:
        if key in cfg:
            return cfg[key]
    except (AttributeError, KeyError, TypeError):
        pass
    return getattr(cfg, key, default)


def resolve_run_dir(path: str | Path) -> Path:
    requested = Path(path).expanduser().resolve()
    if requested.is_file():
        for parent in (requested.parent, *requested.parents):
            if (parent / ".hydra" / "config.yaml").exists():
                return parent
        return requested.parent
    return requested


def parse_checkpoint_name(path: str | Path) -> tuple[int | None, int | None]:
    name = Path(path).name
    epoch_match = re.search(r"(?:^|[-_=])epoch[-_=]?(\d+)", name)
    step_match = re.search(r"(?:^|[-_=])(?:global_)?step[-_=]?(\d+)", name)
    epoch = int(epoch_match.group(1)) if epoch_match else None
    step = int(step_match.group(1)) if step_match else None
    return epoch, step


def _unique_run_dirs(run_dir: str | Path, additional_run_dirs: Any = None) -> list[Path]:
    raw_dirs = [run_dir]
    if additional_run_dirs:
        if isinstance(additional_run_dirs, (str, Path)):
            raw_dirs.append(additional_run_dirs)
        else:
            raw_dirs.extend(list(additional_run_dirs))

    resolved_dirs = []
    seen = set()
    for candidate in raw_dirs:
        resolved = resolve_run_dir(candidate)
        key = str(resolved.resolve())
        if key in seen:
            continue
        seen.add(key)
        resolved_dirs.append(resolved)
    return sorted(resolved_dirs, key=lambda path: str(path))


def discover_checkpoints(
    run_dir: str | Path,
    additional_run_dirs: Any = None,
) -> list[CheckpointCandidate]:
    run_dirs = _unique_run_dirs(run_dir, additional_run_dirs)

    paths: list[Path] = []
    for resolved_run_dir in run_dirs:
        checkpoint_dir = resolved_run_dir / "checkpoints"
        if not checkpoint_dir.exists():
            continue
        paths.extend(
            sorted(
                {
                    *checkpoint_dir.glob("*.ckpt"),
                    *checkpoint_dir.glob("*.pth"),
                    *checkpoint_dir.glob("**/*.ckpt"),
                    *checkpoint_dir.glob("**/*.pth"),
                }
            )
        )
    paths = sorted(set(paths))
    if not paths:
        searched = ", ".join(str(path / "checkpoints") for path in run_dirs)
        raise FileNotFoundError(f"No .ckpt or .pth checkpoints found under {searched}")

    candidates = []
    for path in paths:
        epoch, step = parse_checkpoint_name(path)
        candidates.append(
            CheckpointCandidate(
                path=str(path.expanduser().resolve()),
                name=path.name,
                epoch=epoch,
                step=step,
            )
        )

    def sort_key(candidate: CheckpointCandidate) -> tuple[int, int, int, str, str]:
        epoch_missing = 1 if candidate.epoch is None else 0
        return (
            epoch_missing,
            candidate.epoch if candidate.epoch is not None else -1,
            candidate.step if candidate.step is not None else -1,
            candidate.name,
            candidate.path,
        )

    return sorted(candidates, key=sort_key)


def resolve_checkpoint_epoch(
    run_dir: str | Path,
    epoch: int,
    additional_run_dirs: Any = None,
) -> CheckpointCandidate:
    matches = [
        candidate
        for candidate in discover_checkpoints(run_dir, additional_run_dirs)
        if candidate.epoch == int(epoch)
    ]
    if not matches:
        searched = ", ".join(
            str(path / "checkpoints") for path in _unique_run_dirs(run_dir, additional_run_dirs)
        )
        raise FileNotFoundError(f"No checkpoint for epoch {epoch} under {searched}")
    if len(matches) > 1:
        matches = sorted(matches, key=lambda candidate: (candidate.step or -1, candidate.name))
    return matches[-1]


def checkpoint_candidate_from_path(path: str | Path) -> CheckpointCandidate:
    resolved = Path(path).expanduser().resolve()
    epoch, step = parse_checkpoint_name(resolved)
    return CheckpointCandidate(path=str(resolved), name=resolved.name, epoch=epoch, step=step)


def _resolve_device(requested: str | None) -> torch.device:
    requested = (requested or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _batch_images(batch: Any) -> torch.Tensor:
    if isinstance(batch, Mapping):
        return batch["image"]
    if isinstance(batch, (list, tuple)):
        return batch[0]
    raise TypeError(f"Cannot extract images from batch type {type(batch).__name__}")


def _finite_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def _normalize_metric_values(metrics: Mapping[str, Any]) -> dict[str, float]:
    normalized = {}
    for key, value in metrics.items():
        scalar = _finite_float(value)
        if scalar is not None:
            normalized[str(key)] = scalar
    return normalized


def _metric_path(record: Mapping[str, Any], path: str) -> float | None:
    current: Any = record
    parts = str(path).split(".")
    if (
        parts
        and parts[0] != "tasks"
        and isinstance(record.get("tasks"), Mapping)
        and parts[0] in record["tasks"]
    ):
        current = record["tasks"][parts[0]]
        parts = parts[1:]
    for part in parts:
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return _finite_float(current)


def aggregate_score(
    record: Mapping[str, Any],
    metric_weights: Mapping[str, Any] | None,
    *,
    allow_missing: bool = False,
) -> tuple[float | None, list[str]]:
    if not metric_weights:
        return None, []
    total = 0.0
    weight_sum = 0.0
    missing = []
    for metric_path, weight in metric_weights.items():
        value = _metric_path(record, str(metric_path))
        if value is None:
            missing.append(str(metric_path))
            continue
        weight = float(weight)
        total += value * weight
        weight_sum += weight
    if missing and not allow_missing:
        return None, missing
    if weight_sum <= 0:
        return None, missing
    return total / weight_sum, missing


def _sample_rows(features: torch.Tensor, max_rows: int) -> torch.Tensor:
    if features.shape[0] <= max_rows:
        return features
    indices = torch.linspace(0, features.shape[0] - 1, steps=max_rows).round().long()
    return features[indices]


def _avg_pairwise_cosine(features: torch.Tensor, max_rows: int) -> float:
    if features.shape[0] < 2:
        return 0.0
    sampled = _sample_rows(features.detach().float().cpu(), max_rows)
    normalized = F.normalize(sampled, dim=-1)
    similarities = normalized @ normalized.t()
    mask = ~torch.eye(similarities.shape[0], dtype=torch.bool)
    values = similarities[mask]
    return float(values.mean().item()) if values.numel() else 0.0


def _feature_stats(prefix: str, features: torch.Tensor, sample_size: int) -> dict[str, Any]:
    features = features.detach().float().cpu()
    norms = features.norm(dim=-1)
    variance = features.var(dim=0, unbiased=False) if features.shape[0] > 0 else torch.zeros(1)
    return {
        f"{prefix}_shape": list(features.shape),
        f"{prefix}_finite": bool(torch.isfinite(features).all().item()),
        f"{prefix}_norm_mean": float(norms.mean().item()) if norms.numel() else 0.0,
        f"{prefix}_norm_min": float(norms.min().item()) if norms.numel() else 0.0,
        f"{prefix}_norm_std": float(norms.std(unbiased=False).item()) if norms.numel() else 0.0,
        f"{prefix}_variance_mean": float(variance.mean().item()),
        f"{prefix}_variance_min": float(variance.min().item()),
        f"{prefix}_variance_max": float(variance.max().item()),
        f"{prefix}_avg_cosine": _avg_pairwise_cosine(features, sample_size),
    }


def _modality_gap_sanity_score(gap: float) -> float:
    """Bound a lower-is-better modality gap into a higher-is-better [0, 1] score."""
    if not math.isfinite(float(gap)):
        return 0.0
    return 1.0 / (1.0 + max(float(gap), 0.0))


def compute_embedding_diagnostics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    sample_size: int = 2048,
) -> dict[str, Any]:
    image_features = image_features.detach().float().cpu()
    text_features = text_features.detach().float().cpu()
    image_mean = image_features.mean(dim=0)
    text_mean = text_features.mean(dim=0)
    modality_gap = float(torch.linalg.vector_norm(image_mean - text_mean).item())
    modality_gap_cosine = float((1 - F.cosine_similarity(image_mean, text_mean, dim=0)).item())

    diagnostics = {
        **_feature_stats("image", image_features, sample_size),
        **_feature_stats("text", text_features, sample_size),
        "all_finite": bool(
            torch.isfinite(image_features).all().item()
            and torch.isfinite(text_features).all().item()
        ),
        "modality_gap": modality_gap,
        "modality_gap_cosine": modality_gap_cosine,
        "modality_gap_sanity": _modality_gap_sanity_score(modality_gap),
    }
    return diagnostics


def evaluate_retrieval_task(
    adapter: Any,
    datamodule: Any,
    *,
    task_name: str,
    metric_name: str,
    setup_stage: str = "validate",
    text_batch_size: int = 256,
    diagnostic_sample_size: int = 2048,
) -> dict[str, Any]:
    metric_fn = RETRIEVAL_METRICS.get(metric_name)
    if metric_fn is None:
        raise ValueError(f"Unknown retrieval metric `{metric_name}` for task `{task_name}`.")

    datamodule.setup(stage=setup_stage)
    dataset = getattr(datamodule, "val_set", None)
    if dataset is None:
        raise ValueError(f"`{task_name}` datamodule did not create `val_set`.")
    if (
        not hasattr(dataset, "text")
        or not hasattr(dataset, "txt2img")
        or not hasattr(dataset, "img2txt")
    ):
        raise ValueError(f"`{task_name}` dataset does not expose retrieval mappings.")

    text_chunks = []
    texts = list(dataset.text)
    for start in range(0, len(texts), text_batch_size):
        text_batch = texts[start : start + text_batch_size]
        text_chunks.append(adapter.encode_texts(text_batch, normalize=True).detach().float().cpu())

    image_chunks = []
    for batch in datamodule.val_dataloader():
        images = _batch_images(batch)
        image_chunks.append(adapter.encode_images(images, normalize=True).detach().float().cpu())

    if not text_chunks or not image_chunks:
        raise ValueError(f"`{task_name}` validation split is empty.")

    text_features = torch.cat(text_chunks, dim=0)
    image_features = torch.cat(image_chunks, dim=0)
    metrics = _normalize_metric_values(
        metric_fn(
            image_features=image_features,
            text_features=text_features,
            txt2img=dataset.txt2img,
            img2txt=dataset.img2txt,
        )
    )
    if "i2t_r1" in metrics and "t2i_r1" in metrics:
        metrics["mean_r1"] = (metrics["i2t_r1"] + metrics["t2i_r1"]) / 2.0

    diagnostics = compute_embedding_diagnostics(
        image_features=image_features,
        text_features=text_features,
        sample_size=diagnostic_sample_size,
    )
    return {
        "task": task_name,
        "type": "retrieval",
        "metric_name": metric_name,
        "num_images": int(image_features.shape[0]),
        "num_texts": int(text_features.shape[0]),
        "metrics": metrics,
        "diagnostics": diagnostics,
    }


def evaluate_zero_shot_task(adapter: Any, task_cfg: Any, selector_cfg: Any) -> dict[str, Any]:
    task_type = str(_cfg_get(task_cfg, "type", "retrieval")).lower()
    task_name = str(_cfg_get(task_cfg, "name", _cfg_get(task_cfg, "task_name", task_type)))
    setup_stage = str(_cfg_get(task_cfg, "setup_stage", "validate"))

    if task_type == "retrieval":
        datamodule_cfg = _cfg_get(task_cfg, "datamodule")
        datamodule = (
            hydra.utils.instantiate(datamodule_cfg)
            if isinstance(datamodule_cfg, Mapping) or hasattr(datamodule_cfg, "_target_")
            else datamodule_cfg
        )
        return evaluate_retrieval_task(
            adapter,
            datamodule,
            task_name=task_name,
            metric_name=str(_cfg_get(task_cfg, "metric", task_name)),
            setup_stage=setup_stage,
            text_batch_size=int(_cfg_get(selector_cfg, "text_batch_size", 256)),
            diagnostic_sample_size=int(_cfg_get(selector_cfg, "diagnostic_sample_size", 2048)),
        )

    if task_type == "sugarcrepe":
        datamodule_cfg = _cfg_get(task_cfg, "datamodule")
        datamodule = (
            hydra.utils.instantiate(datamodule_cfg)
            if isinstance(datamodule_cfg, Mapping) or hasattr(datamodule_cfg, "_target_")
            else datamodule_cfg
        )
        metrics = evaluate_sugarcrepe_task(
            adapter,
            datamodule,
            score=str(_cfg_get(task_cfg, "score", "itc")),
            setup_stage=setup_stage,
            include_category_metrics=bool(_cfg_get(task_cfg, "include_category_metrics", False)),
        )
        return {
            "task": task_name,
            "type": "sugarcrepe",
            "metric_name": str(_cfg_get(task_cfg, "metric", task_name)),
            "metrics": metrics,
        }

    raise ValueError(f"Unsupported zero-shot task type `{task_type}` for task `{task_name}`.")


def _health_thresholds(health_cfg: Mapping[str, Any] | None) -> dict[str, float | int]:
    return {
        "min_embedding_variance": float(_cfg_get(health_cfg, "min_embedding_variance", 1.0e-12)),
        "min_embedding_norm": float(_cfg_get(health_cfg, "min_embedding_norm", 1.0e-12)),
        "max_abs_avg_cosine": float(_cfg_get(health_cfg, "max_abs_avg_cosine", 0.999999)),
        "min_samples_for_collapse_check": int(
            _cfg_get(health_cfg, "min_samples_for_collapse_check", 2)
        ),
    }


def candidate_health_reasons(
    record: Mapping[str, Any],
    task_names: list[str],
    health_cfg: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    thresholds = _health_thresholds(health_cfg)
    reasons: list[str] = []
    warnings: list[str] = []
    tasks = record.get("tasks") or {}

    for task_name in task_names:
        task = tasks.get(task_name)
        if not isinstance(task, Mapping):
            reasons.append(f"{task_name}: missing task result")
            continue

        metrics = task.get("metrics") or {}
        if str(task.get("type", "retrieval")) == "retrieval":
            for metric_name in REQUIRED_RETRIEVAL_METRICS:
                value = metrics.get(metric_name)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    reasons.append(f"{task_name}: missing or non-finite `{metric_name}`")

        diagnostics = task.get("diagnostics") or {}
        if not diagnostics:
            continue
        if not diagnostics.get("all_finite", False):
            reasons.append(f"{task_name}: non-finite image or text embeddings")

        for modality in ("image", "text"):
            shape = diagnostics.get(f"{modality}_shape") or [0]
            rows = int(shape[0]) if shape else 0
            norm_min = float(diagnostics.get(f"{modality}_norm_min", 0.0))
            variance_mean = float(diagnostics.get(f"{modality}_variance_mean", 0.0))
            avg_cosine = float(diagnostics.get(f"{modality}_avg_cosine", 0.0))
            if norm_min <= thresholds["min_embedding_norm"]:
                reasons.append(f"{task_name}: {modality} embedding norm collapsed")
            if rows >= thresholds["min_samples_for_collapse_check"]:
                if variance_mean <= thresholds["min_embedding_variance"]:
                    reasons.append(f"{task_name}: {modality} embedding variance collapsed")
                if abs(avg_cosine) >= thresholds["max_abs_avg_cosine"]:
                    reasons.append(f"{task_name}: {modality} average cosine indicates collapse")
                elif abs(avg_cosine) >= 0.99:
                    warnings.append(f"{task_name}: high {modality} average cosine")

    return reasons, warnings


def score_candidate_record(
    record: dict[str, Any],
    task_names: list[str],
    health_cfg: Mapping[str, Any] | None = None,
    aggregate_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task_results = record.get("tasks") or {}
    primary_values = []
    mean_recall_values = []
    modality_gap_values = []
    for task_name in task_names:
        task = task_results.get(task_name) or {}
        metrics = task.get("metrics") or {}
        diagnostics = task.get("diagnostics") or {}
        if isinstance(metrics.get("mean_r1"), (int, float)):
            primary_values.append(float(metrics["mean_r1"]))
        if isinstance(metrics.get("mean_recall"), (int, float)):
            mean_recall_values.append(float(metrics["mean_recall"]))
        if isinstance(diagnostics.get("modality_gap"), (int, float)):
            modality_gap_values.append(float(diagnostics["modality_gap"]))

    aggregate_weights = _cfg_get(aggregate_cfg, "metric_weights")
    allow_missing = bool(_cfg_get(aggregate_cfg, "allow_missing", False))
    health_task_names = [
        name
        for name in task_names
        if not (
            allow_missing
            and isinstance(task_results.get(name), Mapping)
            and task_results[name].get("error")
        )
    ]
    reasons, warnings = candidate_health_reasons(record, health_task_names, health_cfg=health_cfg)
    weighted_score, missing_metrics = aggregate_score(
        record,
        aggregate_weights,
        allow_missing=allow_missing,
    )
    if missing_metrics and not allow_missing:
        reasons.extend([f"missing aggregate metric `{name}`" for name in missing_metrics])
    legacy_primary = (
        float(sum(primary_values) / len(primary_values))
        if len(primary_values) == len(task_names)
        else None
    )
    record["primary_score"] = weighted_score if aggregate_weights else legacy_primary
    record["aggregate_score"] = weighted_score
    record["aggregate_missing_metrics"] = missing_metrics
    record["mean_recall"] = (
        float(sum(mean_recall_values) / len(mean_recall_values)) if mean_recall_values else None
    )
    record["modality_gap"] = (
        float(sum(modality_gap_values) / len(modality_gap_values)) if modality_gap_values else None
    )
    record["health"] = {
        "ok": not reasons,
        "rejection_reasons": reasons,
        "warnings": warnings,
        "warning_count": len(warnings),
    }
    record["status"] = "accepted" if not reasons else "rejected"
    return record


def _record_checkpoint_metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("checkpoint") or {}


def _checkpoint_epoch(record: Mapping[str, Any]) -> int:
    epoch = _record_checkpoint_metadata(record).get("epoch")
    return int(epoch) if isinstance(epoch, int) else -1


def _checkpoint_step(record: Mapping[str, Any]) -> int:
    step = _record_checkpoint_metadata(record).get("step")
    return int(step) if isinstance(step, int) else -1


def _tie_float(value: Any, default: float) -> float:
    scalar = _finite_float(value)
    return scalar if scalar is not None else default


def _tie_key(record: Mapping[str, Any]) -> tuple[float, float, int, int, str]:
    return (
        _tie_float(record.get("mean_recall"), float("-inf")),
        -_tie_float(record.get("modality_gap"), float("inf")),
        _checkpoint_epoch(record),
        _checkpoint_step(record),
        str(_record_checkpoint_metadata(record).get("name") or ""),
    )


def _compare_records(left: Mapping[str, Any], right: Mapping[str, Any], tolerance: float) -> int:
    left_ok = left.get("status") == "accepted"
    right_ok = right.get("status") == "accepted"
    if left_ok != right_ok:
        return -1 if left_ok else 1
    if not left_ok and not right_ok:
        left_name = str(_record_checkpoint_metadata(left).get("name"))
        right_name = str(_record_checkpoint_metadata(right).get("name"))
        if left_name == right_name:
            return 0
        return -1 if left_name < right_name else 1

    left_primary = float(left.get("primary_score") or float("-inf"))
    right_primary = float(right.get("primary_score") or float("-inf"))
    if abs(left_primary - right_primary) > tolerance:
        return -1 if left_primary > right_primary else 1
    left_tie = _tie_key(left)
    right_tie = _tie_key(right)
    if left_tie == right_tie:
        return 0
    return -1 if left_tie > right_tie else 1


def rank_checkpoint_records(
    records: list[dict[str, Any]],
    primary_tolerance: float = 0.001,
) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=cmp_to_key(lambda left, right: _compare_records(left, right, primary_tolerance)),
    )
    for rank, record in enumerate(ranked, start=1):
        record["rank"] = rank if record.get("status") == "accepted" else None
    return ranked


def _task_names(tasks_cfg: Mapping[str, Any]) -> list[str]:
    return [str(name) for name in tasks_cfg.keys()]


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _serialize_cfg(cfg: Any) -> Any:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, Mapping):
        return {key: _serialize_cfg(value) for key, value in cfg.items()}
    return cfg


def build_selection_payload(
    *,
    run_dir: str | Path,
    run_dirs: list[Path] | None = None,
    ranked_records: list[dict[str, Any]],
    selector_cfg: Any,
    tasks_cfg: Any,
    primary_tolerance: float,
) -> dict[str, Any]:
    selected = next(
        (record for record in ranked_records if record.get("status") == "accepted"), None
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(resolve_run_dir(run_dir)),
        "run_dirs": [str(path) for path in (run_dirs or [resolve_run_dir(run_dir)])],
        "selected": selected,
        "ranked_candidates": ranked_records,
        "selection_policy": {
            "primary_metric": "weighted validation aggregate",
            "primary_score": (
                "configured selector.aggregate.metric_weights when provided; "
                "otherwise legacy retrieval mean_r1"
            ),
            "primary_tolerance": float(primary_tolerance),
            "tie_breakers": [
                "higher aggregate score",
                "higher retrieval mean_recall",
                "lower average modality_gap",
                "later parsed epoch/step only if still tied",
            ],
            "test_metrics_used": False,
            "note": "Final test metrics are not run or consumed during checkpoint selection.",
        },
        "selector_config": _serialize_cfg(selector_cfg),
        "tasks_config": _serialize_cfg(tasks_cfg),
        "git": {"commit": _git_commit()},
    }


def write_selection_artifacts(
    payload: Mapping[str, Any],
    *,
    run_dir: str | Path,
    output_subdir: str = "checkpoint_selection",
    write_csv: bool = True,
) -> Path:
    output_dir = resolve_run_dir(run_dir) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "selection.json"
    selection_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if write_csv:
        _write_results_csvs(payload, output_dir)

    selected = payload.get("selected") if isinstance(payload, Mapping) else None
    if isinstance(selected, Mapping):
        checkpoint = selected.get("checkpoint") or {}
        selected_path = checkpoint.get("path")
        if selected_path:
            (output_dir / "best_checkpoint.txt").write_text(f"{selected_path}\n", encoding="utf-8")
    return selection_path


def _write_results_csvs(payload: Mapping[str, Any], output_dir: Path) -> None:
    records = list(payload.get("ranked_candidates") or [])
    summary_fields = [
        "rank",
        "status",
        "checkpoint",
        "epoch",
        "step",
        "primary_score",
        "aggregate_score",
        "mean_recall",
        "modality_gap",
    ]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for record in records:
            checkpoint = record.get("checkpoint") or {}
            writer.writerow(
                {
                    "rank": record.get("rank"),
                    "status": record.get("status"),
                    "checkpoint": checkpoint.get("path"),
                    "epoch": checkpoint.get("epoch"),
                    "step": checkpoint.get("step"),
                    "primary_score": record.get("primary_score"),
                    "aggregate_score": record.get("aggregate_score"),
                    "mean_recall": record.get("mean_recall"),
                    "modality_gap": record.get("modality_gap"),
                }
            )

    long_fields = ["checkpoint", "task", "metric", "value"]
    with (output_dir / "results_long.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields)
        writer.writeheader()
        for record in records:
            checkpoint = (record.get("checkpoint") or {}).get("path")
            for task_name, task in (record.get("tasks") or {}).items():
                for metric_name, value in (task.get("metrics") or {}).items():
                    writer.writerow(
                        {
                            "checkpoint": checkpoint,
                            "task": task_name,
                            "metric": metric_name,
                            "value": value,
                        }
                    )


def _candidate_record(candidate: CheckpointCandidate) -> dict[str, Any]:
    return {
        "checkpoint": asdict(candidate),
        "status": "pending",
        "primary_score": None,
        "aggregate_score": None,
        "aggregate_missing_metrics": [],
        "mean_recall": None,
        "modality_gap": None,
        "tasks": {},
        "health": {
            "ok": False,
            "rejection_reasons": [],
            "warnings": [],
            "warning_count": 0,
        },
    }


def _task_cfg_with_name(task_cfg: Any, task_name: str) -> Any:
    """Return a task config copy with a task name attached.

    Hydra-composed task configs can be struct-protected. Directly merging a new
    ``name`` key into such a DictConfig raises ``ConfigKeyError``. Checkpoint
    selection only needs a lightweight per-task copy, so convert DictConfig
    tasks to a plain mutable OmegaConf object before adding the name.
    """
    if isinstance(task_cfg, DictConfig):
        mutable_cfg = OmegaConf.create(OmegaConf.to_container(task_cfg, resolve=True))
        mutable_cfg["name"] = str(task_name)
        return mutable_cfg
    if isinstance(task_cfg, Mapping):
        return {**task_cfg, "name": str(task_name)}
    return task_cfg


def evaluate_checkpoint_candidate(
    candidate: CheckpointCandidate,
    tasks_cfg: Mapping[str, Any],
    selector_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    record = _candidate_record(candidate)
    task_names = _task_names(tasks_cfg)
    aggregate_cfg = _cfg_get(selector_cfg, "aggregate")
    allow_missing = bool(_cfg_get(aggregate_cfg, "allow_missing", False))
    try:
        adapter = PretrainedModelAdapter(
            ckpt_path=candidate.path,
            bert_size=str(_cfg_get(selector_cfg, "bert_size", "auto")),
            max_text_len=int(_cfg_get(selector_cfg, "max_text_len", 64)),
            strict=bool(_cfg_get(selector_cfg, "strict", False)),
        )
        adapter.to(_resolve_device(_cfg_get(selector_cfg, "device", "auto")))
        adapter.eval()

        with torch.inference_mode():
            for task_name, task_cfg in tasks_cfg.items():
                try:
                    task_cfg = _task_cfg_with_name(task_cfg, str(task_name))
                    result = evaluate_zero_shot_task(adapter, task_cfg, selector_cfg)
                    record["tasks"][str(task_name)] = result
                except Exception as exc:
                    if not allow_missing:
                        raise
                    log.warning(
                        "Skipping optional task %s for %s: %s",
                        task_name,
                        candidate.path,
                        exc,
                    )
                    record["tasks"][str(task_name)] = {
                        "task": str(task_name),
                        "type": str(_cfg_get(task_cfg, "type", "retrieval")),
                        "metrics": {},
                        "error": str(exc),
                    }
    except Exception as exc:
        log.exception("Failed to evaluate checkpoint %s", candidate.path)
        record["status"] = "rejected"
        record["health"]["rejection_reasons"] = [f"evaluation failed: {exc}"]
        return record
    finally:
        try:
            del adapter
        except UnboundLocalError:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return score_candidate_record(
        record,
        task_names,
        health_cfg=_cfg_get(selector_cfg, "health"),
        aggregate_cfg=aggregate_cfg,
    )


def run_checkpoint_selection(cfg: DictConfig) -> dict[str, Any]:
    if not cfg.get("run_dir"):
        raise ValueError("Checkpoint selection requires `run_dir=/path/to/training/run`.")

    selector_cfg = cfg.get("selector") or {}
    tasks_cfg = _cfg_get(selector_cfg, "tasks")
    if not tasks_cfg:
        raise ValueError("Checkpoint selection requires at least one validation retrieval task.")

    run_dir = resolve_run_dir(cfg.run_dir)
    run_dirs = _unique_run_dirs(run_dir, _cfg_get(selector_cfg, "additional_run_dirs", []))
    checkpoint_path = _cfg_get(selector_cfg, "checkpoint_path")
    checkpoint_epoch = _cfg_get(selector_cfg, "checkpoint_epoch")
    if checkpoint_path:
        candidates = [checkpoint_candidate_from_path(str(checkpoint_path))]
        log.info("Evaluating explicit checkpoint: %s", candidates[0].path)
    elif checkpoint_epoch is not None:
        candidates = [
            resolve_checkpoint_epoch(
                run_dir,
                int(checkpoint_epoch),
                additional_run_dirs=_cfg_get(selector_cfg, "additional_run_dirs", []),
            )
        ]
        log.info("Evaluating checkpoint for epoch %s: %s", checkpoint_epoch, candidates[0].path)
    else:
        candidates = discover_checkpoints(
            run_dir,
            additional_run_dirs=_cfg_get(selector_cfg, "additional_run_dirs", []),
        )
        log.info(
            "Discovered %s checkpoints under %s",
            len(candidates),
            ", ".join(str(path / "checkpoints") for path in run_dirs),
        )

    records = []
    for index, candidate in enumerate(candidates, start=1):
        log.info("Evaluating checkpoint %s/%s: %s", index, len(candidates), candidate.path)
        records.append(evaluate_checkpoint_candidate(candidate, tasks_cfg, selector_cfg))

    primary_tolerance = float(_cfg_get(selector_cfg, "primary_tolerance", 0.001))
    ranked = (
        records
        if checkpoint_path or checkpoint_epoch is not None
        else rank_checkpoint_records(records, primary_tolerance=primary_tolerance)
    )
    if checkpoint_path or checkpoint_epoch is not None:
        for record in ranked:
            record["rank"] = 1 if record.get("status") == "accepted" else None
    payload = build_selection_payload(
        run_dir=run_dir,
        run_dirs=run_dirs,
        ranked_records=ranked,
        selector_cfg=selector_cfg,
        tasks_cfg=tasks_cfg,
        primary_tolerance=primary_tolerance,
    )

    if payload["selected"] is None:
        raise RuntimeError("No checkpoint passed validation checkpoint-selection health gates.")

    artifact_path = write_selection_artifacts(
        payload,
        run_dir=run_dir,
        output_subdir=str(_cfg_get(selector_cfg, "output_subdir", "checkpoint_selection")),
        write_csv=bool(_cfg_get(selector_cfg, "write_csv", True)),
    )
    log.info("Selected checkpoint: %s", payload["selected"]["checkpoint"]["path"])
    log.info("Wrote checkpoint selection artifact to %s", artifact_path)
    if bool(_cfg_get(selector_cfg, "print_summary", True)):
        selected = payload["selected"]
        print(
            "Selected checkpoint: "
            f"{selected['checkpoint']['path']} "
            f"(score={selected.get('primary_score')})"
        )
    return payload
