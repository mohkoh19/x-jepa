from __future__ import annotations

import math
from typing import Any

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict

from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.utils.logging_utils import log_hyperparameters
from src.utils.pylogger import RankedLogger
from src.utils.utils import set_matmul_precision

log = RankedLogger(__name__, rank_zero_only=True)


def _num_devices(devices: Any) -> int:
    if devices in (None, "auto"):
        return 1
    if isinstance(devices, int):
        return max(1, devices)
    if isinstance(devices, str):
        if devices.isdigit():
            return max(1, int(devices))
        if "," in devices:
            return max(1, len([part for part in devices.split(",") if part.strip()]))
        return 1
    if isinstance(devices, (list, tuple)):
        return max(1, len(devices))
    return 1


def resolve_effective_batch_size(cfg: DictConfig) -> None:
    """Derive gradient accumulation from a requested global batch size."""
    target_batch_size = cfg.get("effective_batch_size")
    if target_batch_size in (None, "null"):
        return
    if not cfg.get("data") or not cfg.data.get("batch_size"):
        raise ValueError("`effective_batch_size` requires `data.batch_size`.")

    per_device_batch_size = int(cfg.data.batch_size)
    devices = _num_devices(cfg.trainer.get("devices", 1))
    num_nodes = int(cfg.trainer.get("num_nodes", 1))
    local_global_batch = per_device_batch_size * devices * num_nodes
    if local_global_batch <= 0:
        raise ValueError("Resolved local global batch size must be positive.")

    # Ensure that the per‑device batch size and world configuration do not
    # exceed the requested global batch size.  If the local global batch
    # (batch_size × devices × num_nodes) is larger than the target, the
    # training would process more samples per optimisation step than
    # requested.  In such cases we fail early and ask the user to
    # decrease `data.batch_size` or the number of devices/nodes.  This
    # guards against unintentional mismatches in experimental settings and
    # preserves fairness across different world sizes.
    target = int(target_batch_size)
    if local_global_batch > target:
        raise ValueError(
            "Resolved local global batch size {} (batch_size={} × devices={} × nodes={}) "
            "exceeds the requested effective_batch_size {}.  Please reduce "
            "`data.batch_size` or the number of devices/nodes to achieve the desired "
            "global batch size.".format(
                local_global_batch,
                per_device_batch_size,
                devices,
                num_nodes,
                target_batch_size,
            )
        )

    accumulate = max(1, int(math.ceil(target / local_global_batch)))
    resolved_batch_size = local_global_batch * accumulate
    with open_dict(cfg):
        cfg.trainer.accumulate_grad_batches = accumulate
        cfg.effective_batch_size_resolved = resolved_batch_size

    if resolved_batch_size != target:
        log.warning(
            "Requested effective_batch_size=%s, resolved %s with accumulate_grad_batches=%s.",
            target_batch_size,
            resolved_batch_size,
            accumulate,
        )
    else:
        log.info(
            "Using effective_batch_size=%s with accumulate_grad_batches=%s.",
            resolved_batch_size,
            accumulate,
        )


def prepare_runtime(cfg: DictConfig) -> None:
    """Apply shared runtime setup before Hydra-instantiated work begins."""
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    if cfg.get("matmul_precision"):
        set_matmul_precision(cfg.matmul_precision)

    resolve_effective_batch_size(cfg)


def instantiate_datamodule(cfg: DictConfig) -> LightningDataModule | None:
    if not cfg.get("data") or not cfg.data.get("_target_"):
        return None

    log.info("Instantiating datamodule <%s>", cfg.data._target_)
    return hydra.utils.instantiate(cfg.data)


def instantiate_model(cfg: DictConfig) -> LightningModule:
    log.info("Instantiating model <%s>", cfg.model._target_)
    return hydra.utils.instantiate(cfg.model)


def instantiate_trainer(
    cfg: DictConfig,
    callbacks: list[Callback] | None = None,
    logger: list[Logger] | None = None,
) -> Trainer:
    trainer_kwargs: dict[str, Any] = {}
    if callbacks is not None:
        trainer_kwargs["callbacks"] = callbacks
    if logger is not None:
        trainer_kwargs["logger"] = logger

    log.info("Instantiating trainer <%s>", cfg.trainer._target_)
    return hydra.utils.instantiate(cfg.trainer, **trainer_kwargs)


def build_training_context(cfg: DictConfig) -> dict[str, Any]:
    datamodule = instantiate_datamodule(cfg)
    model = instantiate_model(cfg)

    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers = instantiate_loggers(cfg.get("logger"))

    trainer = instantiate_trainer(cfg, callbacks=callbacks, logger=loggers)

    return {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }


def build_evaluation_context(cfg: DictConfig) -> dict[str, Any]:
    model = instantiate_model(cfg)
    datamodule = instantiate_datamodule(cfg)

    log.info("Instantiating loggers...")
    loggers = instantiate_loggers(cfg.get("logger"))

    trainer = instantiate_trainer(cfg, logger=loggers)

    return {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": loggers,
        "trainer": trainer,
    }


def maybe_log_hyperparameters(object_dict: dict[str, Any]) -> None:
    trainer = object_dict["trainer"]
    if trainer.logger:
        log.info("Logging hyperparameters")
        log_hyperparameters(object_dict)
