"""Shared Lightning module used by the X-JEPA and contrastive baseline models."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch.nn as nn
from lightning import LightningModule

logger = logging.getLogger(__name__)


@dataclass
class Vicreg:
    """Legacy objective configuration stored in the released checkpoints.

    The published X-JEPA checkpoints pickle this dataclass as part of their
    ``hyper_parameters``.  It is not used by the current objectives, but it has
    to remain importable so that those checkpoints can be loaded.
    """

    gamma_vis: float = 1.0
    gamma_text: float = 1.0
    eps_vis: float = 1e-4
    eps_text: float = 1e-4
    mu_cov_vis: float = 1.0
    mu_cov_text: float = 1.0
    beta: float = 0.001
    lambda_var_vis: float = 25
    lambda_var_text: float = 25
    lambda_sim: int = 25
    include_cov_term: bool = True
    include_var_term: bool = True
    include_invariance_term: bool = False
    vicreg_dim: int = 768


@dataclass(frozen=True)
class OptimizerStepSchedule:
    """Resolved optimizer-step budget for the learning-rate schedules."""

    num_training_batches: int
    accumulate_grad_batches: int
    optimizer_steps_per_epoch: int
    total_optimizer_steps: int
    warmup_optimizer_steps: int


def _resolve_accumulate_grad_batches(trainer) -> int:
    accumulate = getattr(trainer, "accumulate_grad_batches", 1)
    if isinstance(accumulate, dict):
        current_epoch = int(getattr(trainer, "current_epoch", 0))
        active_epochs = [int(epoch) for epoch in accumulate if int(epoch) <= current_epoch]
        if active_epochs:
            accumulate = accumulate[max(active_epochs)]
        else:
            accumulate = accumulate[min(accumulate)]
    return max(1, int(accumulate))


def compute_optimizer_step_schedule(
    trainer,
    ipe_scale: float,
    warmup: float,
) -> OptimizerStepSchedule:
    """Derive the optimizer-step budget used by the warmup and cosine schedules.

    ``ipe_scale`` scales the schedule horizon relative to the nominal training
    length and ``warmup`` gives the warmup length in epochs.  Both values are
    part of the reported pretraining setup.
    """

    num_training_batches = getattr(trainer, "num_training_batches", None)
    if num_training_batches == float("inf"):
        num_training_batches = len(trainer.datamodule.train_dataloader())
    if num_training_batches is None:
        raise ValueError("Cannot compute optimizer schedule without trainer.num_training_batches.")

    num_training_batches = int(num_training_batches)
    accumulate_grad_batches = _resolve_accumulate_grad_batches(trainer)
    optimizer_steps_per_epoch = math.ceil(num_training_batches / accumulate_grad_batches)
    total_optimizer_steps = int(
        optimizer_steps_per_epoch * int(trainer.max_epochs) * float(ipe_scale)
    )
    warmup_optimizer_steps = int(float(warmup) * optimizer_steps_per_epoch)

    return OptimizerStepSchedule(
        num_training_batches=num_training_batches,
        accumulate_grad_batches=accumulate_grad_batches,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        total_optimizer_steps=total_optimizer_steps,
        warmup_optimizer_steps=warmup_optimizer_steps,
    )


class BaseModule(LightningModule):
    """Gradient-accumulation, BERT-freezing and schedule helpers.

    The class contains only behaviour shared by the X-JEPA variants and the
    CLIP/SigLIP baselines.
    """

    def _accumulate_grad_batches(self) -> int:
        return max(1, int(getattr(self.trainer, "accumulate_grad_batches", 1)))

    def _is_last_training_batch(self, batch_idx: int) -> bool:
        num_batches = getattr(self.trainer, "num_training_batches", None)
        return num_batches not in (None, float("inf")) and (batch_idx + 1) >= num_batches

    def _should_step_optimizers(self, batch_idx: int) -> bool:
        accumulate = self._accumulate_grad_batches()
        return ((batch_idx + 1) % accumulate == 0) or self._is_last_training_batch(batch_idx)

    def _current_accumulation_window(self, batch_idx: int) -> int:
        if self._is_last_training_batch(batch_idx):
            return (batch_idx % self._accumulate_grad_batches()) + 1
        return self._accumulate_grad_batches()

    def _optimizer_step_schedule(self) -> OptimizerStepSchedule:
        schedule = compute_optimizer_step_schedule(
            self.trainer,
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
            "Optimizer-step schedule: batches=%s accumulate=%s steps_per_epoch=%s "
            "total_steps=%s warmup_steps=%s",
            schedule.num_training_batches,
            schedule.accumulate_grad_batches,
            schedule.optimizer_steps_per_epoch,
            schedule.total_optimizer_steps,
            schedule.warmup_optimizer_steps,
        )
        return schedule

    def on_fit_start(self):
        self.world_size = self.trainer.world_size

        if self.trainer.ckpt_path:
            logger.info("Restoring EMA scheduler state to the current step.")
            if hasattr(self.hparams, "momentum_scheduler_vis"):
                self.hparams.momentum_scheduler_vis.encoder = self.vis_encoder
                self.hparams.momentum_scheduler_vis.target_encoder = self.target_vis_encoder
                self.hparams.momentum_scheduler_vis.current_step = self.current_step

            if hasattr(self.hparams, "momentum_scheduler_text"):
                self.hparams.momentum_scheduler_text.encoder = self.text_encoder
                self.hparams.momentum_scheduler_text.target_encoder = self.target_text_encoder
                self.hparams.momentum_scheduler_text.current_step = self.current_step

        return super().on_fit_start()

    def on_load_checkpoint(self, checkpoint):
        """Restore ``global_step`` from a checkpoint without rescaling it.

        PyTorch Lightning stores ``global_step`` as the number of optimizer
        updates regardless of the number of ranks, so it is assigned directly
        instead of being divided by the world size.
        """

        self.current_step = int(checkpoint.get("global_step", 0))
        return super().on_load_checkpoint(checkpoint)

    def unfreeze_bert_layers(self, unfreeze_last_n: int = -1) -> None:
        """Control which parts of the BERT text encoder are trainable.

        Semantics:

        - ``unfreeze_last_n < 0``: unfreeze *all* BERT parameters.
        - ``unfreeze_last_n == 0``: freeze *all* BERT parameters.
        - ``unfreeze_last_n > 0``: freeze everything, then unfreeze the last
          ``unfreeze_last_n`` encoder layers (and the pooler, if present).

        Embeddings stay frozen when ``unfreeze_last_n > 0``.
        """

        if not hasattr(self, "text_encoder") or self.text_encoder is None:
            raise AttributeError("BaseModule.unfreeze_bert_layers() requires self.text_encoder")

        encoder_layers = getattr(getattr(self.text_encoder, "encoder", None), "layer", None)
        total_layers = len(encoder_layers) if encoder_layers is not None else 0

        if int(unfreeze_last_n) < 0:
            for param in self.text_encoder.parameters():
                param.requires_grad = True
            logger.info("Unfreeze all BERT parameters.")

        elif int(unfreeze_last_n) == 0:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            logger.info("Freeze all BERT parameters.")

        else:
            requested_n = int(unfreeze_last_n)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            logger.info("Freeze all BERT parameters.")

            if encoder_layers is None:
                logger.warning(
                    "Cannot unfreeze last %s BERT layers because text_encoder.encoder.layer is missing.",
                    requested_n,
                )
            else:
                effective_n = min(requested_n, total_layers)
                for layer in encoder_layers[-effective_n:]:
                    for param in layer.parameters():
                        param.requires_grad = True

                logger.info("Unfreeze last %s/%s BERT layers.", effective_n, total_layers)
                if requested_n > total_layers:
                    logger.info(
                        "Requested unfreeze_last_n=%s exceeds total_layers=%s; "
                        "unfreezing all encoder layers.",
                        requested_n,
                        total_layers,
                    )

            pooler = getattr(self.text_encoder, "pooler", None)
            if pooler is not None:
                for param in pooler.parameters():
                    param.requires_grad = True

        # Keep module mode consistent with trainability.  Lightning toggles
        # every submodule on train/eval switches, which must not re-enable
        # dropout inside a frozen text encoder.
        if any(p.requires_grad for p in self.text_encoder.parameters()):
            self.text_encoder.train(self.training)
        else:
            self.text_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)

        text_encoder = getattr(self, "text_encoder", None)
        if isinstance(text_encoder, nn.Module):
            if any(p.requires_grad for p in text_encoder.parameters()):
                text_encoder.train(mode)
            else:
                text_encoder.eval()

        return self

    def update_weight_decay_and_ema(self):
        if hasattr(self.hparams, "momentum_scheduler_vis"):
            self.hparams.momentum_scheduler_vis.step()
        if hasattr(self.hparams, "momentum_scheduler_text"):
            self.hparams.momentum_scheduler_text.step()
