# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Mohammad Kohankhaki
# Licensed under CC BY-NC 4.0 (see LICENSE file)
#

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


class MomentumScheduler:
    """
    Implements an Exponential Moving Average (EMA) Momentum Scheduler
    for updating a target encoder from an encoder.

    Args:
        encoder (torch.nn.Module): The online encoder model.
        target_encoder (torch.nn.Module): The target encoder model (EMA model).
        base_m (float): Initial EMA momentum.
        final_m (float): Final EMA momentum.
        total_steps (int): Total number of training steps (epochs * steps_per_epoch).
    """

    def __init__(self, encoder, target_encoder, base_m, final_m, total_steps, current_step=0):
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.base_m = base_m
        self.final_m = final_m
        self.total_steps = max(1, int(total_steps))
        self.current_step = current_step  # Track the current training step

    def get_momentum(self):
        """Compute current EMA momentum based on step progress."""
        progress = min(
            max(self.current_step, 0) / self.total_steps, 1.0
        )  # Clamp progress to [0,1]
        return self.base_m + (self.final_m - self.base_m) * progress  # Linear momentum decay

    @torch.no_grad()
    def step(self):
        """Update target encoder parameters using EMA."""
        self.current_step += 1
        m = self.get_momentum()  # Compute new momentum

        # Perform EMA update
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            if param_q.numel() == 0 or param_k.numel() == 0:
                continue  # Skip empty parameters
            param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

    def state_dict(self):
        """Returns the scheduler state for checkpointing."""
        return {"step": self.current_step}

    def load_state_dict(self, state_dict):
        """Loads the scheduler state from a checkpoint."""
        self.current_step = int(state_dict.get("step", 0))


# The following schedulers are based on https://github.com/facebookresearch/ijepa/blob/main/src/utils/schedulers.py


class WarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, T_max, final_lr=0.0):
        if ref_lr == 0:
            raise ValueError("WarmupCosineSchedule requires non-zero `ref_lr`.")
        self.warmup_steps = max(0, int(warmup_steps))
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.T_max = max(1, int(T_max) - self.warmup_steps)
        self.final_lr = final_lr

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                # Linear warmup phase
                progress = float(current_step) / float(max(1, self.warmup_steps))
                return (self.start_lr + progress * (self.ref_lr - self.start_lr)) / self.ref_lr
            else:
                # Cosine annealing phase
                progress = min(
                    float(current_step - self.warmup_steps) / float(max(1, self.T_max)),
                    1.0,
                )
                cosine_lr = max(
                    self.final_lr,
                    self.final_lr
                    + (self.ref_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
                return cosine_lr / self.ref_lr

        super().__init__(optimizer, lr_lambda)

