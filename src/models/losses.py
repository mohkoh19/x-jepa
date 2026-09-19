"""Objective terms shared by the released models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCELoss(nn.Module):
    """Symmetric image-text InfoNCE loss used for CLIP and X-JEPA [P,A]/[TC]."""

    def __init__(self, cache_labels: bool = False):
        super().__init__()
        self.cache_labels = cache_labels
        self.prev_num_logits = 0
        self.labels = {}

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        output_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        total_loss = (loss_i2t + loss_t2i) / 2
        if output_dict:
            return {
                "loss": total_loss,
                "loss_i2t": loss_i2t,
                "loss_t2i": loss_t2i,
                "acc_i2t": (logits_per_image.argmax(dim=1) == labels).float().mean(),
                "acc_t2i": (logits_per_text.argmax(dim=1) == labels).float().mean(),
            }
        return total_loss
