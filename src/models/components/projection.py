"""Output heads used by the X-JEPA variants."""

from __future__ import annotations

import torch.nn as nn


class GlobalProjectionHead(nn.Module):
    """Map pooled encoder features into the shared global space.

    The head is either a single linear layer or a two-layer MLP.  Both shapes
    are kept because the released checkpoints use the linear variant while the
    MLP variant is useful when the head has to be re-trained from scratch.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 1024, mlp: bool = False):
        super().__init__()
        if mlp:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Linear(in_dim, out_dim)

    def forward(self, features):
        return self.net(features)
