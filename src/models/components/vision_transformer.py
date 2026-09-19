# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, key_padding_mask: torch.Tensor | None = None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device=x.device, dtype=torch.bool)
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :],
                -torch.finfo(attn.dtype).max,
            )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x, return_attention=False, key_padding_mask: torch.Tensor | None = None):
        y, attn = self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device=x.device, dtype=torch.bool)
            x = x.masked_fill(key_padding_mask[..., None], 0.0)
        return x


class SDPAAttention(nn.Module):
    """Attention module with the same parameter names as Attention but SDPA kernels.

    The module preserves the custom Attention interface and key-padding semantics:
    key_padding_mask is a boolean [B, N] tensor where True means invalid/padded key.
    Query positions marked invalid are zeroed by SDPABlock after the residual path,
    matching Block's behavior.
    """

    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.attn_drop_p = float(attn_drop)

        # Keep parameter/module names compatible with the custom Attention block.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _attention_weights(self, q, k, key_padding_mask: torch.Tensor | None = None):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device=q.device, dtype=torch.bool)
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :],
                -torch.finfo(attn.dtype).max,
            )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def forward(
        self,
        x,
        key_padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if return_attention:
            attn = self._attention_weights(q, k, key_padding_mask=key_padding_mask)
            y = (attn @ v).transpose(1, 2).reshape(B, N, C)
            y = self.proj(y)
            y = self.proj_drop(y)
            return y, attn

        attn_mask = None
        if key_padding_mask is not None:
            # Use an additive mask to avoid bool-mask semantic ambiguity.  The
            # custom attention masks keys with -finfo.max before softmax.
            key_padding_mask = key_padding_mask.to(device=x.device, dtype=torch.bool)
            attn_mask = torch.zeros(B, 1, 1, N, dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill(
                key_padding_mask[:, None, None, :],
                -torch.finfo(q.dtype).max,
            )

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
        y = y.transpose(1, 2).reshape(B, N, C)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y, None


class SDPABlock(nn.Module):
    """Drop-in SDPA variant of Block with matching input/output semantics."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SDPAAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x, return_attention=False, key_padding_mask: torch.Tensor | None = None):
        y, attn = self.attn(
            self.norm1(x),
            key_padding_mask=key_padding_mask,
            return_attention=return_attention,
        )
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device=x.device, dtype=torch.bool)
            x = x.masked_fill(key_padding_mask[..., None], 0.0)
        return x




def make_transformer_block(
    *,
    dim,
    num_heads,
    mlp_ratio=4.0,
    qkv_bias=False,
    qk_scale=None,
    drop=0.0,
    attn_drop=0.0,
    drop_path=0.0,
    act_layer=nn.GELU,
    norm_layer=nn.LayerNorm,
    impl: str = "sdpa",
):
    """Factory for custom vs. SDPA transformer blocks.

    The SDPA block is the default fast middle-fusion implementation.  The
    custom block remains available for compatibility/diagnostics via
    fusion_block_impl=custom.
    """
    impl = str(impl).lower()
    kwargs = dict(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        drop=drop,
        attn_drop=attn_drop,
        drop_path=drop_path,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    if impl == "custom":
        return Block(**kwargs)
    if impl == "sdpa":
        return SDPABlock(**kwargs)
    raise ValueError(
        "Unsupported transformer block implementation " f"{impl!r}. Use 'custom' or 'sdpa'."
    )
