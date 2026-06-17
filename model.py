"""Tiny ViT encoder + predictor for I-JEPA-style latent prediction.

Small enough to pretrain in minutes on CPU (and much faster on a GPU), but a
faithful miniature of I-JEPA: a patch-embedding ViT context/target encoder and a
narrow transformer predictor that maps context tokens (+ target position
queries) to predicted target latents. The levers (width, depth, heads, predictor
size, patch size) are the main architectural knobs the climb explores.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, E, grid, grid)
        return x.flatten(2).transpose(1, 2)  # (B, N, E)


def sincos_pos_embed(num_patches: int, dim: int) -> torch.Tensor:
    """Fixed 1-D sin-cos positional embedding, shape (1, num_patches, dim)."""
    pos = torch.arange(num_patches, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(dim // 2, dtype=torch.float32).unsqueeze(0)
    denom = torch.pow(10000.0, (2 * i) / max(dim, 1))
    ang = pos / denom
    emb = torch.zeros(num_patches, dim)
    emb[:, 0::2] = torch.sin(ang)
    emb[:, 1::2] = torch.cos(ang)
    return emb.unsqueeze(0)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    """ViT encoder producing per-patch latents."""

    def __init__(self, img_size, patch_size, in_chans, embed_dim, depth, heads, mlp_ratio, dropout):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        n = self.patch_embed.num_patches
        self.register_buffer("pos_embed", sincos_pos_embed(n, embed_dim), persistent=False)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.num_patches = n
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.patch_embed(x) + self.pos_embed
        for blk in self.blocks:
            tok = blk(tok)
        return self.norm(tok)  # (B, N, E)


class Predictor(nn.Module):
    """Maps context latents (+ target position embeddings) to predicted target
    latents. Narrow transformer in a projected space (I-JEPA predictor)."""

    def __init__(self, embed_dim, pred_dim, depth, heads, num_patches, mlp_ratio, dropout):
        super().__init__()
        self.embed = nn.Linear(embed_dim, pred_dim)
        self.register_buffer(
            "pos_embed", sincos_pos_embed(num_patches, pred_dim), persistent=False
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(
            [Block(pred_dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(pred_dim)
        self.out = nn.Linear(pred_dim, embed_dim)

    def forward(self, ctx: torch.Tensor, ctx_idx, tgt_idx) -> torch.Tensor:
        # ctx: (B, n_ctx, E). Build full token grid in predictor space.
        b = ctx.shape[0]
        x = self.embed(ctx) + self.pos_embed[:, ctx_idx, :]
        tgt_tokens = self.mask_token.expand(b, len(tgt_idx), -1) + self.pos_embed[:, tgt_idx, :]
        seq = torch.cat([x, tgt_tokens], dim=1)
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)
        pred_tgt = seq[:, x.shape[1]:, :]  # the target-position outputs
        return self.out(pred_tgt)  # (B, n_tgt, E)
