"""Entrypoint: pretrain a tiny I-JEPA, then linear-probe it on the synthetic
ImageNet-like task and write ``result.json``.

Contract (read by the climb harness):
    result.json = {"objective": <imagenet_linear_probe_top1 in [0,1]>,
                   "metrics": {"jepa_loss": ..., "probe_top1": ..., ...}}

The objective is the held-out linear-probe top-1 accuracy of the FROZEN encoder
features — the standard SSL evaluation. It genuinely responds to representation
quality, so better JEPA recipes (see config.py) climb higher.

Deterministic given config.SEED so the leaderboard compares like-for-like.
Runs on CPU in a few minutes; uses CUDA automatically when available.
"""

from __future__ import annotations

import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from data import make_dataset
from model import Encoder, Predictor


def cfg(name, default):
    return C.get(name, default)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_masks(num_patches: int, mask_ratio: float, gen: torch.Generator):
    """Random split of patch indices into (context, target) per the mask ratio."""
    n_tgt = max(1, min(num_patches - 1, int(round(num_patches * mask_ratio))))
    perm = torch.randperm(num_patches, generator=gen)
    tgt_idx = perm[:n_tgt].tolist()
    ctx_idx = perm[n_tgt:].tolist()
    return sorted(ctx_idx), sorted(tgt_idx)


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, m: float):
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(m).add_(op.data, alpha=1.0 - m)
    for tb, ob in zip(target.buffers(), online.buffers()):
        tb.data.copy_(ob.data)


def pretrain(device):
    img_size = cfg("img_size", C.IMG_SIZE)
    patch = cfg("patch_size", C.PATCH_SIZE)
    embed_dim = cfg("embed_dim", C.EMBED_DIM)
    depth = cfg("depth", C.DEPTH)
    heads = cfg("num_heads", C.NUM_HEADS)
    mlp_ratio = cfg("mlp_ratio", C.MLP_RATIO)
    dropout = cfg("dropout", C.DROPOUT)
    pred_dim = cfg("predictor_dim", C.PREDICTOR_DIM)
    pred_depth = cfg("predictor_depth", C.PREDICTOR_DEPTH)
    pred_heads = cfg("predictor_heads", C.PREDICTOR_HEADS)
    mask_ratio = cfg("mask_ratio", C.MASK_RATIO)
    ema_m = cfg("ema_momentum", C.EMA_MOMENTUM)
    epochs = cfg("epochs", C.EPOCHS)
    bs = cfg("batch_size", C.BATCH_SIZE)
    lr = cfg("lr", C.LR)
    wd = cfg("weight_decay", C.WEIGHT_DECAY)
    warmup_frac = cfg("warmup_frac", C.WARMUP_FRAC)

    x_train, y_train, x_test, y_test = make_dataset(
        num_classes=cfg("num_classes", C.NUM_CLASSES),
        img_size=img_size,
        per_class_train=cfg("per_class_train", C.PER_CLASS_TRAIN),
        per_class_test=cfg("per_class_test", C.PER_CLASS_TEST),
        noise=cfg("data_noise", C.DATA_NOISE),
        seed=cfg("seed", C.SEED),
    )
    xt = torch.from_numpy(x_train).to(device)
    n = xt.shape[0]

    online = Encoder(img_size, patch, 3, embed_dim, depth, heads, mlp_ratio, dropout).to(device)
    target = copy.deepcopy(online).to(device)
    for p in target.parameters():
        p.requires_grad_(False)
    num_patches = online.num_patches
    predictor = Predictor(
        embed_dim, pred_dim, pred_depth, pred_heads, num_patches, mlp_ratio, dropout
    ).to(device)

    params = list(online.parameters()) + list(predictor.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, n // bs)
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_at(step):
        if step < warmup_steps:
            return lr * step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * lr * (1 + np.cos(np.pi * prog))

    gen = torch.Generator().manual_seed(cfg("seed", C.SEED) + 7)
    step = 0
    last_loss = float("nan")
    online.train()
    predictor.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for b in range(steps_per_epoch):
            idx = perm[b * bs:(b + 1) * bs]
            imgs = xt[idx]
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            ctx_idx, tgt_idx = build_masks(num_patches, mask_ratio, gen)
            ctx_idx_t = torch.tensor(ctx_idx, device=device)
            tgt_idx_t = torch.tensor(tgt_idx, device=device)

            full = online(imgs)                     # (B, N, E)
            ctx = full[:, ctx_idx_t, :]
            with torch.no_grad():
                tgt_full = target(imgs)
                tgt = tgt_full[:, tgt_idx_t, :]
                tgt = F.layer_norm(tgt, (tgt.shape[-1],))  # stabilize targets
            pred = predictor(ctx, ctx_idx, tgt_idx)
            loss = F.smooth_l1_loss(pred, tgt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ema_update(target, online, ema_m)
            last_loss = float(loss.item())
            step += 1

    return target, (x_train, y_train, x_test, y_test), last_loss


@torch.no_grad()
def features(encoder, x_np, device, bs=256):
    encoder.eval()
    feats = []
    x = torch.from_numpy(x_np)
    for i in range(0, len(x), bs):
        out = encoder(x[i:i + bs].to(device))   # (B, N, E)
        feats.append(out.mean(dim=1).cpu())      # mean-pool patches
    return torch.cat(feats, 0)


def linear_probe(encoder, data, device):
    x_train, y_train, x_test, y_test = data
    ftr = features(encoder, x_train, device)
    fte = features(encoder, x_test, device)
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6
    ftr = (ftr - mu) / sd
    fte = (fte - mu) / sd
    yt = torch.from_numpy(y_train)
    ye = torch.from_numpy(y_test)

    num_classes = int(cfg("num_classes", C.NUM_CLASSES))
    clf = nn.Linear(ftr.shape[1], num_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=cfg("probe_lr", C.PROBE_LR), weight_decay=1e-4)
    for _ in range(int(cfg("probe_epochs", C.PROBE_EPOCHS))):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(clf(ftr), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = clf(fte).argmax(1)
        top1 = (pred == ye).float().mean().item()
    return top1


def main():
    t0 = time.time()
    set_seed(int(cfg("seed", C.SEED)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    target_encoder, data, jepa_loss = pretrain(device)
    top1 = linear_probe(target_encoder, data, device)

    result = {
        "objective": round(float(top1), 6),
        "metrics": {
            "imagenet_linear_probe_top1": round(float(top1), 6),
            "jepa_loss": round(float(jepa_loss), 6),
            "wall_seconds": round(time.time() - t0, 2),
            "params_encoder": int(sum(p.numel() for p in target_encoder.parameters())),
        },
    }
    out_path = os.environ.get("CLIMB_OBJECTIVE_PATH", "result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("RESULT", json.dumps(result))


if __name__ == "__main__":
    main()
