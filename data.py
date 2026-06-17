"""Synthetic ImageNet-like dataset for the JEPA scaling climb.

A self-contained, download-free stand-in for ImageNet-1K: K classes of small RGB
images, each class drawn from a distinct low-rank generative prototype (smooth
color/texture fields) plus per-sample noise. The structure is genuinely
learnable by self-supervised pretraining, so a linear probe on frozen JEPA
features lands well above the 1/K chance floor and *responds to model quality* —
better representation learning => higher ``imagenet_linear_probe_top1``.

Deterministic given the global seed, so every run measures the same task and the
leaderboard compares like-for-like. Returns float32 tensors in [0, 1].
"""

from __future__ import annotations

import numpy as np


def _class_prototypes(num_classes: int, img_size: int, rng: np.random.Generator) -> np.ndarray:
    """One smooth low-frequency RGB prototype field per class.

    Built as a sum of a few random 2-D sinusoids per channel (a low-rank, smooth
    structure that a patch encoder can pick up), normalized to [0, 1].
    """
    h = w = img_size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ys = ys / h
    xs = xs / w
    protos = np.zeros((num_classes, 3, h, w), dtype=np.float32)
    n_waves = 4
    for c in range(num_classes):
        for ch in range(3):
            field = np.zeros((h, w), dtype=np.float32)
            for _ in range(n_waves):
                fx, fy = rng.uniform(0.5, 3.0, size=2)
                phase = rng.uniform(0, 2 * np.pi)
                amp = rng.uniform(0.5, 1.0)
                field += amp * np.sin(2 * np.pi * (fx * xs + fy * ys) + phase)
            field -= field.min()
            field /= field.max() + 1e-8
            protos[c, ch] = field
    return protos


def make_dataset(
    num_classes: int = 16,
    img_size: int = 32,
    per_class_train: int = 64,
    per_class_test: int = 24,
    noise: float = 0.45,
    seed: int = 0,
):
    """Return ``(x_train, y_train, x_test, y_test)`` as numpy arrays.

    ``x`` is ``(N, 3, img_size, img_size)`` float32 in [0, 1]; ``y`` is ``(N,)`` int64.
    ``noise`` controls how hard the task is (more noise => lower achievable top-1).
    """
    rng = np.random.default_rng(seed)
    protos = _class_prototypes(num_classes, img_size, rng)

    def _sample(per_class: int, gen: np.random.Generator):
        xs, ys = [], []
        for c in range(num_classes):
            base = protos[c][None]  # (1, 3, H, W)
            n = per_class
            # Per-sample low-rank perturbation: scale + brightness + gaussian noise.
            scale = gen.uniform(0.7, 1.3, size=(n, 1, 1, 1)).astype(np.float32)
            bright = gen.uniform(-0.1, 0.1, size=(n, 1, 1, 1)).astype(np.float32)
            eps = gen.normal(0, noise, size=(n, 3, img_size, img_size)).astype(np.float32)
            imgs = np.clip(base * scale + bright + eps, 0.0, 1.0)
            xs.append(imgs)
            ys.append(np.full(n, c, dtype=np.int64))
        x = np.concatenate(xs, 0)
        y = np.concatenate(ys, 0)
        perm = gen.permutation(len(x))
        return x[perm], y[perm]

    x_train, y_train = _sample(per_class_train, np.random.default_rng(seed + 1))
    x_test, y_test = _sample(per_class_test, np.random.default_rng(seed + 2))
    return x_train, y_train, x_test, y_test
