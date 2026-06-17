"""Hyperparameters / levers for the JEPA ImageNet scaling climb.

THIS FILE IS THE MAIN SEARCH SPACE. A climb run implements a variant by editing
these defaults (and/or the model/training code), then a controlled run of
``python train.py`` measures ``imagenet_linear_probe_top1``. Any value here can
also be overridden at run time by a ``KNOB_<NAME_UPPER>`` environment variable
(set when the proposer pins a specific knob value), so edits and pinned knobs
both work.

Levers that matter most (roughly):
  - embed_dim / depth / num_heads     encoder capacity
  - predictor_dim / predictor_depth   predictor capacity (I-JEPA: a narrow pred.)
  - patch_size                        token granularity (32/patch_size)**2 tokens
  - mask_ratio                        fraction of patches predicted as targets
  - ema_momentum                      target-encoder EMA (representation stability)
  - epochs / lr / weight_decay / warmup_frac / batch_size   optimization
  - temperature                       (reserved) variance-regularization strength

Keep the model small enough that a full pretrain + linear probe finishes in a
few minutes on CPU; a GPU run is much faster.
"""

from __future__ import annotations

import os


# ---- data ------------------------------------------------------------------
NUM_CLASSES = 16
IMG_SIZE = 32
PER_CLASS_TRAIN = 64
PER_CLASS_TEST = 24
DATA_NOISE = 0.45
SEED = 0

# ---- encoder ---------------------------------------------------------------
PATCH_SIZE = 8          # 32/8 => 4x4 = 16 patches
EMBED_DIM = 64
DEPTH = 2
NUM_HEADS = 4
MLP_RATIO = 2.0
DROPOUT = 0.0

# ---- predictor (I-JEPA) ----------------------------------------------------
PREDICTOR_DIM = 48
PREDICTOR_DEPTH = 2
PREDICTOR_HEADS = 4

# ---- JEPA objective --------------------------------------------------------
MASK_RATIO = 0.5        # fraction of patches used as prediction targets
EMA_MOMENTUM = 0.996    # target-encoder EMA decay
TEMPERATURE = 1.0       # reserved (variance regularization weight)

# ---- optimization ----------------------------------------------------------
EPOCHS = 15
BATCH_SIZE = 128
LR = 1.5e-3
WEIGHT_DECAY = 0.04
WARMUP_FRAC = 0.1

# ---- linear probe ----------------------------------------------------------
PROBE_EPOCHS = 60
PROBE_LR = 1.0e-2


def _coerce(default, raw: str):
    if isinstance(default, bool):
        return raw.strip() not in ("0", "false", "False", "")
    if isinstance(default, int):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    return raw


def get(name: str, default):
    """Return a config value, letting a ``KNOB_<NAME>`` env var override it."""
    raw = os.environ.get("KNOB_" + name.upper())
    if raw is None or raw == "":
        return default
    try:
        return _coerce(default, raw)
    except (TypeError, ValueError):
        return default
