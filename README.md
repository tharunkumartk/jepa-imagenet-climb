# JEPA ImageNet Scaling — climb scaffold

A small, self-contained **I-JEPA** (Image Joint-Embedding Predictive Architecture)
that self-supervised-pretrains on a synthetic ImageNet-like dataset and is
evaluated by a **linear probe** on frozen features. The objective is
`imagenet_linear_probe_top1` (held-out top-1 accuracy, in `[0, 1]`,
**maximize**). This is the scaffold a hill-climbing campaign optimizes: each run
implements one variant (a focused code/recipe change) and a controlled run of
`python train.py` measures the objective.

## Run

```bash
pip install -r requirements.txt
python train.py        # writes result.json: {"objective": <top1>, "metrics": {...}}
```

Runs in a few minutes on CPU; uses CUDA automatically when present. Deterministic
given `config.SEED`, so the leaderboard compares like-for-like.

## What's here

| File | Role |
| --- | --- |
| `train.py` | entrypoint: I-JEPA pretrain → linear probe → write `result.json` |
| `model.py` | tiny ViT encoder + I-JEPA predictor |
| `data.py` | synthetic ImageNet-like dataset (download-free, deterministic) |
| `config.py` | **the search space** — every lever, with `KNOB_<NAME>` env overrides |

## The levers (edit `config.py`, or override via `KNOB_<NAME>` env)

- **Encoder capacity**: `EMBED_DIM`, `DEPTH`, `NUM_HEADS`, `MLP_RATIO`, `DROPOUT`
- **Predictor (I-JEPA)**: `PREDICTOR_DIM`, `PREDICTOR_DEPTH`, `PREDICTOR_HEADS`
- **Tokenization**: `PATCH_SIZE` (token count = `(IMG_SIZE / PATCH_SIZE)**2`)
- **JEPA objective**: `MASK_RATIO` (target fraction), `EMA_MOMENTUM` (target EMA),
  `TEMPERATURE` (reserved for variance regularization)
- **Optimization**: `EPOCHS`, `BATCH_SIZE`, `LR`, `WEIGHT_DECAY`, `WARMUP_FRAC`
- **Probe**: `PROBE_EPOCHS`, `PROBE_LR`

Bigger/deeper encoders, a well-sized predictor, a sensible mask ratio (~0.4–0.7),
high EMA momentum, and enough epochs/warmup generally climb higher — but the
search is open-ended: improving the masking strategy, the prediction target
(e.g. multi-block targets, normalization), the architecture, or the optimizer
are all fair game, as long as `python train.py` still writes a valid
`result.json` and the objective measurement is never weakened or faked.

## Result contract

`train.py` must write `result.json` in the repo root:

```json
{"objective": 0.42, "metrics": {"imagenet_linear_probe_top1": 0.42, "jepa_loss": 0.01}}
```

`objective` is the value the campaign hill-climbs (higher = better).
