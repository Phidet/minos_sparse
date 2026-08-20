# MINOS CC/NC Event Classification

PyTorch pipeline for classifying MINOS neutrino events as Charged Current (CC)
vs. Neutral Current (NC) through sparse-CNN, GNN, and
transformer models.

---

## Quickstart

```bash
uv sync
uv run python train.py --sntp path/to/your_file.sntp.root --models gnn_timing_aux
```

`--sntp` points at your MINOS sntp ROOT file — there's no default or bundled
copy, so it's required. It's only actually opened if no cache exists yet for
the requested model's `feature_mode`; once a cache is built (in `data/cache/`)
subsequent runs read from it directly.

Then explore results — leaderboard, curves, event displays, ROC/confusion
matrix — in [`evaluate.ipynb`](evaluate.ipynb). The notebook takes the same
path via a `ROOT_FILEPATH` variable near the top (cell 2) — set it there.

## Models

Model configs live in `src/model_configs.py` (`MODEL_CONFIGS`); pick one with
`train.py --models <name>` or train everything with `--all`. Three families:

- **`cnn_*`** — TorchSparse dual-view sparse CNNs (dense, cross-attention,
  ResNet, transformer, and other fusion variants).
- **`gnn_*`** — hetero-graph models over both MINOS views plus shared
  `nexus` nodes; `gnn_timing*` variants add dual-ended strip-timing features.
- **`transformer_hitset*`** — `HitSetTransformer`, a per-hit set transformer
  with an auxiliary primary-lepton segmentation head.

All architectures are defined in `src/models.py`.
