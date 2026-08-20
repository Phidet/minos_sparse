# MINOS CC/NC Event Classification (TorchSparse + PyG)

PyTorch pipeline for classifying MINOS neutrino events as Charged Current (CC)
vs. Neutral Current (NC), from raw ROOT files through sparse-CNN, GNN, and
transformer models.

---

## Quickstart

```bash
uv sync
uv run python train.py --models gnn_timing_aux
```

Or work interactively:

```bash
uv run jupyter notebook train.ipynb
```

`train.py` builds its dataset cache from the ROOT file on first run. To
preprocess ahead of time instead, use `scripts/preprocess_minos.py`.

---

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

## Evaluation & analysis tools (`scripts/`)

- `compare_models.py` — bootstrap-CI comparison between leaderboard runs;
  refuses to compare runs from incompatible dataset splits.
- `evaluate_by_energy.py` — accuracy/F1/ROC-AUC stratified by true neutrino energy.
- `evaluate_hit_head.py` — validates `HitSetTransformer`'s auxiliary per-hit head.
- `diagnose_timing_availability.py` — checks strip-timing feature availability vs. energy.
- `verify_hitset.py` — regression, structure, and physics checks for `feature_mode='hitset'`.
- `preprocess_minos.py` — caches a ROOT file into `.pt` datasets ahead of training.

## Current status

Best leaderboard result: `gnn_timing_aux` (`TimingAwareGNN` + auxiliary loss),
ROC-AUC 0.936 on the 11,925-event validation split. `model_leaderboard.csv`
holds runs comparable under the current dataset split; earlier runs from a
now-superseded split are archived in `model_leaderboard_archive.csv`.

---

## Code Base Structure

- [`src/dataset.py`](src/dataset.py): dataset builders (single-view, multi-view graph), feature modes (`sum`, `timing`, `hitset`), collators.
- [`src/models.py`](src/models.py): all model architectures.
- [`src/model_configs.py`](src/model_configs.py): `MODEL_CONFIGS` registry and `DATASET_CONFIG`.
- [`src/trainer.py`](src/trainer.py): training loop, metrics, checkpointing, leaderboard logging.
- [`train.py`](train.py): CLI entry point for single/multi-model training campaigns.
- [`train.ipynb`](train.ipynb): notebook harness for interactive experimentation.
- [`minos_event_display_2d.ipynb`](minos_event_display_2d.ipynb) / [`minos_event_display_3d.ipynb`](minos_event_display_3d.ipynb): event visualization notebooks.
