# MINOS U-View Sparse Models (TorchSparse + PyG)

A clean PyTorch implementation for MINOS Charged Current (CC) vs Neutral Current (NC) neutrino event display classification. The repository now supports both the original **U-view sparse CNN** path and a minimum viable **multi-view hetero-GNN** path built with **PyTorch Geometric**.

---

## Quickstart

### 1. Environment Setup with `uv`
```bash
/home/philip/.local/bin/uv venv
/home/philip/.local/bin/uv sync
```

### 2. Launch Jupyter Notebook
```bash
/home/philip/.local/bin/uv run jupyter notebook u_view_sparse_cnn.ipynb
```

### 3. Choose a model path

Set `CONFIG["model_type"]` in the notebook to either `cnn` or `gnn`.

- `cnn`: the original single-view TorchSparse baseline.
- `gnn`: a small hetero-GNN that consumes both MINOS views plus shared nexus nodes.

---

## Code Base Structure

- [`src/dataset.py`](file:///home/philip/UCL/minos/src/dataset.py): Single U-view dataset, sparse collator, and a new multi-view graph dataset for PyG.
- [`src/models.py`](file:///home/philip/UCL/minos/src/models.py): `SimpleUViewSparseCNN` plus a minimum viable hetero-GNN classifier.
- [`src/trainer.py`](file:///home/philip/UCL/minos/src/trainer.py): Shared training engine with weighted CrossEntropy, F1, Accuracy, and ROC-AUC metrics.
- [`u_view_sparse_cnn.ipynb`](file:///home/philip/UCL/minos/u_view_sparse_cnn.ipynb): Notebook experiment harness with `cnn` and `gnn` switching.

## GNN Notes

The GNN path uses a simple heterograph design:

- one node type per MINOS view,
- a shared `nexus` node type keyed by plane coordinate,
- intra-view edges built from local plane/strip proximity,
- hit-to-nexus edges for cross-view coupling.

This is intentionally small and readable so it can serve as an MVP before experimenting with more advanced graph construction.
