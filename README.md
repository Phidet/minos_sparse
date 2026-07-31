# MINOS U-View Sparse CNN (TorchSparse Engine)

A clean, high-performance PyTorch implementation of a **U-View Sparse Convolutional Neural Network** for MINOS Charged Current (CC) vs Neutral Current (NC) neutrino event display classification using **TorchSparse** (`torch_sparse`).

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

---

## Code Base Structure

- [`src/dataset.py`](file:///home/philip/UCL/minos/src/dataset.py): Single U-view (`planeview=2`) MINOS ROOT dataset parser and sparse collator.
- [`src/models.py`](file:///home/philip/UCL/minos/src/models.py): `SimpleUViewSparseCNN` model leveraging `torch_sparse` sparse matrix convolution & pooling.
- [`src/trainer.py`](file:///home/philip/UCL/minos/src/trainer.py): Streamlined training engine with weighted CrossEntropy, F1, Accuracy, and ROC-AUC metrics.
- [`u_view_sparse_cnn.ipynb`](file:///home/philip/UCL/minos/u_view_sparse_cnn.ipynb): End-to-end evaluation notebook.
