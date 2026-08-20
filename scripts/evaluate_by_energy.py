#!/usr/bin/env python3
"""
Energy-stratified evaluation (scripts/evaluate_by_energy.py)
--------------------------------------------------------------
Loads one or more saved MINOS checkpoints, rebuilds the exact validation
split used at training time, and reports accuracy / F1 / ROC-AUC / per-class
recall binned by true neutrino energy. This is the diagnostic the repo was
missing: model_leaderboard.csv only logs a single aggregate metric per run,
so there has never been a direct measurement of "how much worse is this
model at low energy" or "does TimingAwareGNN actually help there".

The decision threshold used for every bin is the single threshold recorded
in the checkpoint's best_metrics (i.e. the threshold chosen on the *whole*
validation set) — bins are NOT given their own re-optimized threshold, since
low-energy bins have few events and a per-bin-optimal threshold would just
be overfitting noise, not a real effect.

Usage:
    python3 scripts/evaluate_by_energy.py \
        saved_models/gnn_timing_20260806_154127_e1cf805.pt \
        saved_models/cnn_deep_resnet_cross_attention_<timestamp>_<hash>.pt \
        --bins 0,1,2,3,5,8,100
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (
    MODEL_CONFIGS,
    DATASET_CONFIG,
    MINOSMultiViewGraphDataset,
    create_multiview_gnn_dataloaders,
    run_inference,
    compute_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Energy-stratified evaluation of saved MINOS checkpoints")
    parser.add_argument("checkpoints", nargs="+", type=str, help="Path(s) to saved_models/*.pt checkpoint files")
    parser.add_argument(
        "--bins", type=str, default="0,1,2,3,5,8,100",
        help="Comma-separated true-energy bin edges in GeV (default: 0,1,2,3,5,8,100)"
    )
    parser.add_argument("--device", type=str, default=None, help="Override device (default: cuda if available)")
    return parser.parse_args()


def build_dataset_for_config(config):
    feature_mode = config.get("feature_mode", DATASET_CONFIG["feature_mode"])
    cache_path = config.get("cache_path", DATASET_CONFIG["cache_path"])
    return MINOSMultiViewGraphDataset(
        root_filepath=DATASET_CONFIG["root_filepath"],
        max_events=DATASET_CONFIG["max_events"],
        view_ids=DATASET_CONFIG["view_ids"],
        plane_radius=DATASET_CONFIG["plane_radius"],
        strip_radius=DATASET_CONFIG["strip_radius"],
        feature_mode=feature_mode,
        cache_path=cache_path,
        allow_root_fallback=DATASET_CONFIG.get("allow_root_fallback", True),
        ablate_timing=config.get("ablate_timing", False),
    )


def build_model_for_config(config, dataset):
    sample_graph = dataset[0]
    if config.get("is_gnn", False):
        if hasattr(sample_graph, "metadata"):
            metadata = sample_graph.metadata()
            in_channels = sample_graph["view_a"].x.size(-1)
        else:
            metadata = None
            in_channels = sample_graph.x.size(-1)
        return config["model_factory"](metadata, in_channels)
    return config["model"]


def evaluate_checkpoint(checkpoint_path: str, bin_edges: np.ndarray, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type = checkpoint["model_type"]
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_type '{model_type}' in checkpoint {checkpoint_path}")
    config = MODEL_CONFIGS[model_type]

    print(f"\n=== {checkpoint_path} ===")
    print(f"model_type={model_type} | model_name={checkpoint.get('config', {}).get('model_name', '?')}")
    logged_metrics = checkpoint.get("best_metrics", {})
    threshold = float(logged_metrics.get("threshold", 0.5))
    print(f"logged best_metrics: {logged_metrics}")
    print(f"using fixed decision threshold={threshold:.3f} for all bins")

    dataset = build_dataset_for_config(config)
    _, val_loader, _, _ = create_multiview_gnn_dataloaders(
        dataset,
        batch_size=config.get("batch_size", 32),
        val_split=DATASET_CONFIG.get("val_split", 0.20),
        random_seed=DATASET_CONFIG.get("random_seed", 42),
    )

    model = build_model_for_config(config, dataset)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    probs, targets, energies = run_inference(model, val_loader, device=device)
    preds = (probs >= threshold).astype(int)

    bin_idx = np.digitize(energies, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)

    rows = []
    for b in range(len(bin_edges) - 1):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append((bin_edges[b], bin_edges[b + 1], 0, 0, 0, None, None, None, None, None))
            continue
        y = targets[mask]
        p = probs[mask]
        yhat = preds[mask]
        n_cc = int(y.sum())
        n_nc = n - n_cc
        acc = float((yhat == y).mean())
        try:
            m = compute_metrics(y, p, preds=yhat)
            auc = m["roc_auc"] if len(np.unique(y)) > 1 else None
            f1 = m["f1"]
        except Exception:
            auc, f1 = None, None
        cc_recall = float(recall_score(y, yhat, pos_label=1, zero_division=0)) if n_cc > 0 else None
        nc_recall = float(recall_score(y, yhat, pos_label=0, zero_division=0)) if n_nc > 0 else None
        rows.append((bin_edges[b], bin_edges[b + 1], n, n_cc, n_nc, acc, f1, auc, cc_recall, nc_recall))

    header = f"{'E range (GeV)':>16} {'n':>6} {'n_CC':>6} {'n_NC':>6} {'acc':>7} {'f1':>7} {'roc_auc':>8} {'CC recall':>10} {'NC recall':>10}"
    print(header)
    print("-" * len(header))
    for lo, hi, n, n_cc, n_nc, acc, f1, auc, cc_r, nc_r in rows:
        def fmt(v):
            return f"{v:.4f}" if v is not None else "n/a"
        print(f"{lo:6.1f}-{hi:<8.1f} {n:6d} {n_cc:6d} {n_nc:6d} {fmt(acc):>7} {fmt(f1):>7} {fmt(auc):>8} {fmt(cc_r):>10} {fmt(nc_r):>10}")

    return rows


def main():
    args = parse_args()
    bin_edges = np.array([float(x) for x in args.bins.split(",")])
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for ckpt in args.checkpoints:
        evaluate_checkpoint(ckpt, bin_edges, device)


if __name__ == "__main__":
    main()
