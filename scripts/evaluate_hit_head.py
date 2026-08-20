#!/usr/bin/env python3
"""
Per-hit segmentation-head evaluation (scripts/evaluate_hit_head.py)
--------------------------------------------------------------------
Validates the HitSetTransformer's auxiliary primary-lepton head, which is the
part of the model that is checkable rather than opaque.

Reports:

1. Per-hit ROC-AUC of the predicted primary-lepton probability against thstp truth --
   i.e. did the head actually learn to localise the lepton, or is it just
   riding the event-level loss?

2. ROC-AUC of interpretable event variables *derived from the per-hit
   probabilities alone* (tagged-hit count, tagged plane span, chain
   contiguity, tagged charge fraction). These are the learned lepton-finder
   analogues of MINOS's published kNN inputs, and each one can be checked
   against data on control samples in a way an event score cannot.

3. The same variables computed from truth, as a ceiling. A truth oracle using
   tagged-hit count alone scores ROC-AUC 0.9901 on this file; the gap between
   the predicted and truth columns is how much of the available separation the
   head is actually recovering.

Usage:
    python3 scripts/evaluate_hit_head.py saved_models/transformer_hitset_aux_<ts>_<hash>.pt
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import MODEL_CONFIGS, DATASET_CONFIG, create_multiview_gnn_dataloaders  # noqa: E402
from src.dataset import get_hit_lepton_frac, get_has_lepton_truth  # noqa: E402
from scripts.evaluate_by_energy import (  # noqa: E402
    build_dataset_for_config,
    build_model_for_config,
)

from torch_geometric.utils import to_dense_batch  # noqa: E402

Z_IDX = 5                # z_norm column in the 14-feature hit vector
PE_E_IDX, PE_W_IDX = 0, 1
PLANES_PER_ZNORM = 15.0 / 0.0594


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the per-hit primary-lepton head")
    p.add_argument("checkpoint", type=str, help="Path to a transformer_hitset_aux checkpoint")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Per-hit probability threshold for 'tagged' (default 0.5)")
    return p.parse_args()


def chain_vars(tag: torch.Tensor, mask: torch.Tensor, dense: torch.Tensor):
    """Event variables derived from a boolean per-hit tag. Returns dict of [B] arrays."""
    tag = tag & mask
    n_tag = tag.sum(dim=1).float()

    z = dense[..., Z_IDX]
    plane = (z * PLANES_PER_ZNORM)
    big = torch.full_like(plane, float("inf"))
    small = torch.full_like(plane, float("-inf"))
    pmin = torch.where(tag, plane, big).min(dim=1).values
    pmax = torch.where(tag, plane, small).max(dim=1).values
    span = torch.where(n_tag > 0, (pmax - pmin).clamp(min=0.0), torch.zeros_like(n_tag))

    # contiguity: tagged hits per plane of span (1.0 => a dense unbroken chain)
    contig = torch.where(span > 0, n_tag / (span + 1.0), torch.zeros_like(n_tag))

    q = dense[..., PE_E_IDX] + dense[..., PE_W_IDX]
    q_tot = (q * mask).sum(dim=1)
    q_tag = (q * tag).sum(dim=1)
    q_frac = torch.where(q_tot > 0, q_tag / q_tot, torch.zeros_like(q_tot))

    return {
        "tagged hit count": n_tag,
        "tagged plane span": span,
        "chain contiguity": contig,
        "tagged charge frac": q_frac,
    }


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_type = ckpt["model_type"]
    config = MODEL_CONFIGS[model_type]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== {args.checkpoint} ===")
    print(f"model_type={model_type}  logged={ckpt.get('best_metrics', {})}")

    dataset = build_dataset_for_config(config)
    _, val_loader, _, _ = create_multiview_gnn_dataloaders(
        dataset,
        batch_size=config.get("batch_size", 16),
        val_split=DATASET_CONFIG.get("val_split", 0.20),
        random_seed=DATASET_CONFIG.get("random_seed", 42),
    )
    model = build_model_for_config(config, dataset)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    hit_p, hit_t = [], []
    pred_vars, true_vars, labels = {}, {}, []

    # HitSetTransformer returns a dense [B, N] map plus a padding mask (it pads
    # and truncates to max_hits); TimingAwareGNN returns node-indexed probs with
    # no truncation. Densify the GNN's output so the rest of the analysis is
    # layout-agnostic. This works because the timing 9-feature vector and the
    # hitset 14-feature vector share their first 9 columns by construction, so
    # the Z_IDX / PE_*_IDX offsets used by chain_vars are valid for both.
    max_hits = getattr(model, "max_hits", None)

    def hit_probs_and_mask(batch):
        out = model.predict_hit_lepton_prob(batch)
        if isinstance(out, tuple):          # transformer: (probs [B,N], mask [B,N])
            return out
        dense_p, m = to_dense_batch(        # GNN: [total_nodes] -> [B, N]
            out.unsqueeze(-1), batch.batch, max_num_nodes=max_hits)
        return dense_p.squeeze(-1), m

    for batch in val_loader:
        batch = batch.to(device)
        probs, mask = hit_probs_and_mask(batch)
        dense, _ = to_dense_batch(batch.x, batch.batch, max_num_nodes=max_hits)
        truth, _ = to_dense_batch(
            get_hit_lepton_frac(batch).unsqueeze(-1), batch.batch, max_num_nodes=max_hits
        )
        truth = truth.squeeze(-1)

        ok = mask & get_has_lepton_truth(batch).view(-1, 1)
        hit_p.append(probs[ok].cpu().numpy())
        hit_t.append((truth[ok] > 0.5).float().cpu().numpy())

        for name, v in chain_vars(probs > args.threshold, mask, dense).items():
            pred_vars.setdefault(name, []).append(v.cpu().numpy())
        for name, v in chain_vars(truth > 0.5, mask, dense).items():
            true_vars.setdefault(name, []).append(v.cpu().numpy())
        labels.append(batch.y.cpu().numpy())

    hit_p = np.concatenate(hit_p)
    hit_t = np.concatenate(hit_t)
    labels = np.concatenate(labels)

    print(f"\nvalidation events: {len(labels)}   hits scored: {len(hit_t):,}   "
          f"true primary-lepton hits: {hit_t.mean():.2%}")

    print("\n--- 1. per-hit localisation ---")
    if 0 < hit_t.mean() < 1:
        print(f"  per-hit ROC-AUC (pred vs thstp truth): {roc_auc_score(hit_t, hit_p):.4f}")
        tagged = hit_p > args.threshold
        tp = (tagged & (hit_t > 0.5)).sum()
        print(f"  at threshold {args.threshold}: purity={tp / max(1, tagged.sum()):.3f}  "
              f"efficiency={tp / max(1, (hit_t > 0.5).sum()):.3f}")
    else:
        print("  degenerate truth, skipping")

    print("\n--- 2. event ROC-AUC from derived lepton-chain variables ---")
    print(f"  {'variable':<22s} {'from prediction':>16s} {'from truth (ceiling)':>22s}")
    for name in pred_vars:
        pv = np.concatenate(pred_vars[name])
        tv = np.concatenate(true_vars[name])
        a_p = roc_auc_score(labels, pv) if len(np.unique(pv)) > 1 else float("nan")
        a_t = roc_auc_score(labels, tv) if len(np.unique(tv)) > 1 else float("nan")
        print(f"  {name:<22s} {a_p:>16.4f} {a_t:>22.4f}")

    print("\n  (event-level model ROC-AUC for reference: "
          f"{ckpt.get('best_metrics', {}).get('roc_auc', float('nan')):.4f})")


if __name__ == "__main__":
    main()
