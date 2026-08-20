#!/usr/bin/env python3
"""
Paired model comparison with bootstrap error bars (scripts/compare_models.py)
------------------------------------------------------------------------------
model_leaderboard.csv reports a single ROC-AUC per run with no uncertainty, and
the top of the board spans ~0.004 -- smaller than the resolution of the
validation set. This script supplies the missing error bars *without retraining*,
because the dominant uncertainty here is finite validation data, not seed jitter.

Two things it does that the leaderboard cannot:

1. **Refuses incomparable comparisons.** Runs in this repo fall into two families
   depending on which dataset builder produced them:

       both views non-empty (_build_event_graph)  -> 11,694 events
       n_hits >= 1        (v3 fix, and 'hitset')  -> 11,925 events

   Within a family the validation split is identical (fixed
   Generator().manual_seed(42) over an equal-length, identically-filtered
   dataset), so runs are directly comparable. Across families they are not, and
   the 11,694 set is *easier* -- it drops the 231 sparse single-view events.
   Comparing across families silently produces a meaningless number, which is
   exactly how the apparent gnn_timing_v2 -> v3 "regression" arose. This script
   asserts a shared validation set and refuses otherwise.

2. **Paired bootstrap on the difference.** Resampling the *same* validation
   events for both models and taking the distribution of dAUC exploits the fact
   that the models' errors are highly correlated, giving a far tighter interval
   than comparing two independent CIs. For scale, the Hanley-McNeil SE on a
   single AUC at n~2,385 and 75% CC is ~0.005 -- already larger than most gaps
   being ranked -- so the paired difference is the only statistic here with real
   resolving power.

Usage:
    python3 scripts/compare_models.py \
        saved_models/transformer_hitset_aux_20260811_111730_manual-run.pt \
        saved_models/transformer_hitset_20260810_165837_manual-run.pt \
        saved_models/gnn_timing_v3_20260807_165729_manual-run.pt
"""

import sys
import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import MODEL_CONFIGS, DATASET_CONFIG, create_multiview_gnn_dataloaders, run_inference  # noqa: E402
from scripts.evaluate_by_energy import build_dataset_for_config, build_model_for_config  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Paired bootstrap comparison of saved MINOS checkpoints")
    p.add_argument("checkpoints", nargs="+", type=str, help="Paths to saved_models/*.pt")
    p.add_argument("--n-boot", type=int, default=10000, help="Bootstrap resamples (default 10000)")
    p.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed (default 0)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--allow-mismatch", action="store_true",
                   help="Bypass the shared-validation-set guard (produces meaningless numbers)")
    return p.parse_args()


# Datasets are memoised by cache_path: the timing caches are ~1.2 GB each, and
# comparing several checkpoints that share one would otherwise reload it per model.
_DATASET_CACHE = {}


def _get_dataset(config):
    key = config.get("cache_path", DATASET_CONFIG["cache_path"])
    if key not in _DATASET_CACHE:
        print(f"  loading dataset for cache_path={key} ...")
        _DATASET_CACHE[key] = build_dataset_for_config(config)
    return _DATASET_CACHE[key]


def collect_predictions(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type = ckpt["model_type"]
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_type '{model_type}' in {checkpoint_path}")
    config = MODEL_CONFIGS[model_type]

    dataset = _get_dataset(config)
    _, val_loader, _, val_idx = create_multiview_gnn_dataloaders(
        dataset,
        batch_size=config.get("batch_size", 32),
        val_split=DATASET_CONFIG.get("val_split", 0.20),
        random_seed=DATASET_CONFIG.get("random_seed", 42),
    )
    model = build_model_for_config(config, dataset)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    probs, targets, _ = run_inference(model, val_loader, device=device)
    return {
        "path": checkpoint_path,
        "name": Path(checkpoint_path).stem,
        "model_type": model_type,
        "n_events": len(dataset),
        "val_idx": tuple(val_idx),
        "probs": probs,
        "targets": targets,
        "logged_auc": float(ckpt.get("best_metrics", {}).get("roc_auc", float("nan"))),
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def boot_indices(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(n_boot, n))


def auc_over_resamples(y: np.ndarray, p: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.empty(idx.shape[0])
    for i, take in enumerate(idx):
        yy = y[take]
        # A resample can be single-class; AUC undefined there.
        out[i] = roc_auc_score(yy, p[take]) if yy.min() != yy.max() else np.nan
    return out


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}   bootstrap resamples: {args.n_boot}\n")

    results = [collect_predictions(c, device) for c in args.checkpoints]

    # ── Comparability guard ────────────────────────────────────────────
    print("=" * 78)
    print("VALIDATION SET COMPARABILITY")
    print("=" * 78)
    for r in results:
        print(f"  {r['model_type']:<26s} dataset={r['n_events']:>6d} events  "
              f"val={len(r['targets']):>5d}")
    families = {r["n_events"] for r in results}
    if len(families) > 1:
        msg = (f"\nCheckpoints span {len(families)} different dataset sizes {sorted(families)}. "
               "They do NOT share a validation set and cannot be compared -- the smaller set "
               "excludes the sparse single-view events and is systematically easier.")
        if not args.allow_mismatch:
            raise SystemExit(msg + "\nRefusing. Re-train onto a common event set, or pass "
                                   "--allow-mismatch to override (results will be meaningless).")
        print(msg + "\n  [--allow-mismatch given, continuing anyway]")
    else:
        ref = results[0]
        for r in results[1:]:
            if r["val_idx"] != ref["val_idx"]:
                raise SystemExit("Same dataset size but different validation indices -- "
                                 "the split seed must have changed. Refusing.")
            if not np.array_equal(r["targets"], ref["targets"]):
                raise SystemExit("Validation labels differ between checkpoints. Refusing.")
        print(f"\n  OK: all {len(results)} checkpoints share an identical validation set "
              f"({len(ref['targets'])} events, {ref['targets'].mean():.1%} CC)")

    y = results[0]["targets"]
    n = len(y)
    rng = np.random.default_rng(args.seed)
    idx = boot_indices(n, args.n_boot, rng)

    # ── Per-model AUC with CI ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PER-MODEL ROC-AUC (95% bootstrap CI)")
    print("=" * 78)
    print(f"  {'model':<26s} {'params':>9s} {'AUC':>8s} {'95% CI':>18s} {'logged':>8s}")
    for r in results:
        d = auc_over_resamples(y, r["probs"], idx)
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        r["auc"] = roc_auc_score(y, r["probs"])
        r["boot"] = d
        print(f"  {r['model_type']:<26s} {r['params']:>9,d} {r['auc']:>8.4f} "
              f"  [{lo:.4f}, {hi:.4f}] {r['logged_auc']:>8.4f}")
    print(f"\n  (single-AUC CI half-width ~{(hi - lo) / 2:.4f} -- this is the resolution of the "
          f"validation set,\n   and is why the paired differences below are the meaningful statistic.)")

    # ── Paired differences ─────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PAIRED dAUC (same resampled events for both models)")
    print("=" * 78)
    print(f"  {'comparison':<48s} {'dAUC':>8s} {'95% CI':>18s} {'p':>7s}")
    for a, b in itertools.combinations(results, 2):
        diff = a["boot"] - b["boot"]
        diff = diff[~np.isnan(diff)]
        point = a["auc"] - b["auc"]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        # Two-sided bootstrap p: how often the resampled difference crosses zero.
        p = 2.0 * min((diff <= 0).mean(), (diff >= 0).mean())
        p = min(1.0, max(p, 1.0 / len(diff)))
        star = "  *" if lo > 0 or hi < 0 else ""
        print(f"  {a['model_type'] + ' - ' + b['model_type']:<48s} {point:>+8.4f} "
              f"  [{lo:+.4f}, {hi:+.4f}] {p:>7.4f}{star}")
    print("\n  * = 95% CI excludes zero")


if __name__ == "__main__":
    main()
