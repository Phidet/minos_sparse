#!/usr/bin/env python3
"""
Timing-feature availability diagnostic (scripts/diagnose_timing_availability.py)
----------------------------------------------------------------------------
Pure data analysis, no training. Tests the hypothesis that dual-ended strip
timing (the feature TimingAwareGNN relies on for its dt_scaled node/edge
features) is disproportionately unavailable for low-energy events -- which
would help explain why TimingAwareGNN's ROC-AUC gain over the best CNN is
concentrated in the 3-8+ GeV range while it actually underperforms the CNN
at 1-3 GeV (see scripts/evaluate_by_energy.py output).

Two things are measured, both binned by true neutrino energy:
  1. Among events kept in the cached "timing" dataset, what fraction of hits
     have both PMT ends valid (ph0>0 and ph1>0) -- i.e. what fraction of
     hits actually carry a real dt_scaled value vs. a zeroed one
     (src/dataset.py:659-662,674,710).
  2. What fraction of *all* events (read directly from the ROOT file) get
     silently dropped by the timing graph builder's n_hits<2 / empty-view
     guard (src/dataset.py:642-643,656-657), and whether that drop rate is
     energy-dependent.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import MINOSMultiViewGraphDataset, DATASET_CONFIG
from src.model_configs import MODEL_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose timing-feature availability vs. true neutrino energy")
    parser.add_argument(
        "--bins", type=str, default="0,1,2,3,5,8,100",
        help="Comma-separated true-energy bin edges in GeV (default: 0,1,2,3,5,8,100)"
    )
    return parser.parse_args()


def part1_hit_validity(bin_edges: np.ndarray):
    gnn_timing_cfg = MODEL_CONFIGS["gnn_timing"]
    cache_path = gnn_timing_cfg["cache_path"]

    print("Loading cached timing dataset...")
    ds = MINOSMultiViewGraphDataset(
        root_filepath=DATASET_CONFIG["root_filepath"],
        max_events=DATASET_CONFIG["max_events"],
        view_ids=DATASET_CONFIG["view_ids"],
        plane_radius=DATASET_CONFIG["plane_radius"],
        strip_radius=DATASET_CONFIG["strip_radius"],
        feature_mode="timing",
        cache_path=cache_path,
        allow_root_fallback=True,
    )
    print(f"Loaded {len(ds)} kept events (timing dataset)")

    kept_energy = np.zeros(len(ds), dtype=np.float64)
    kept_both_valid_frac = np.zeros(len(ds), dtype=np.float64)
    kept_n_hits = np.zeros(len(ds), dtype=np.int64)
    for i in range(len(ds)):
        ev = ds[i]
        x = ev.x  # [N, 9]: ... col 7 = valid_east, col 8 = valid_west
        valid_east = x[:, 7]
        valid_west = x[:, 8]
        both_valid = ((valid_east > 0) & (valid_west > 0)).float().mean().item()
        kept_energy[i] = float(ev.true_energy)
        kept_both_valid_frac[i] = both_valid
        kept_n_hits[i] = x.shape[0]

    bin_idx = np.clip(np.digitize(kept_energy, bin_edges) - 1, 0, len(bin_edges) - 2)

    print("\n=== Part 1: dual-ended (both-PMT-valid) hit fraction among KEPT timing events ===")
    header = f"{'E range (GeV)':>16} {'n_events':>9} {'mean_hits':>10} {'both_valid_frac':>17}"
    print(header)
    print("-" * len(header))
    for b in range(len(bin_edges) - 1):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            print(f"{bin_edges[b]:6.1f}-{bin_edges[b + 1]:<8.1f} {n:9d} {'n/a':>10} {'n/a':>17}")
            continue
        mean_hits = float(kept_n_hits[mask].mean())
        mean_frac = float(kept_both_valid_frac[mask].mean())
        print(f"{bin_edges[b]:6.1f}-{bin_edges[b + 1]:<8.1f} {n:9d} {mean_hits:10.2f} {mean_frac:17.4f}")


def part2_drop_rate(bin_edges: np.ndarray):
    print("\nReading raw ROOT file to compute the timing graph builder's drop rate...")
    rf = uproot.open(DATASET_CONFIG["root_filepath"])
    tree_key = "NtpSt" if "NtpSt" in rf else ("NtpSt;1" if "NtpSt;1" in rf else "sntp")
    tree = rf[tree_key]
    req_branches = [
        "NtpStRecord/stp/stp.planeview",
        "NtpStRecord/mc/mc.p4neu[4]",
    ]
    branches = tree.arrays(req_branches, entry_stop=DATASET_CONFIG["max_events"])
    view_a_id, view_b_id = DATASET_CONFIG["view_ids"]

    n = len(branches)
    all_energy = np.zeros(n, dtype=np.float64)
    all_dropped = np.zeros(n, dtype=bool)
    for i in range(n):
        all_energy[i] = float(np.asarray(branches["NtpStRecord/mc/mc.p4neu[4]"][i]).reshape(-1)[-1])
        views = np.array(branches["NtpStRecord/stp/stp.planeview"][i])
        mask_a = views == view_a_id
        mask_b = views == view_b_id
        n_hits = int(np.sum(mask_a | mask_b))
        all_dropped[i] = (np.sum(mask_a) == 0) or (np.sum(mask_b) == 0) or (n_hits < 2)

    print(f"Total events read: {n} | dropped by timing builder: {int(all_dropped.sum())} ({100 * all_dropped.mean():.2f}%)")

    bin_idx = np.clip(np.digitize(all_energy, bin_edges) - 1, 0, len(bin_edges) - 2)
    print("\n=== Part 2: event drop rate (timing graph builder) vs. true energy ===")
    header = f"{'E range (GeV)':>16} {'n_total':>8} {'n_dropped':>10} {'drop_rate':>10}"
    print(header)
    print("-" * len(header))
    for b in range(len(bin_edges) - 1):
        mask = bin_idx == b
        nb = int(mask.sum())
        if nb == 0:
            print(f"{bin_edges[b]:6.1f}-{bin_edges[b + 1]:<8.1f} {nb:8d} {'n/a':>10} {'n/a':>10}")
            continue
        nd = int(all_dropped[mask].sum())
        print(f"{bin_edges[b]:6.1f}-{bin_edges[b + 1]:<8.1f} {nb:8d} {nd:10d} {nd / nb:10.4f}")


def main():
    args = parse_args()
    bin_edges = np.array([float(x) for x in args.bins.split(",")])
    part1_hit_validity(bin_edges)
    part2_drop_rate(bin_edges)


if __name__ == "__main__":
    main()
