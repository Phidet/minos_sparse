#!/usr/bin/env python3
"""
Verification for feature_mode='hitset' (scripts/verify_hitset.py).

Three independent checks:

1. Regression: ``MINOSMultiViewGraphDataset._hit_time_features`` was extracted
   out of the inline block in ``_build_timing_graph`` so 'timing' and 'hitset'
   share it. This re-implements the ORIGINAL inline code and asserts the
   extracted helper is bit-identical, so the existing leaderboard results for
   gnn_timing* remain reproducible.

2. Structure: a small 'hitset' dataset builds, and every event has the
   expected shapes, dtypes, finite features and in-range labels.

3. Physics: per-hit primary-lepton labels behave (CC >> NC, contiguous chain).

Usage:  python scripts/verify_hitset.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import (  # noqa: E402
    MINOSMultiViewGraphDataset,
    get_hit_lepton_frac,
    get_has_lepton_truth,
)
from src.model_configs import DATASET_CONFIG  # noqa: E402

N_EVENTS = 400
FAILURES = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        FAILURES.append(msg)


def original_time_features(sel_time0, sel_time1, sel_views, view_a_id,
                           valid_east, valid_west):
    """Verbatim copy of the pre-extraction inline block, for regression only."""
    n_hits = len(sel_time0)
    t0_ns = sel_time0 * 1e9
    t1_ns = sel_time1 * 1e9
    both_valid = (valid_east > 0) & (valid_west > 0)
    t_mean_raw = np.zeros(n_hits, dtype=np.float64)
    dt_raw = np.zeros(n_hits, dtype=np.float64)
    if np.any(both_valid):
        t_mean_raw[both_valid] = (t0_ns[both_valid] + t1_ns[both_valid]) / 2.0
        dt_raw[both_valid] = t0_ns[both_valid] - t1_ns[both_valid]
    east_only = (valid_east > 0) & (valid_west == 0)
    west_only = (valid_west > 0) & (valid_east == 0)
    if np.any(east_only):
        t_mean_raw[east_only] = t0_ns[east_only]
    if np.any(west_only):
        t_mean_raw[west_only] = t1_ns[west_only]
    any_valid = (valid_east > 0) | (valid_west > 0)
    t0_event = float(np.median(t_mean_raw[any_valid])) if np.any(any_valid) else 0.0
    t_rel = t_mean_raw - t0_event
    view_sign = np.where(sel_views == view_a_id, 1.0, -1.0)
    dt_raw = dt_raw * view_sign
    t_scaled = np.clip(t_rel, -150.0, 150.0) / 150.0
    dt_scaled = np.clip(dt_raw, -300.0, 300.0) / 300.0
    no_valid = ~any_valid
    t_scaled[no_valid] = 0.0
    dt_scaled[no_valid] = 0.0
    dt_scaled[~both_valid] = 0.0
    return t_rel, t_scaled, dt_scaled


print("=" * 72)
print("1. REGRESSION: extracted _hit_time_features vs. original inline code")
print("=" * 72)

import uproot  # noqa: E402

rf = uproot.open(DATASET_CONFIG["root_filepath"])
tree = rf["NtpSt"]
raw = tree.arrays(
    [
        "NtpStRecord/stp/stp.planeview",
        "NtpStRecord/stp/stp.ph0.pe",
        "NtpStRecord/stp/stp.ph1.pe",
        "NtpStRecord/stp/stp.time0",
        "NtpStRecord/stp/stp.time1",
    ],
    entry_stop=N_EVENTS,
)

view_a_id, view_b_id = DATASET_CONFIG["view_ids"]
ds_shim = MINOSMultiViewGraphDataset.__new__(MINOSMultiViewGraphDataset)
max_abs_diff = 0.0
n_compared = 0
for i in range(N_EVENTS):
    views = np.asarray(raw["NtpStRecord/stp/stp.planeview"][i])
    if views.size == 0:
        continue
    m = (views == view_a_id) | (views == view_b_id)
    if not m.any():
        continue
    p0 = np.asarray(raw["NtpStRecord/stp/stp.ph0.pe"][i])[m]
    p1 = np.asarray(raw["NtpStRecord/stp/stp.ph1.pe"][i])[m]
    ve, vw = (p0 > 0).astype(np.float32), (p1 > 0).astype(np.float32)
    t0 = np.asarray(raw["NtpStRecord/stp/stp.time0"][i])[m]
    t1 = np.asarray(raw["NtpStRecord/stp/stp.time1"][i])[m]
    sv = views[m]

    new = ds_shim._hit_time_features(t0, t1, sv, view_a_id, ve, vw)
    old = original_time_features(t0, t1, sv, view_a_id, ve, vw)
    for a, b in zip(new[:3], old):
        max_abs_diff = max(max_abs_diff, float(np.max(np.abs(a - b))) if a.size else 0.0)
    n_compared += 1

check(max_abs_diff == 0.0,
      f"bit-identical over {n_compared} events (max abs diff = {max_abs_diff})")

print()
print("=" * 72)
print("2. STRUCTURE: hitset dataset builds and is well-formed")
print("=" * 72)

ds = MINOSMultiViewGraphDataset(
    root_filepath=DATASET_CONFIG["root_filepath"],
    max_events=N_EVENTS,
    view_ids=DATASET_CONFIG["view_ids"],
    plane_radius=DATASET_CONFIG["plane_radius"],
    strip_radius=DATASET_CONFIG["strip_radius"],
    feature_mode="hitset",
    cache_path=None,
    allow_root_fallback=True,
)
print(f"  built {len(ds)} events from {N_EVENTS} entries")

check(ds.in_channels == 14, f"in_channels == 14 (got {ds.in_channels})")

bad_shape = bad_finite = bad_range = bad_edge = 0
n_hits_all, lepton_all = [], []
for ev in ds:
    n = ev.x.shape[0]
    if ev.x.shape != (n, 14) or get_hit_lepton_frac(ev).shape != (n,):
        bad_shape += 1
    if not torch.isfinite(ev.x).all():
        bad_finite += 1
    if get_hit_lepton_frac(ev).min() < 0 or get_hit_lepton_frac(ev).max() > 1:
        bad_range += 1
    if getattr(ev, "edge_index", None) is not None:
        bad_edge += 1
    n_hits_all.append(n)
    lepton_all.append(float(get_hit_lepton_frac(ev).mean()))

check(bad_shape == 0, f"x is [N,14] and hit_lepton_frac is [N] for all events ({bad_shape} bad)")
check(bad_finite == 0, f"all features finite ({bad_finite} events with NaN/Inf)")
check(bad_range == 0, f"hit_lepton_frac within [0,1] ({bad_range} out of range)")
check(bad_edge == 0, f"no edge_index stored ({bad_edge} events carry one)")

# z-sorting: z_norm is feature index 5
unsorted = sum(1 for ev in ds if not torch.all(ev.x[1:, 5] >= ev.x[:-1, 5] - 1e-6))
check(unsorted == 0, f"hits sorted by z ({unsorted} events unsorted)")

# Batching contract used by the model
from torch_geometric.loader import DataLoader as PyGDataLoader  # noqa: E402
from torch_geometric.utils import to_dense_batch  # noqa: E402

loader = PyGDataLoader(ds, batch_size=16, shuffle=False)
b = next(iter(loader))
dense, mask = to_dense_batch(b.x, b.batch, max_num_nodes=256)
lepton_dense, _ = to_dense_batch(get_hit_lepton_frac(b).unsqueeze(-1), b.batch, max_num_nodes=256)
check(dense.shape[0] == 16 and dense.shape[2] == 14,
      f"to_dense_batch gives [16,N,14] (got {tuple(dense.shape)})")
check(lepton_dense.shape[:2] == dense.shape[:2],
      f"hit_lepton_frac densifies to match x (got {tuple(lepton_dense.shape)})")
check(get_has_lepton_truth(b).shape == (16,),
      f"has_lepton_truth batches to [B] (got {tuple(get_has_lepton_truth(b).shape)})")
check(b.y.shape == (16,), f"y batches to [B] (got {tuple(b.y.shape)})")

print()
print("=" * 72)
print("3. PHYSICS: per-hit primary-lepton labels")
print("=" * 72)

labels = np.array([int(ev.y) for ev in ds])
lepton_all = np.array(lepton_all)
truth_ok = np.array([bool(get_has_lepton_truth(ev)) for ev in ds])

cc = (labels == 1) & truth_ok
nc = (labels == 0) & truth_ok
cc_lep, nc_lep = lepton_all[cc].mean(), lepton_all[nc].mean()
print(f"  mean per-event lepton_frac:  CC {cc_lep:.4f}   NC {nc_lep:.4f}")
print(f"  events lacking lepton truth: {(~truth_ok).sum()} / {len(ds)} "
      f"({(~truth_ok).mean():.2%})")

check(cc_lep > 0.3, f"CC events carry substantial primary-lepton charge ({cc_lep:.4f} > 0.3)")
check(nc_lep < 0.01, f"NC events carry ~no primary-lepton charge ({nc_lep:.4f} < 0.01)")
check((~truth_ok).mean() < 0.03,
      f"lepton truth available for >97% of events ({truth_ok.mean():.2%})")

frac_cc_tagged = np.mean([
    float((get_hit_lepton_frac(ev) > 0.5).any()) for ev, k in zip(ds, cc) if k
])
check(frac_cc_tagged > 0.9,
      f"{frac_cc_tagged:.2%} of CC events have >=1 confidently-tagged hit (>90%)")

print()
print("=" * 72)
print("4. TIMING_AUX: graph equivalence with 'timing', and label agreement")
print("=" * 72)

ds_timing = MINOSMultiViewGraphDataset(
    root_filepath=DATASET_CONFIG["root_filepath"], max_events=N_EVENTS,
    view_ids=DATASET_CONFIG["view_ids"], plane_radius=DATASET_CONFIG["plane_radius"],
    strip_radius=DATASET_CONFIG["strip_radius"], feature_mode="timing",
    cache_path=None, allow_root_fallback=True,
)
ds_aux = MINOSMultiViewGraphDataset(
    root_filepath=DATASET_CONFIG["root_filepath"], max_events=N_EVENTS,
    view_ids=DATASET_CONFIG["view_ids"], plane_radius=DATASET_CONFIG["plane_radius"],
    strip_radius=DATASET_CONFIG["strip_radius"], feature_mode="timing_aux",
    cache_path=None, allow_root_fallback=True,
)
check(len(ds_timing) == len(ds_aux),
      f"same event count ({len(ds_timing)} vs {len(ds_aux)})")

# The graph must be bit-identical: this is what lets the already-trained
# gnn_timing_v1_on_v3 checkpoint serve as the control for the aux run.
graph_bad = 0
for a, b in zip(ds_timing, ds_aux):
    if not (torch.equal(a.x, b.x)
            and torch.equal(a.edge_index, b.edge_index)
            and torch.equal(a.edge_attr, b.edge_attr)
            and torch.equal(a.y, b.y)):
        graph_bad += 1
check(graph_bad == 0,
      f"timing_aux graphs bit-identical to timing on x/edge_index/edge_attr/y "
      f"({graph_bad} mismatched) -- required for the free control")

check(all(hasattr(e, "hit_lepton_frac") for e in ds_aux), "every timing_aux event carries hit_lepton_frac")
check(not any(hasattr(e, "hit_lepton_frac") for e in ds_timing), "plain 'timing' events carry no labels")
check(all(get_hit_lepton_frac(e).shape[0] == e.x.shape[0] for e in ds_aux),
      "hit_lepton_frac aligns with node count")

# timing_aux keeps combined_mask order; hitset sorts by (z, tpos). Same mask, so
# the label multisets must match event-for-event.
lab_bad = 0
for a, b in zip(ds_aux, ds):
    if a.x.shape[0] != b.x.shape[0]:
        lab_bad += 1
        continue
    if not torch.allclose(torch.sort(get_hit_lepton_frac(a)).values,
                          torch.sort(get_hit_lepton_frac(b)).values, atol=1e-6):
        lab_bad += 1
check(lab_bad == 0,
      f"timing_aux labels match hitset labels as multisets ({lab_bad} mismatched)")

aux_cc = np.array([float(get_hit_lepton_frac(e).mean()) for e in ds_aux
                   if int(e.y) == 1 and bool(get_has_lepton_truth(e))])
aux_nc = np.array([float(get_hit_lepton_frac(e).mean()) for e in ds_aux
                   if int(e.y) == 0 and bool(get_has_lepton_truth(e))])
print(f"  mean per-event lepton_frac:  CC {aux_cc.mean():.4f}   NC {aux_nc.mean():.4f}")
check(aux_cc.mean() > 0.3 and aux_nc.mean() < 0.01, "CC/NC label separation holds in timing_aux")

print()
print("=" * 72)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("ALL CHECKS PASSED")
print("=" * 72)
