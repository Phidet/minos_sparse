from typing import Tuple, List, Optional, Sequence, Dict
from pathlib import Path
import numpy as np
import uproot
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from torch_geometric.data import HeteroData, Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:
    HeteroData = None
    Data = None
    PyGDataLoader = None


_CACHE_MAGIC = "minos_sparse_cache_v2"


def _load_cache_file(cache_path: Path) -> dict:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict) or cache.get("cache_magic") != _CACHE_MAGIC:
        raise ValueError(f"Invalid cache file: {cache_path}")
    return cache


def _save_cache_file(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cache_magic": _CACHE_MAGIC, **payload}, cache_path)


class MINOSSingleViewDataset(Dataset):
    """
    Lightweight Dataset parser for single-view (U-view) MINOS event displays.
    Target view = 2 (U-view).
    Target label: CC (1) vs NC (0) based on iaction branch.

    Supported feature_mode options:
      - 'sum': Single channel [log1p(ph0 + ph1)]
      - 'dual_ph': 2 channels [log1p(ph0), log1p(ph1)]
      - 'ph_full': 4 channels [log1p(ph0 + ph1), log1p(ph0), log1p(ph1), asymmetry_ratio]
    """

    def __init__(
        self,
        root_filepath: str,
        max_events: Optional[int] = None,
        target_view: int = 2,
        feature_mode: str = "dual_ph",
        cache_path: Optional[str] = None,
        allow_root_fallback: bool = True,
    ):
        super().__init__()
        self.root_filepath = Path(root_filepath)
        self.target_view = target_view
        self.feature_mode = str(feature_mode)
        self.events = []

        cache_file = Path(cache_path) if cache_path is not None else None
        if cache_file is not None and cache_file.exists():
            cache = _load_cache_file(cache_file)
            if cache.get("kind") != "single_view":
                raise ValueError(f"Cache {cache_file} does not contain single-view events")

            cached_feature_mode = cache.get("feature_mode", "sum")
            cached_max_events = cache.get("max_events", None)
            cached_events = cache["events"]

            if cached_feature_mode == self.feature_mode:
                if max_events is not None and len(cached_events) < max_events and cached_max_events is not None and cached_max_events < max_events:
                    if not allow_root_fallback:
                        raise ValueError(
                            f"Cache {cache_file} contains {len(cached_events)} events (capped at {cached_max_events}), "
                            f"but requested max_events={max_events} and allow_root_fallback=False."
                        )
                else:
                    self.root_filepath = Path(cache.get("root_filepath", self.root_filepath))
                    self.target_view = int(cache.get("target_view", self.target_view))
                    self.events = cached_events[:max_events] if max_events is not None else cached_events
                    return
            elif not allow_root_fallback:
                raise ValueError(
                    f"Cache {cache_file} has feature_mode='{cached_feature_mode}', "
                    f"but requested feature_mode='{self.feature_mode}' and allow_root_fallback=False."
                )

        if cache_file is not None and not allow_root_fallback:
            raise FileNotFoundError(f"Cache file not found: {cache_file}")

        if not allow_root_fallback:
            raise ValueError("cache_path is required when allow_root_fallback is False")

        if not self.root_filepath.exists():
            raise FileNotFoundError(f"ROOT file not found: {self.root_filepath}")

        rf = uproot.open(self.root_filepath)
        tree_key = "NtpSt" if "NtpSt" in rf else ("NtpSt;1" if "NtpSt;1" in rf else "sntp")
        tree = rf[tree_key]

        req_branches = [
            "NtpStRecord/stp/stp.planeview",
            "NtpStRecord/stp/stp.strip",
            "NtpStRecord/stp/stp.plane",
            "NtpStRecord/stp/stp.ph0.pe",
            "NtpStRecord/stp/stp.ph1.pe",
            "NtpStRecord/mc/mc.iaction",
            "NtpStRecord/mc/mc.p4neu[4]",
        ]

        branches = tree.arrays(
            req_branches,
            entry_stop=max_events
        )

        for i in range(len(branches)):
            iaction = branches["NtpStRecord/mc/mc.iaction"][i]
            label = 1 if iaction == 1 else 0
            true_energy = float(np.asarray(branches["NtpStRecord/mc/mc.p4neu[4]"][i]).reshape(-1)[-1])

            views = np.array(branches["NtpStRecord/stp/stp.planeview"][i])
            planes = np.array(branches["NtpStRecord/stp/stp.plane"][i])
            strips = np.array(branches["NtpStRecord/stp/stp.strip"][i])
            ph0 = np.array(branches["NtpStRecord/stp/stp.ph0.pe"][i])
            ph1 = np.array(branches["NtpStRecord/stp/stp.ph1.pe"][i])
            phs = ph0 + ph1

            mask = (views == self.target_view)
            if not np.any(mask):
                continue

            sel_planes = planes[mask]
            sel_strips = strips[mask]
            sel_ph0 = ph0[mask]
            sel_ph1 = ph1[mask]
            sel_phs = phs[mask]

            if self.feature_mode == "dual_ph":
                norm_ph0 = np.log1p(np.maximum(0.0, sel_ph0)).astype(np.float32)
                norm_ph1 = np.log1p(np.maximum(0.0, sel_ph1)).astype(np.float32)
                feats = np.column_stack((norm_ph0, norm_ph1))
            elif self.feature_mode == "ph_full":
                norm_sum = np.log1p(np.maximum(0.0, sel_phs)).astype(np.float32)
                norm_ph0 = np.log1p(np.maximum(0.0, sel_ph0)).astype(np.float32)
                norm_ph1 = np.log1p(np.maximum(0.0, sel_ph1)).astype(np.float32)
                denom = sel_phs + 1e-5
                asym = ((sel_ph0 - sel_ph1) / denom).astype(np.float32)
                feats = np.column_stack((norm_sum, norm_ph0, norm_ph1, asym))
            else:
                norm_phs = np.log1p(np.maximum(0.0, sel_phs)).astype(np.float32)
                feats = norm_phs[:, np.newaxis]

            coords = np.column_stack((sel_planes, sel_strips)).astype(np.int64)

            self.events.append({
                "coords": torch.tensor(coords, dtype=torch.long),
                "feats": torch.tensor(feats, dtype=torch.float32),
                "label": torch.tensor(label, dtype=torch.long),
                "true_energy": torch.tensor(true_energy, dtype=torch.float32),
            })

        if cache_file is not None:
            _save_cache_file(
                cache_file,
                {
                    "kind": "single_view",
                    "root_filepath": str(self.root_filepath),
                    "target_view": self.target_view,
                    "feature_mode": self.feature_mode,
                    "max_events": max_events,
                    "events": self.events,
                },
            )

    @property
    def in_channels(self) -> int:
        if self.feature_mode == "dual_ph":
            return 2
        elif self.feature_mode == "ph_full":
            return 4
        return 1

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> dict:
        item = self.events[idx]
        return {
            "coords": item["coords"].clone(),
            "feats": item["feats"].clone(),
            "label": item["label"].clone() if isinstance(item["label"], torch.Tensor) else torch.tensor(item["label"], dtype=torch.long),
            "true_energy": item["true_energy"].clone() if isinstance(item["true_energy"], torch.Tensor) else torch.tensor(item["true_energy"], dtype=torch.float32),
        }

    def get_class_weights(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        labels = [e["label"].item() for e in self.events]
        total = len(labels)
        cc_count = sum(labels)
        nc_count = total - cc_count
        w_nc = total / (2.0 * max(1, nc_count))
        w_cc = total / (2.0 * max(1, cc_count))
        return torch.tensor([w_nc, w_cc], dtype=torch.float32, device=device)


def sparse_uview_collate_fn(batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collates sparse event display hits into batch tensor [batch_idx, plane, strip]."""
    coords_list = []
    feats_list = []
    labels_list = []
    energies_list = []

    for batch_idx, item in enumerate(batch):
        c = item["coords"]
        f = item["feats"]
        lbl = item["label"]
        eng = item["true_energy"]

        batch_idx_col = torch.full((c.shape[0], 1), batch_idx, dtype=torch.long)
        b_coords = torch.cat([batch_idx_col, c], dim=1)

        coords_list.append(b_coords)
        feats_list.append(f)
        labels_list.append(lbl)
        energies_list.append(eng)

    batch_coords = torch.cat(coords_list, dim=0)
    batch_feats = torch.cat(feats_list, dim=0)
    batch_labels = torch.stack(labels_list, dim=0)
    batch_energies = torch.stack(energies_list, dim=0)

    return batch_coords, batch_feats, batch_labels, batch_energies



def create_uview_dataloaders(
    dataset: MINOSSingleViewDataset,
    batch_size: int = 32,
    val_split: float = 0.20,
    random_seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(random_seed)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sparse_uview_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_uview_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_ds.indices, val_ds.indices


class MINOSMultiViewGraphDataset(Dataset):
    """
    Multi-view MINOS graph dataset for PyG.

    Each event becomes a heterograph with one node type per view and a shared
    nexus node type keyed by plane coordinate. Supports standard radius edge
    construction and Delaunay/KNN mesh edge construction.

    Supported feature_mode options:
      - 'sum': 1 hit feature [log1p(ph0 + ph1)]
      - 'dual_ph': 2 hit features [log1p(ph0), log1p(ph1)]
      - 'ph_full': 4 hit features [log1p(ph0 + ph1), log1p(ph0), log1p(ph1), asymmetry_ratio]
      - 'timing': 9 hit features with timing/spatial info, merged homogeneous graph
                  [PE_east_log, PE_west_log, t_scaled, dt_scaled, tpos_norm, z_norm,
                   view_flag, readout_valid_east, readout_valid_west]
    """

    def __init__(
        self,
        root_filepath: str,
        max_events: Optional[int] = None,
        view_ids: Optional[Sequence[int]] = None,
        plane_radius: int = 1,
        strip_radius: int = 2,
        graph_mode: str = "radius",
        feature_mode: str = "dual_ph",
        cache_path: Optional[str] = None,
        allow_root_fallback: bool = True,
    ):
        super().__init__()

        self.root_filepath = Path(root_filepath)
        self.plane_radius = int(plane_radius)
        self.strip_radius = int(strip_radius)
        self.graph_mode = str(graph_mode)
        self.feature_mode = str(feature_mode)
        self.events = []

        cache_file = Path(cache_path) if cache_path is not None else None
        if cache_file is not None and cache_file.exists():
            cache = _load_cache_file(cache_file)
            if cache.get("kind") != "multi_view_graph":
                raise ValueError(f"Cache {cache_file} does not contain multi-view graph events")

            cached_view_ids = tuple(cache.get("view_ids", ()))
            cached_feature_mode = cache.get("feature_mode", "sum")
            cached_max_events = cache.get("max_events", None)
            cached_events = cache["events"]
            valid_view_ids = (view_ids is None or tuple(int(v) for v in view_ids) == cached_view_ids)
            valid_mode = (cached_feature_mode == self.feature_mode)

            if valid_view_ids and valid_mode:
                if max_events is not None and len(cached_events) < max_events and cached_max_events is not None and cached_max_events < max_events:
                    if not allow_root_fallback:
                        raise ValueError(
                            f"Cache {cache_file} contains {len(cached_events)} events (capped at {cached_max_events}), "
                            f"but requested max_events={max_events} and allow_root_fallback=False."
                        )
                else:
                    self.root_filepath = Path(cache.get("root_filepath", self.root_filepath))
                    self.view_ids = cached_view_ids
                    self.plane_radius = int(cache.get("plane_radius", self.plane_radius))
                    self.strip_radius = int(cache.get("strip_radius", self.strip_radius))
                    self.graph_mode = str(cache.get("graph_mode", self.graph_mode))
                    self.events = cached_events[:max_events] if max_events is not None else cached_events
                    return
            elif not allow_root_fallback:
                raise ValueError(
                    f"Cache {cache_file} invalid for view_ids={view_ids}, feature_mode={self.feature_mode} "
                    f"and allow_root_fallback=False."
                )

        if cache_file is not None and not allow_root_fallback:
            raise FileNotFoundError(f"Cache file not found: {cache_file}")

        if not allow_root_fallback:
            raise ValueError("cache_path is required when allow_root_fallback is False")

        if not self.root_filepath.exists():
            raise FileNotFoundError(f"ROOT file not found: {self.root_filepath}")

        if HeteroData is None:
            raise ImportError(
                "torch_geometric is required for MINOSMultiViewGraphDataset. "
                "Install PyTorch Geometric to use the GNN path."
            )

        rf = uproot.open(self.root_filepath)
        tree_key = "NtpSt" if "NtpSt" in rf else ("NtpSt;1" if "NtpSt;1" in rf else "sntp")
        tree = rf[tree_key]

        req_branches = [
            "NtpStRecord/stp/stp.planeview",
            "NtpStRecord/stp/stp.strip",
            "NtpStRecord/stp/stp.plane",
            "NtpStRecord/stp/stp.ph0.pe",
            "NtpStRecord/stp/stp.ph1.pe",
            "NtpStRecord/mc/mc.iaction",
            "NtpStRecord/mc/mc.p4neu[4]",
        ]
        if self.feature_mode == "timing":
            req_branches.extend([
                "NtpStRecord/stp/stp.time0",
                "NtpStRecord/stp/stp.time1",
                "NtpStRecord/stp/stp.tpos",
                "NtpStRecord/stp/stp.z",
            ])

        branches = tree.arrays(req_branches, entry_stop=max_events)

        if view_ids is None:
            observed_views = []
            for chunk in branches["NtpStRecord/stp/stp.planeview"]:
                observed_views.extend(np.asarray(chunk).tolist())

            unique_views = sorted({int(v) for v in observed_views})
            if len(unique_views) < 2:
                raise ValueError(
                    f"Could not auto-detect two planeview ids from {self.root_filepath}. "
                    f"Observed values: {unique_views}"
                )

            self.view_ids = (unique_views[0], unique_views[1])
        else:
            if len(view_ids) != 2:
                raise ValueError("MINOSMultiViewGraphDataset expects exactly two view ids")
            self.view_ids = tuple(int(v) for v in view_ids)

        for i in range(len(branches)):
            iaction = branches["NtpStRecord/mc/mc.iaction"][i]
            label = 1 if iaction == 1 else 0
            true_energy = float(np.asarray(branches["NtpStRecord/mc/mc.p4neu[4]"][i]).reshape(-1)[-1])

            views = np.array(branches["NtpStRecord/stp/stp.planeview"][i])
            planes = np.array(branches["NtpStRecord/stp/stp.plane"][i])
            strips = np.array(branches["NtpStRecord/stp/stp.strip"][i])
            ph0 = np.array(branches["NtpStRecord/stp/stp.ph0.pe"][i])
            ph1 = np.array(branches["NtpStRecord/stp/stp.ph1.pe"][i])

            if self.feature_mode == "timing":
                time0 = np.array(branches["NtpStRecord/stp/stp.time0"][i])
                time1 = np.array(branches["NtpStRecord/stp/stp.time1"][i])
                tpos = np.array(branches["NtpStRecord/stp/stp.tpos"][i])
                z = np.array(branches["NtpStRecord/stp/stp.z"][i])
                event = self._build_timing_graph(
                    views, planes, strips, ph0, ph1,
                    time0, time1, tpos, z, label, true_energy,
                )
            else:
                event = self._build_event_graph(views, planes, strips, ph0, ph1, label, true_energy)
            if event is not None:
                self.events.append(event)

        if not self.events:
            raise ValueError(
                f"No multi-view graph events were constructed for view_ids={self.view_ids}. "
                "Check that the chosen planeview ids match the ROOT file."
            )

        if cache_file is not None:
            _save_cache_file(
                cache_file,
                {
                    "kind": "multi_view_graph",
                    "root_filepath": str(self.root_filepath),
                    "view_ids": self.view_ids,
                    "plane_radius": self.plane_radius,
                    "strip_radius": self.strip_radius,
                    "graph_mode": self.graph_mode,
                    "feature_mode": self.feature_mode,
                    "max_events": max_events,
                    "events": self.events,
                },
            )

    @property
    def in_channels(self) -> int:
        if self.feature_mode == "timing":
            return 9
        elif self.feature_mode == "dual_ph":
            return 2
        elif self.feature_mode == "ph_full":
            return 4
        return 1

    def _view_feature_matrix(
        self,
        planes: np.ndarray,
        strips: np.ndarray,
        ph0: np.ndarray,
        ph1: np.ndarray,
    ) -> torch.Tensor:
        phs = ph0 + ph1
        plane_feat = planes.astype(np.float32)
        strip_feat = strips.astype(np.float32)
        denom_plane = float(max(1, int(np.max(np.abs(plane_feat))) if plane_feat.size else 1))
        denom_strip = float(max(1, int(np.max(np.abs(strip_feat))) if strip_feat.size else 1))

        if self.feature_mode == "dual_ph":
            norm_ph0 = np.log1p(np.maximum(0.0, ph0)).astype(np.float32)
            norm_ph1 = np.log1p(np.maximum(0.0, ph1)).astype(np.float32)
            charge_feats = [norm_ph0, norm_ph1]
        elif self.feature_mode == "ph_full":
            norm_sum = np.log1p(np.maximum(0.0, phs)).astype(np.float32)
            norm_ph0 = np.log1p(np.maximum(0.0, ph0)).astype(np.float32)
            norm_ph1 = np.log1p(np.maximum(0.0, ph1)).astype(np.float32)
            denom = phs + 1e-5
            asym = ((ph0 - ph1) / denom).astype(np.float32)
            charge_feats = [norm_sum, norm_ph0, norm_ph1, asym]
        else:
            norm_charge = np.log1p(np.maximum(0.0, phs)).astype(np.float32)
            charge_feats = [norm_charge]

        feats = np.column_stack(
            charge_feats + [
                plane_feat / denom_plane,
                strip_feat / denom_strip,
                np.ones_like(planes, dtype=np.float32),
            ]
        ).astype(np.float32)
        return torch.tensor(feats, dtype=torch.float32)

    def _dense_edges(self, coords: np.ndarray) -> torch.Tensor:
        if coords.shape[0] < 2:
            return torch.empty((2, 0), dtype=torch.long)

        plane = torch.tensor(coords[:, 0], dtype=torch.long)
        strip = torch.tensor(coords[:, 1], dtype=torch.long)

        plane_diff = (plane[:, None] - plane[None, :]).abs()
        strip_diff = (strip[:, None] - strip[None, :]).abs()
        adjacency = (
            (plane_diff <= self.plane_radius)
            & (strip_diff <= self.strip_radius)
            & ~torch.eye(coords.shape[0], dtype=torch.bool)
        )

        edge_index = adjacency.nonzero(as_tuple=False).t().contiguous()
        return edge_index

    def _delaunay_knn_edges(
        self,
        coords: np.ndarray,
        max_plane_diff: int = 3,
        max_strip_diff: int = 8,
    ) -> torch.Tensor:
        if coords.shape[0] < 2:
            return torch.empty((2, 0), dtype=torch.long)
        if coords.shape[0] < 4:
            return self._dense_edges(coords)

        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(coords)
            edges_set = set()
            for simplex in tri.simplices:
                for u, v in [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]:
                    if u != v:
                        p_diff = abs(int(coords[u, 0]) - int(coords[v, 0]))
                        s_diff = abs(int(coords[u, 1]) - int(coords[v, 1]))
                        if p_diff <= max_plane_diff and s_diff <= max_strip_diff:
                            edges_set.add((u, v))
                            edges_set.add((v, u))
            if not edges_set:
                return self._dense_edges(coords)
            edges_arr = np.array(list(edges_set), dtype=np.int64).T
            return torch.tensor(edges_arr, dtype=torch.long)
        except Exception:
            return self._dense_edges(coords)

    def _build_event_graph(
        self,
        views: np.ndarray,
        planes: np.ndarray,
        strips: np.ndarray,
        ph0: np.ndarray,
        ph1: np.ndarray,
        label: int,
        true_energy: float,
    ):
        view_a, view_b = self.view_ids
        mask_a = views == view_a
        mask_b = views == view_b

        if not np.any(mask_a) or not np.any(mask_b):
            return None

        data = HeteroData()

        view_specs = [
            ("view_a", mask_a),
            ("view_b", mask_b),
        ]

        nexus_planes = np.unique(np.concatenate([planes[mask_a], planes[mask_b]])).astype(np.int64)
        nexus_lookup: Dict[int, int] = {int(plane): idx for idx, plane in enumerate(nexus_planes.tolist())}
        nexus_feats = np.column_stack(
            [
                nexus_planes.astype(np.float32),
                np.ones_like(nexus_planes, dtype=np.float32),
                np.zeros_like(nexus_planes, dtype=np.float32),
                np.zeros_like(nexus_planes, dtype=np.float32),
            ]
        ).astype(np.float32)

        data["nexus"].x = torch.tensor(nexus_feats, dtype=torch.float32)
        data["nexus"].plane = torch.tensor(nexus_planes, dtype=torch.long)

        for node_type, mask in view_specs:
            node_planes = planes[mask].astype(np.int64)
            node_strips = strips[mask].astype(np.int64)
            node_ph0 = ph0[mask]
            node_ph1 = ph1[mask]
            coords = np.column_stack([node_planes, node_strips]).astype(np.int64)

            data[node_type].x = self._view_feature_matrix(node_planes, node_strips, node_ph0, node_ph1)
            data[node_type].plane = torch.tensor(node_planes, dtype=torch.long)
            data[node_type].strip = torch.tensor(node_strips, dtype=torch.long)
            data[node_type].pos = torch.tensor(coords, dtype=torch.long)

            if self.graph_mode == "delaunay_knn":
                same_view_edges = self._delaunay_knn_edges(coords)
            else:
                same_view_edges = self._dense_edges(coords)

            data[(node_type, "same_view", node_type)].edge_index = same_view_edges

            if coords.shape[0] > 0:
                nexus_targets = torch.tensor(
                    [nexus_lookup[int(plane)] for plane in node_planes],
                    dtype=torch.long,
                )
                hit_to_nexus = torch.stack(
                    [torch.arange(coords.shape[0], dtype=torch.long), nexus_targets],
                    dim=0,
                )
            else:
                hit_to_nexus = torch.empty((2, 0), dtype=torch.long)

            data[(node_type, "to_nexus", "nexus")].edge_index = hit_to_nexus
            data[("nexus", f"rev_to_{node_type}", node_type)].edge_index = hit_to_nexus.flip(0)

        data.y = torch.tensor(label, dtype=torch.long)
        data.true_energy = torch.tensor(true_energy, dtype=torch.float32)
        return data

    # ── Timing-aware merged graph construction ────────────────────────

    def _build_timing_graph(
        self,
        views: np.ndarray,
        planes: np.ndarray,
        strips: np.ndarray,
        ph0: np.ndarray,
        ph1: np.ndarray,
        time0: np.ndarray,
        time1: np.ndarray,
        tpos: np.ndarray,
        z: np.ndarray,
        label: int,
        true_energy: float,
    ):
        """Build a merged homogeneous graph with timing features for both views."""
        if Data is None:
            raise ImportError(
                "torch_geometric is required for timing graph construction."
            )

        view_a_id, view_b_id = self.view_ids
        mask_a = views == view_a_id
        mask_b = views == view_b_id
        combined_mask = mask_a | mask_b

        if np.sum(mask_a) == 0 or np.sum(mask_b) == 0:
            return None

        # Select hits from both views
        sel_ph0 = ph0[combined_mask]
        sel_ph1 = ph1[combined_mask]
        sel_time0 = time0[combined_mask]
        sel_time1 = time1[combined_mask]
        sel_tpos = tpos[combined_mask]
        sel_z = z[combined_mask]
        sel_planes = planes[combined_mask]
        sel_views = views[combined_mask]
        n_hits = int(np.sum(combined_mask))

        if n_hits < 2:
            return None

        # --- Readout validity masks ---
        # Sentinel value for missing readout is -999999 in time, and 0 in ph
        valid_east = (sel_ph0 > 0).astype(np.float32)
        valid_west = (sel_ph1 > 0).astype(np.float32)

        # --- Charge features (log-scaled) ---
        pe_east_log = np.log1p(np.maximum(0.0, sel_ph0)).astype(np.float32)
        pe_west_log = np.log1p(np.maximum(0.0, sel_ph1)).astype(np.float32)

        # --- Time features ---
        # Convert times from seconds to nanoseconds
        t0_ns = sel_time0 * 1e9
        t1_ns = sel_time1 * 1e9

        # Compute per-hit mean time and dt where both sides are valid
        both_valid = (valid_east > 0) & (valid_west > 0)
        t_mean_raw = np.zeros(n_hits, dtype=np.float64)
        dt_raw = np.zeros(n_hits, dtype=np.float64)

        # For both-valid hits: use mean and difference
        if np.any(both_valid):
            t_mean_raw[both_valid] = (t0_ns[both_valid] + t1_ns[both_valid]) / 2.0
            dt_raw[both_valid] = t0_ns[both_valid] - t1_ns[both_valid]

        # For single-ended hits: use the valid side's time as the mean
        east_only = (valid_east > 0) & (valid_west == 0)
        west_only = (valid_west > 0) & (valid_east == 0)
        if np.any(east_only):
            t_mean_raw[east_only] = t0_ns[east_only]
        if np.any(west_only):
            t_mean_raw[west_only] = t1_ns[west_only]

        # Event-relative zeroing using median of valid hit times
        any_valid = (valid_east > 0) | (valid_west > 0)
        if np.any(any_valid):
            t0_event = float(np.median(t_mean_raw[any_valid]))
        else:
            t0_event = 0.0
        t_rel = t_mean_raw - t0_event

        # Clip and scale
        T_MAX_NS = 150.0
        DT_MAX_NS = 300.0
        t_scaled = np.clip(t_rel, -T_MAX_NS, T_MAX_NS) / T_MAX_NS
        dt_scaled = np.clip(dt_raw, -DT_MAX_NS, DT_MAX_NS) / DT_MAX_NS

        # Zero out timing features for hits with no valid readout
        no_valid = ~any_valid
        t_scaled[no_valid] = 0.0
        dt_scaled[no_valid] = 0.0
        # For single-ended hits, dt is not meaningful — zero it out
        dt_scaled[~both_valid] = 0.0

        # --- Spatial features ---
        tpos_norm = (sel_tpos / 4.0).astype(np.float32)
        z_norm = ((sel_z - 15.0) / 15.0).astype(np.float32)

        # --- View flag ---
        view_flag = np.where(sel_views == view_a_id, 0.0, 1.0).astype(np.float32)

        # --- Assemble 9-feature node vector ---
        node_feats = np.column_stack([
            pe_east_log,
            pe_west_log,
            t_scaled.astype(np.float32),
            dt_scaled.astype(np.float32),
            tpos_norm,
            z_norm,
            view_flag,
            valid_east,
            valid_west,
        ]).astype(np.float32)

        x = torch.tensor(node_feats, dtype=torch.float32)

        # --- Edge construction: spacetime kNN ---
        edge_index, edge_attr = self._build_timing_edges(
            sel_z, sel_tpos, t_rel.astype(np.float32),
            sel_views, sel_planes, view_a_id, view_b_id, k=8,
        )

        # --- Build homogeneous Data object ---
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(label, dtype=torch.long),
        )
        data.true_energy = torch.tensor(true_energy, dtype=torch.float32)
        return data

    def _build_timing_edges(
        self,
        z: np.ndarray,
        tpos: np.ndarray,
        t_rel: np.ndarray,
        views: np.ndarray,
        planes: np.ndarray,
        view_a_id: int,
        view_b_id: int,
        k: int = 8,
    ) -> tuple:
        """Build spacetime kNN edges with 6-dimensional edge features."""
        n = len(z)
        if n < 2:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 6), dtype=torch.float32)
            return edge_index, edge_attr

        # Speed of light in fiber ~ 0.2 m/ns (c_fiber for MINOS WLS fiber)
        C_SCALE = 0.2  # meters per nanosecond

        # Compute pairwise spacetime distance
        dz = z[:, None] - z[None, :]       # [N, N]
        dtpos = tpos[:, None] - tpos[None, :]  # [N, N]
        dt = t_rel[:, None] - t_rel[None, :]   # [N, N] in nanoseconds

        spatial_dist_sq = dz ** 2 + dtpos ** 2
        spacetime_dist = np.sqrt(spatial_dist_sq + (C_SCALE * dt) ** 2)

        # Same-view flag
        same_view = (views[:, None] == views[None, :]).astype(np.float32)

        # kNN: for each node, select k nearest neighbors (excluding self)
        np.fill_diagonal(spacetime_dist, np.inf)
        effective_k = min(k, n - 1)
        knn_indices = np.argpartition(spacetime_dist, effective_k, axis=1)[:, :effective_k]

        # Also add cross-view edges at same/adjacent planes
        plane_diff = np.abs(planes[:, None].astype(int) - planes[None, :].astype(int))
        cross_view = (same_view == 0) & (plane_diff <= self.plane_radius)

        # Build edge lists
        src_list = []
        dst_list = []
        edge_set = set()

        # Add kNN edges
        for i in range(n):
            for j in knn_indices[i]:
                j = int(j)
                if (i, j) not in edge_set:
                    edge_set.add((i, j))
                    edge_set.add((j, i))
                    src_list.extend([i, j])
                    dst_list.extend([j, i])

        # Add cross-view edges (may already be in kNN set)
        cross_src, cross_dst = np.where(cross_view)
        for i, j in zip(cross_src, cross_dst):
            i, j = int(i), int(j)
            if (i, j) not in edge_set:
                edge_set.add((i, j))
                src_list.append(i)
                dst_list.append(j)

        if not src_list:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 6), dtype=torch.float32)
            return edge_index, edge_attr

        src = np.array(src_list, dtype=np.int64)
        dst = np.array(dst_list, dtype=np.int64)

        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

        # --- Edge features [Δz, Δtpos, Δt_scaled, ||r||, causal_flag, same_view_flag] ---
        e_dz = (z[dst] - z[src]).astype(np.float32)
        e_dtpos = (tpos[dst] - tpos[src]).astype(np.float32)
        e_dt = (t_rel[dst] - t_rel[src]).astype(np.float32)
        e_dt_scaled = np.clip(e_dt, -300.0, 300.0) / 300.0
        e_dist = np.sqrt(e_dz ** 2 + e_dtpos ** 2).astype(np.float32)

        # Causality flag: |Δt| >= ||Δr|| / c  (is it physically reachable at <= c?)
        # c in vacuum ~ 0.3 m/ns
        C_VACUUM = 0.3  # meters per nanosecond
        abs_dt_ns = np.abs(e_dt)
        causal_flag = (abs_dt_ns >= e_dist / C_VACUUM).astype(np.float32)

        same_view_flag = (views[src] == views[dst]).astype(np.float32)

        edge_features = np.column_stack([
            e_dz / 15.0,        # Normalize Δz ~ same scale as z_norm
            e_dtpos / 4.0,      # Normalize Δtpos ~ same scale as tpos_norm
            e_dt_scaled,
            e_dist / 15.0,      # Normalize distance
            causal_flag,
            same_view_flag,
        ]).astype(np.float32)

        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        return edge_index, edge_attr


    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int):
        event = self.events[idx]
        if hasattr(event, "to"):
            event = event.to("cpu")
        if hasattr(event, "clone"):
            return event.clone()
        return event

    def get_class_weights(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        labels = [e.y.item() for e in self.events]
        total = len(labels)
        cc_count = sum(labels)
        nc_count = total - cc_count
        w_nc = total / (2.0 * max(1, nc_count))
        w_cc = total / (2.0 * max(1, cc_count))
        return torch.tensor([w_nc, w_cc], dtype=torch.float32, device=device)


def create_multiview_gnn_dataloaders(
    dataset: MINOSMultiViewGraphDataset,
    batch_size: int = 32,
    val_split: float = 0.20,
    random_seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    if PyGDataLoader is None:
        raise ImportError(
            "torch_geometric is required for create_multiview_gnn_dataloaders()."
        )

    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(random_seed)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = PyGDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = PyGDataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_ds.indices, val_ds.indices
