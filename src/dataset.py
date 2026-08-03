from typing import Tuple, List, Optional, Sequence, Dict
from pathlib import Path
import numpy as np
import uproot
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:
    HeteroData = None
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
            if cached_feature_mode == self.feature_mode:
                self.root_filepath = Path(cache.get("root_filepath", self.root_filepath))
                self.target_view = int(cache.get("target_view", self.target_view))
                self.events = cache["events"]
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
            valid_view_ids = (view_ids is None or tuple(int(v) for v in view_ids) == cached_view_ids)
            valid_mode = (cached_feature_mode == self.feature_mode)

            if valid_view_ids and valid_mode:
                self.root_filepath = Path(cache.get("root_filepath", self.root_filepath))
                self.view_ids = cached_view_ids
                self.plane_radius = int(cache.get("plane_radius", self.plane_radius))
                self.strip_radius = int(cache.get("strip_radius", self.strip_radius))
                self.graph_mode = str(cache.get("graph_mode", self.graph_mode))
                self.events = cache["events"]
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
        if self.feature_mode == "dual_ph":
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
