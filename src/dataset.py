from typing import Tuple, List, Optional
from pathlib import Path
import numpy as np
import uproot
import torch
from torch.utils.data import Dataset, DataLoader


class MINOSSingleViewDataset(Dataset):
    """
    Lightweight Dataset parser for single-view (U-view) MINOS event displays.
    Target view = 2 (U-view).
    Target label: CC (1) vs NC (0) based on iaction branch.
    """

    def __init__(self, root_filepath: str, max_events: Optional[int] = None, target_view: int = 2):
        super().__init__()
        self.root_filepath = Path(root_filepath)
        self.target_view = target_view
        self.events = []

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
            "NtpStRecord/mc/mc.iaction"
        ]

        branches = tree.arrays(
            req_branches,
            entry_stop=max_events
        )

        for i in range(len(branches)):
            iaction = branches["NtpStRecord/mc/mc.iaction"][i]
            label = 1 if iaction == 1 else 0

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
            sel_phs = phs[mask]

            norm_phs = np.log1p(np.maximum(0.0, sel_phs)).astype(np.float32)
            coords = np.column_stack((sel_planes, sel_strips)).astype(np.int64)
            feats = norm_phs[:, np.newaxis]

            self.events.append({
                "coords": torch.tensor(coords, dtype=torch.long),
                "feats": torch.tensor(feats, dtype=torch.float32),
                "label": torch.tensor(label, dtype=torch.long)
            })

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> dict:
        return self.events[idx]

    def get_class_weights(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        labels = [e["label"].item() for e in self.events]
        total = len(labels)
        cc_count = sum(labels)
        nc_count = total - cc_count
        w_nc = total / (2.0 * max(1, nc_count))
        w_cc = total / (2.0 * max(1, cc_count))
        return torch.tensor([w_nc, w_cc], dtype=torch.float32, device=device)


def sparse_uview_collate_fn(batch: List[dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collates sparse event display hits into batch tensor [batch_idx, plane, strip]."""
    coords_list = []
    feats_list = []
    labels_list = []

    for batch_idx, item in enumerate(batch):
        c = item["coords"]
        f = item["feats"]
        lbl = item["label"]

        batch_idx_col = torch.full((c.shape[0], 1), batch_idx, dtype=torch.long)
        b_coords = torch.cat([batch_idx_col, c], dim=1)

        coords_list.append(b_coords)
        feats_list.append(f)
        labels_list.append(lbl)

    batch_coords = torch.cat(coords_list, dim=0)
    batch_feats = torch.cat(feats_list, dim=0)
    batch_labels = torch.stack(labels_list, dim=0)

    return batch_coords, batch_feats, batch_labels


def create_uview_dataloaders(
    dataset: MINOSSingleViewDataset,
    batch_size: int = 32,
    val_split: float = 0.20,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(random_seed)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=sparse_uview_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=sparse_uview_collate_fn
    )

    return train_loader, val_loader, train_ds.indices, val_ds.indices
