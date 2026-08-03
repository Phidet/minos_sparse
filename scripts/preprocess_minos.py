#!/usr/bin/env python3
"""Preprocess MINOS ROOT files into cached PyTorch datasets.

This script extracts the sparse single-view U-view events and the two-view
hetero-graph events into .pt cache files that can be loaded quickly from the
main notebook without reopening the ROOT file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import MINOSSingleViewDataset, MINOSMultiViewGraphDataset  # noqa: E402


def build_single_view(root_filepath: str, output_path: Path, target_view: int, max_events: int | None) -> None:
    if output_path.exists():
        output_path.unlink()

    dataset = MINOSSingleViewDataset(
        root_filepath=root_filepath,
        max_events=max_events,
        target_view=target_view,
        cache_path=str(output_path),
    )
    print(f"Saved single-view cache to {output_path} with {len(dataset)} events")


def build_multi_view(
    root_filepath: str,
    output_path: Path,
    view_ids: tuple[int, int],
    plane_radius: int,
    strip_radius: int,
    max_events: int | None,
) -> None:
    if output_path.exists():
        output_path.unlink()

    dataset = MINOSMultiViewGraphDataset(
        root_filepath=root_filepath,
        max_events=max_events,
        view_ids=view_ids,
        plane_radius=plane_radius,
        strip_radius=strip_radius,
        cache_path=str(output_path),
    )
    print(f"Saved multi-view cache to {output_path} with {len(dataset)} events and view_ids={dataset.view_ids}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess MINOS ROOT data into cached tensors")
    parser.add_argument(
        "--root-filepath",
        type=str,
        default="../f21048000_0000_L010185N_D07_r3.sntp.dogwood5.0.root",
        help="Path to the MINOS ROOT file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/cache",
        help="Directory where cache files will be written",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=12000,
        help="Maximum number of events to preprocess",
    )
    parser.add_argument(
        "--target-view",
        type=int,
        default=2,
        help="Single-view planeview id to extract for the CNN baseline",
    )
    parser.add_argument(
        "--view-ids",
        type=int,
        nargs=2,
        default=(2, 3),
        metavar=("VIEW_A", "VIEW_B"),
        help="Two planeview ids to extract for the graph models",
    )
    parser.add_argument(
        "--plane-radius",
        type=int,
        default=1,
        help="Neighborhood radius in plane index for graph edges",
    )
    parser.add_argument(
        "--strip-radius",
        type=int,
        default=2,
        help="Neighborhood radius in strip index for graph edges",
    )
    parser.add_argument(
        "--single-name",
        type=str,
        default="minos_uview_single_view.pt",
        help="Filename for the cached single-view dataset",
    )
    parser.add_argument(
        "--multi-name",
        type=str,
        default="minos_uview_multi_view_graph.pt",
        help="Filename for the cached multi-view graph dataset",
    )
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Only generate the single-view cache",
    )
    parser.add_argument(
        "--multi-only",
        action="store_true",
        help="Only generate the multi-view cache",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_filepath = Path(args.root_filepath)
    if not root_filepath.exists():
        raise FileNotFoundError(f"ROOT file not found: {root_filepath}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_single = not args.multi_only
    run_multi = not args.single_only

    if run_single:
        build_single_view(
            root_filepath=str(root_filepath),
            output_path=output_dir / args.single_name,
            target_view=args.target_view,
            max_events=args.max_events,
        )

    if run_multi:
        build_multi_view(
            root_filepath=str(root_filepath),
            output_path=output_dir / args.multi_name,
            view_ids=tuple(args.view_ids),
            plane_radius=args.plane_radius,
            strip_radius=args.strip_radius,
            max_events=args.max_events,
        )


if __name__ == "__main__":
    main()
