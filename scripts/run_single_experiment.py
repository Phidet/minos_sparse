import sys
import argparse
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src
from src import (
    MINOSMultiViewGraphDataset,
    create_multiview_gnn_dataloaders,
    train_model,
    auto_commit_and_get_hash,
    save_model_checkpoint,
    log_experiment,
    display_leaderboard,
)


def run_experiment(config: dict):
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using PyTorch device: {device}")

    feature_mode = config.get("feature_mode", "sum")
    if feature_mode == "dual_ph":
        cache_path = "data/cache/minos_uview_multi_view_graph_dual_ph.pt"
    else:
        cache_path = "data/cache/minos_uview_multi_view_graph.pt"

    dataset = MINOSMultiViewGraphDataset(
        root_filepath=config["root_filepath"],
        max_events=config["max_events"],
        view_ids=config["view_ids"],
        plane_radius=config.get("plane_radius", 1),
        strip_radius=config.get("strip_radius", 2),
        feature_mode=feature_mode,
        cache_path=cache_path,
        allow_root_fallback=True,
    )

    train_loader, val_loader, _, _ = create_multiview_gnn_dataloaders(
        dataset,
        batch_size=config["batch_size"],
        val_split=config["val_split"],
        random_seed=config["random_seed"],
    )

    model_cls = getattr(src, config["model_class_name"])
    model_kwargs = {
        "in_channels": config.get("in_channels", 1),
        "conv_channels": config.get("conv_channels", [32, 64]),
        "fc_dims": config.get("fc_dims", [16]),
        "dropout": config.get("dropout", 0.1),
    }
    if "num_heads" in config:
        model_kwargs["num_heads"] = config["num_heads"]

    model = model_cls(**model_kwargs)
    print(f"\n========================================================")
    print(f"Running Experiment: {config['model_type']} ({config['model_class_name']})")
    print(f"Params: {model.get_num_params():,} | Epochs: {config['num_epochs']} | LR: {config['lr']} | Gamma: {config.get('gamma', 0.3)}")
    print(f"========================================================\n")

    history, best_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config["num_epochs"],
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4),
        step_size=config.get("step_size", 3),
        gamma=config.get("gamma", 0.3),
        selection_metric=config.get("selection_metric", "roc_auc"),
        device=device,
        verbose=True,
    )

    git_hash = auto_commit_and_get_hash(f"Auto-commit run {config['model_type']}")
    model_path = save_model_checkpoint(
        model, config, best_metrics, git_hash, save_dir=config.get("save_dir", "saved_models")
    )
    log_experiment(
        config,
        config["model_class_name"],
        best_metrics,
        git_hash,
        model_path,
        csv_path=config.get("leaderboard_csv", "model_leaderboard.csv"),
    )
    display_leaderboard(csv_path=config.get("leaderboard_csv", "model_leaderboard.csv"))
    return best_metrics


if __name__ == "__main__":
    BASE_CONFIG = {
        "model_type": "cnn_resnet_cross_attention",
        "model_class_name": "DualViewResNetCrossAttentionSparseCNN",
        "root_filepath": "f21048000_0000_L010185N_D07_r3.sntp.dogwood5.0.root",
        "view_ids": (2, 3),
        "max_events": 12000,
        "batch_size": 32,
        "val_split": 0.20,
        "random_seed": 42,
        "feature_mode": "sum",
        "in_channels": 1,
        "conv_channels": [32, 64],
        "fc_dims": [16],
        "dropout": 0.1,
        "num_epochs": 15,
        "lr": 0.001,
        "weight_decay": 1e-4,
        "step_size": 4,
        "gamma": 0.5,
        "selection_metric": "roc_auc",
        "leaderboard_csv": "model_leaderboard.csv",
        "save_dir": "saved_models",
    }
    run_experiment(BASE_CONFIG)
