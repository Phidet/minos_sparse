#!/usr/bin/env python3
"""
MINOS Multi-Model Training Script (train.py)
--------------------------------------------
Trains single or multiple MINOS model configurations sequentially,
auto-commits repository state before starting, saves checkpoints with training history,
and logs metrics to model_leaderboard.csv.

Usage examples:
    python train.py --sntp path/to/file.sntp.root --models cnn_resnet_cross_attention cnn_cross_attention
    python train.py --sntp path/to/file.sntp.root --all
    python train.py --sntp path/to/file.sntp.root --models gnn_nugraph --num_epochs 15 --max_events 5000
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src
from src import (
    MODEL_CONFIGS,
    DATASET_CONFIG,
    get_config,
    MINOSMultiViewGraphDataset,
    create_multiview_gnn_dataloaders,
    train_model,
    auto_commit_and_get_hash,
    save_model_checkpoint,
    log_experiment,
    display_leaderboard,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MINOS event classification models")
    parser.add_argument(
        "--sntp", type=str, required=True,
        help="Path to the MINOS sntp ROOT file, e.g. f21048000_0000_L010185N_D07_r3.sntp.dogwood5.0.root "
             "(only actually read if no cache exists yet for the requested feature_mode)"
    )
    parser.add_argument(
        "--models", "-m", nargs="+", type=str, default=None,
        help="List of model configuration keys to train (e.g. cnn_resnet_cross_attention gnn_nugraph)"
    )
    parser.add_argument(
        "--all", action="store_true", default=False,
        help="Train all registered model configurations sequentially"
    )
    parser.add_argument(
        "--max_events", type=int, default=None,
        help="Override max_events for dataset loading"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=None,
        help="Override num_epochs for training"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate for training"
    )
    parser.add_argument(
        "--no_auto_commit", action="store_true", default=False,
        help="Disable automatic git commit before training"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for model init and batch shuffling (default: DATASET_CONFIG['random_seed']). "
             "The train/val split is NOT affected -- it stays pinned to DATASET_CONFIG['random_seed'] "
             "so results remain comparable across seeds."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine model configuration keys to run
    if args.all:
        target_keys = list(MODEL_CONFIGS.keys())
    elif args.models:
        target_keys = args.models
    else:
        # Default run if no args provided
        target_keys = ["cnn_resnet_cross_attention"]
        print("No --models or --all specified. Defaulting to 'cnn_resnet_cross_attention'.")
        print(f"Available model configurations: {list(MODEL_CONFIGS.keys())}\n")

    # Validate model keys
    for k in target_keys:
        if k not in MODEL_CONFIGS:
            print(f"Error: Unknown model config key '{k}'.")
            print(f"Available configs: {list(MODEL_CONFIGS.keys())}")
            sys.exit(1)

    print(f"=========================================================================")
    print(f"MINOS Training Campaign: {len(target_keys)} model configuration(s) queued.")
    print(f"Models: {', '.join(target_keys)}")
    print(f"=========================================================================\n")

    # Perform Git auto-commit once before starting the multi-model training run
    if not args.no_auto_commit and DATASET_CONFIG.get("auto_commit", True):
        git_hash = auto_commit_and_get_hash(
            f"Auto-commit before multi-model training campaign ({len(target_keys)} models)"
        )
    else:
        git_hash = "manual-run"

    # Two distinct seeds, deliberately separated:
    #
    #   split_seed -- ALWAYS DATASET_CONFIG['random_seed']. Pins the train/val
    #     split so every run scores on the same validation events and stays
    #     comparable. Never overridden by --seed.
    #   run_seed   -- model init and batch shuffle order. Re-applied at the top
    #     of each model's iteration below (NOT once here), so a model's result
    #     depends only on (config, seed) and not on its position in --models.
    #     Previously this was seeded once before the loop, which meant the same
    #     config gave different results depending on what ran before it.
    split_seed = DATASET_CONFIG.get("random_seed", 42)
    run_seed = args.seed if args.seed is not None else split_seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Split seed (fixed): {split_seed} | Run seed (init/shuffle): {run_seed}")
    print(f"PyTorch Device: {device}\n")

    # Determine which distinct datasets are needed across all target models.
    # Keyed by cache_path (not just feature_mode): two configs can share a
    # feature_mode but need different data (e.g. 'timing' vs. an
    # ablate_timing=True variant pointing at a different cache_path).
    max_events = args.max_events if args.max_events is not None else DATASET_CONFIG["max_events"]
    needed_cache_paths = {}
    for key in target_keys:
        config = get_config(key)
        cache_path = config.get("cache_path", DATASET_CONFIG["cache_path"])
        needed_cache_paths.setdefault(cache_path, config)

    # Lazily load datasets keyed by cache_path
    datasets = {}
    for cache_path, config in needed_cache_paths.items():
        mode = config.get("feature_mode", DATASET_CONFIG["feature_mode"])
        ablate_timing = config.get("ablate_timing", False)

        print(f"Loading MINOS dataset (max_events={max_events}, feature_mode='{mode}', ablate_timing={ablate_timing}, cache_path='{cache_path}')...")
        datasets[cache_path] = MINOSMultiViewGraphDataset(
            root_filepath=args.sntp,
            max_events=max_events,
            view_ids=DATASET_CONFIG["view_ids"],
            plane_radius=DATASET_CONFIG["plane_radius"],
            strip_radius=DATASET_CONFIG["strip_radius"],
            feature_mode=mode,
            cache_path=cache_path,
            allow_root_fallback=DATASET_CONFIG.get("allow_root_fallback", True),
            ablate_timing=ablate_timing,
        )
        print(f"  Loaded {len(datasets[cache_path])} events for feature_mode='{mode}'")

    # Train each model configuration
    completed_runs = []
    for idx, key in enumerate(target_keys, 1):
        # Re-seed per model (not once before the loop) so that a model's weight
        # init and shuffle order depend only on (config, run_seed) -- identical
        # whether it runs first or fifth in the campaign.
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)

        config = get_config(key)
        model_name = config["model_name"]
        num_epochs = args.num_epochs if args.num_epochs is not None else config.get("num_epochs", 12)
        lr = args.lr if args.lr is not None else config.get("lr", 1e-3)
        batch_size = config.get("batch_size", 32)
        model_feature_mode = config.get("feature_mode", DATASET_CONFIG["feature_mode"])
        model_cache_path = config.get("cache_path", DATASET_CONFIG["cache_path"])
        dataset = datasets[model_cache_path]

        criterion = config.get("loss", config.get("loss_fn", config.get("criterion", None)))
        loss_name = criterion.__class__.__name__ if hasattr(criterion, "__class__") else str(criterion)

        print(f"\n[{idx}/{len(target_keys)}] Starting Training: {key} ({model_name})")
        print(f"    Epochs: {num_epochs} | LR: {lr:.1e} | Batch Size: {batch_size} | Weight Decay: {config.get('weight_decay', 1e-4)}")
        print(f"    LR Scheduler: StepLR(step_size={config.get('step_size', 3)}, gamma={config.get('gamma', 0.3)})")
        print(f"    Loss Function: {loss_name}")
        print(f"    Feature Mode: {model_feature_mode}")

        train_loader, val_loader, _, _ = create_multiview_gnn_dataloaders(
            dataset,
            batch_size=batch_size,
            val_split=DATASET_CONFIG.get("val_split", 0.20),
            random_seed=split_seed,
        )

        if config.get("use_class_weights", False):
            class_weights = dataset.get_class_weights(device=device)
        else:
            class_weights = None

        # Build PyTorch model instance
        sample_graph = dataset[0]
        if config.get("is_gnn", False):
            # Timing GNN uses homogeneous Data (no metadata/nexus);
            # other GNNs use HeteroData with metadata()
            if hasattr(sample_graph, "metadata"):
                metadata = sample_graph.metadata()
                graph_in_channels = sample_graph["view_a"].x.size(-1)
            else:
                metadata = None
                graph_in_channels = sample_graph.x.size(-1)
            model = config["model_factory"](metadata, graph_in_channels)
        else:
            model = config["model"]

        num_params = model.get_num_params()
        print(f"    Total Trainable Parameters: {num_params:,}")

        # Train model
        history, best_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=config.get("weight_decay", 1e-4),
            step_size=config.get("step_size", 3),
            gamma=config.get("gamma", 0.3),
            criterion=criterion,
            class_weights=class_weights,
            use_energy_weighting=config.get("use_energy_weights", False),
            selection_metric="roc_auc",
            device=device,
            verbose=True,
        )

        # Save checkpoint locally with training history
        model_path = save_model_checkpoint(
            model=model,
            config=config,
            best_metrics=best_metrics,
            git_hash=git_hash,
            save_dir=DATASET_CONFIG.get("save_dir", "saved_models"),
            history=history,
        )

        # Log metrics & hyperparameters to CSV leaderboard.
        # Record the *resolved* values rather than the config defaults, so that
        # --num_epochs/--lr overrides are reflected in the leaderboard instead
        # of it silently claiming the config's default.
        run_config = dict(config)
        run_config["max_events"] = len(dataset)
        run_config["num_epochs"] = num_epochs
        run_config["lr"] = lr
        # n_events is the realised dataset length, which is what determines the
        # validation split and therefore which runs are comparable at all.
        # (max_events historically recorded the requested cap for pre-Aug-6 runs.)
        run_config["n_events"] = len(dataset)
        run_config["seed"] = run_seed
        leaderboard_df = log_experiment(
            config=run_config,
            model_name=model_name,
            best_metrics=best_metrics,
            git_hash=git_hash,
            model_path=model_path,
            num_params=num_params,
            csv_path=DATASET_CONFIG.get("leaderboard_csv", "model_leaderboard.csv"),
        )

        completed_runs.append((key, model_name, best_metrics.get("roc_auc", 0.0), best_metrics.get("f1", 0.0)))
        print(f"Finished {key} -> ROC-AUC: {best_metrics.get('roc_auc', 0.0):.4f} | F1: {best_metrics.get('f1', 0.0):.4f}")

    # Display final leaderboard
    print(f"\n=========================================================================")
    print(f"CAMPAIGN COMPLETED: {len(completed_runs)} model(s) trained.")
    print(f"=========================================================================")
    display_leaderboard(csv_path=DATASET_CONFIG.get("leaderboard_csv", "model_leaderboard.csv"))


if __name__ == "__main__":
    main()
