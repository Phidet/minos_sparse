import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src
from scripts.run_single_experiment import run_experiment

EXPERIMENTS = [
    {
        "model_type": "cnn_deep_resnet_cross_attention",
        "model_class_name": "DualViewDeepResNetCrossAttentionSparseCNN",
        "root_filepath": "f21048000_0000_L010185N_D07_r3.sntp.dogwood5.0.root",
        "view_ids": (2, 3),
        "max_events": 12000,
        "batch_size": 32,
        "val_split": 0.20,
        "random_seed": 42,
        "feature_mode": "sum",
        "in_channels": 1,
        "conv_channels": [32, 64, 128],
        "fc_dims": [32, 16],
        "dropout": 0.1,
        "num_epochs": 15,
        "lr": 0.001,
        "weight_decay": 1e-4,
        "step_size": 4,
        "gamma": 0.5,
        "selection_metric": "roc_auc",
        "leaderboard_csv": "model_leaderboard.csv",
        "save_dir": "saved_models",
    },
    {
        "model_type": "cnn_resnet_dualpool_cross_attention",
        "model_class_name": "DualViewResNetDualPoolCrossAttentionSparseCNN",
        "root_filepath": "f21048000_0000_L010185N_D07_r3.sntp.dogwood5.0.root",
        "view_ids": (2, 3),
        "max_events": 12000,
        "batch_size": 32,
        "val_split": 0.20,
        "random_seed": 42,
        "feature_mode": "sum",
        "in_channels": 1,
        "conv_channels": [32, 64],
        "fc_dims": [32, 16],
        "dropout": 0.1,
        "num_epochs": 15,
        "lr": 0.001,
        "weight_decay": 1e-4,
        "step_size": 4,
        "gamma": 0.5,
        "selection_metric": "roc_auc",
        "leaderboard_csv": "model_leaderboard.csv",
        "save_dir": "saved_models",
    },
    {
        "model_type": "cnn_resnet_multistage_cross_attention",
        "model_class_name": "DualViewResNetMultiStageCrossAttentionSparseCNN",
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
    },
    {
        "model_type": "cnn_resnet_cross_attention_dual_ph",
        "model_class_name": "DualViewResNetCrossAttentionSparseCNN",
        "root_filepath": "f21048000_0000_L010185N_D07_r3.sntp.dogwood5.0.root",
        "view_ids": (2, 3),
        "max_events": 12000,
        "batch_size": 32,
        "val_split": 0.20,
        "random_seed": 42,
        "feature_mode": "dual_ph",
        "in_channels": 2,
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
    },
    {
        "model_type": "cnn_deep_resnet_cross_attention_dual_ph",
        "model_class_name": "DualViewDeepResNetCrossAttentionSparseCNN",
        "root_filepath": "f21048000_0000_L010185N_D07_r3.sntp.dogwood5.0.root",
        "view_ids": (2, 3),
        "max_events": 12000,
        "batch_size": 32,
        "val_split": 0.20,
        "random_seed": 42,
        "feature_mode": "dual_ph",
        "in_channels": 2,
        "conv_channels": [32, 64, 128],
        "fc_dims": [32, 16],
        "dropout": 0.1,
        "num_epochs": 15,
        "lr": 0.001,
        "weight_decay": 1e-4,
        "step_size": 4,
        "gamma": 0.5,
        "selection_metric": "roc_auc",
        "leaderboard_csv": "model_leaderboard.csv",
        "save_dir": "saved_models",
    },
]

def main():
    best_overall_auc = 0.0
    winning_config = None

    for exp in EXPERIMENTS:
        print(f"\n=========================================================================")
        print(f"Starting Campaign Experiment: {exp['model_type']}")
        print(f"=========================================================================")
        try:
            metrics = run_experiment(exp)
            auc = metrics.get("roc_auc", 0.0)
            if auc > best_overall_auc:
                best_overall_auc = auc
                winning_config = exp
            print(f"Result for {exp['model_type']}: ROC-AUC = {auc:.4f}")
            if best_overall_auc >= 0.9400:
                print(f"\n🎉 GOAL ACHIEVED! ROC-AUC = {best_overall_auc:.4f} >= 0.9400!")
                break
        except Exception as e:
            print(f"Error during experiment {exp['model_type']}: {e}")

    print(f"\n=========================================================================")
    print(f"Campaign Completed. Best overall ROC-AUC: {best_overall_auc:.4f}")
    if winning_config:
        print(f"Winning Model: {winning_config['model_type']} ({winning_config['model_class_name']})")
    print(f"=========================================================================")

if __name__ == "__main__":
    main()
