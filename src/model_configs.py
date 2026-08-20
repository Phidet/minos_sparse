"""
MINOS Model Configurations (src/model_configs.py)
-------------------------------------------------
Plain-dictionary model configs. Each entry explicitly constructs its model
and declares its own training hyperparameters.

GNN models use a ``model_factory`` callable because they need runtime
``metadata`` and ``in_channels`` from the loaded dataset.

``DATASET_CONFIG`` holds settings that are the same across all experiments
(file paths, cache paths, split ratios, random seed, view geometry).
"""

import torch
import torch.nn as nn
import torchvision.ops as ops


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss implementation utilizing torchvision.ops.sigmoid_focal_loss.

    Expects raw logits (without prior sigmoid) and binary target labels (0 or 1).
    Supports 2-class logits shape (N, 2) by converting to binary logits (z1 - z0),
    1-class logits shape (N, 1) or (N,), and target shape matching.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 2 and logits.size(1) == 2:
            binary_logits = logits[:, 1:2] - logits[:, 0:1]
        else:
            binary_logits = logits.view(-1, 1)

        targets_float = targets.float().view_as(binary_logits)

        loss = ops.sigmoid_focal_loss(
            binary_logits,
            targets_float,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        if self.reduction == "none" and loss.ndim > 1:
            loss = loss.squeeze(-1)
        return loss


FocalLoss = BinaryFocalLoss

from .models import (
    DualViewSparseCNN,
    SimplifiedDualViewSparseCNN,
    DualViewDenseCNN,
    DualViewPlaneSummarySparseCNN,
    DualViewCrossAttentionSparseCNN,
    DualViewDeepCrossAttentionSparseCNN,
    DualViewResNetCrossAttentionSparseCNN,
    DualViewTransformerCrossAttnSparseCNN,
    SimplifiedDualViewCrossAttentionSparseCNN,
    DualViewPositionalCrossAttentionSparseCNN,
    DualViewMultiStageCrossAttentionSparseCNN,
    DualView3DIntersectionSparseCNN,
    DualViewCrossGateSparseCNN,
    DualViewHybridTransformerSparseCNN,
    DualViewMultiLayerTransformerSparseCNN,
    DualViewZipperSparseCNN,
    MinimumViableMINOSGNN,
    NuGraphInspiredBinaryGNN,
    DualViewDeepResNetCrossAttentionSparseCNN,
    DualViewResNetDualPoolCrossAttentionSparseCNN,
    DualViewResNetMultiStageCrossAttentionSparseCNN,
    TimingAwareGNN,
    TimingAwareGNNV2,
    HitSetTransformer,
)


# ── Shared dataset / infrastructure settings ─────────────────────────
# root_filepath is deliberately absent: it's machine-specific, so it's
# supplied at the call site instead (train.py's --sntp flag; set a
# ROOT_FILEPATH variable when working in a notebook).
DATASET_CONFIG = {
    "cache_path": "data/cache/minos_uview_multi_view_graph_dual_ph.pt",
    "view_ids": (2, 3),
    "plane_radius": 1,
    "strip_radius": 2,
    "max_events": 12000,
    "val_split": 0.20,
    "random_seed": 42,
    "feature_mode": "sum",
    "allow_root_fallback": True,
    "save_dir": "saved_models",
    "leaderboard_csv": "model_leaderboard.csv",
    "auto_commit": True,
}


# ── Per-model configurations ─────────────────────────────────────────
MODEL_CONFIGS = {

    # ── Dual-View Sparse CNNs ────────────────────────────────────────

    "cnn_dualview": {
        "model": DualViewSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_dualview",
        "model_name": "DualViewSparseCNN",
        "num_epochs": 12,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 3,
        "gamma": 0.3,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "cnn_simplified_dualview": {
        "model": SimplifiedDualViewSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_simplified_dualview",
        "model_name": "SimplifiedDualViewSparseCNN",
        "num_epochs": 12,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 3,
        "gamma": 0.3,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "cnn_dense": {
        "model": DualViewDenseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_dense",
        "model_name": "DualViewDenseCNN",
        "num_epochs": 12,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 3,
        "gamma": 0.3,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "cnn_plane_summary": {
        "model": DualViewPlaneSummarySparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_plane_summary",
        "model_name": "DualViewPlaneSummarySparseCNN",
        "num_epochs": 12,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 3,
        "gamma": 0.3,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    # ── Cross-Attention Variants ─────────────────────────────────────

    "cnn_cross_attention": {
        "model": DualViewCrossAttentionSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
            num_heads=8,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_cross_attention",
        "model_name": "DualViewCrossAttentionSparseCNN",
        "num_epochs": 12,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 3,
        "gamma": 0.3,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "cnn_deep_cross_attention": {
        "model": DualViewDeepCrossAttentionSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
            num_heads=8,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_deep_cross_attention",
        "model_name": "DualViewDeepCrossAttentionSparseCNN",
        "num_epochs": 12,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 3,
        "gamma": 0.3,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "cnn_resnet_cross_attention": {
        "model": DualViewResNetCrossAttentionSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
            num_heads=8,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_resnet_cross_attention",
        "model_name": "DualViewResNetCrossAttentionSparseCNN",
        "num_epochs": 15,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 4,
        "gamma": 0.5,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "cnn_transformer_cross_attention": {
        "model": DualViewTransformerCrossAttnSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
            num_heads=8,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_transformer_cross_attention",
        "model_name": "DualViewTransformerCrossAttnSparseCNN",
        "num_epochs": 12,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 3,
        "gamma": 0.3,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "cnn_simplified_cross_attention": {
        "model": SimplifiedDualViewCrossAttentionSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
            num_heads=8,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_simplified_cross_attention",
        "model_name": "SimplifiedDualViewCrossAttentionSparseCNN",
        "num_epochs": 12,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 3,
        "gamma": 0.3,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "cnn_simplified_cross_attention_more_events": {
        "model": SimplifiedDualViewCrossAttentionSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
            num_heads=8,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_simplified_cross_attention_more_events",
        "model_name": "SimplifiedDualViewCrossAttentionSparseCNN",
        "num_epochs": 12,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 3,
        "gamma": 0.3,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 999999999,
    },
    
    "cnn_simplified_cross_attention_focal": {
        "model": SimplifiedDualViewCrossAttentionSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
            num_heads=8,
        ),
        "loss": BinaryFocalLoss(alpha=0.25, gamma=2.0),
        "model_type": "cnn_simplified_cross_attention_focal",
        "model_name": "SimplifiedDualViewCrossAttentionSparseCNN",
        "num_epochs": 12,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 3,
        "gamma": 0.3,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "cnn_positional_cross_attention": {
        "model": DualViewPositionalCrossAttentionSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
            num_heads=8,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_positional_cross_attention",
        "model_name": "DualViewPositionalCrossAttentionSparseCNN",
        "num_epochs": 15,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 4,
        "gamma": 0.5,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "cnn_multistage_cross_attention": {
        "model": DualViewMultiStageCrossAttentionSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
            num_heads=8,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_multistage_cross_attention",
        "model_name": "DualViewMultiStageCrossAttentionSparseCNN",
        "num_epochs": 12,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 3,
        "gamma": 0.3,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    # ── Other Fusion Architectures ───────────────────────────────────

    "cnn_3d_intersection": {
        "model": DualView3DIntersectionSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_3d_intersection",
        "model_name": "DualView3DIntersectionSparseCNN",
        "num_epochs": 12,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 3,
        "gamma": 0.3,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "cnn_cross_gate": {
        "model": DualViewCrossGateSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_cross_gate",
        "model_name": "DualViewCrossGateSparseCNN",
        "num_epochs": 12,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 3,
        "gamma": 0.3,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "cnn_hybrid_transformer": {
        "model": DualViewHybridTransformerSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
            num_heads=8,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_hybrid_transformer",
        "model_name": "DualViewHybridTransformerSparseCNN",
        "num_epochs": 12,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 3,
        "gamma": 0.3,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "cnn_multilayer_transformer": {
        "model": DualViewMultiLayerTransformerSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
            num_heads=8,
            num_transformer_layers=3,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_multilayer_transformer",
        "model_name": "DualViewMultiLayerTransformerSparseCNN",
        "num_epochs": 12,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 3,
        "gamma": 0.3,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "cnn_zipper": {
        "model": DualViewZipperSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            conv1d_channels=[32],
            fc_dims=[16],
            dropout=0.1,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_zipper",
        "model_name": "DualViewZipperSparseCNN",
        "num_epochs": 12,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 3,
        "gamma": 0.3,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    # ── Deeper ResNet Variants ───────────────────────────────────────

    "cnn_deep_resnet_cross_attention": {
        "model": DualViewDeepResNetCrossAttentionSparseCNN(
            in_channels=1,
            conv_channels=[32, 64, 128],
            fc_dims=[32, 16],
            dropout=0.1,
            num_heads=8,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_deep_resnet_cross_attention",
        "model_name": "DualViewDeepResNetCrossAttentionSparseCNN",
        "num_epochs": 15,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 4,
        "gamma": 0.5,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "cnn_resnet_dual_pool_cross_attention": {
        "model": DualViewResNetDualPoolCrossAttentionSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[32, 16],
            dropout=0.1,
            num_heads=8,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_resnet_dual_pool_cross_attention",
        "model_name": "DualViewResNetDualPoolCrossAttentionSparseCNN",
        "num_epochs": 15,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 4,
        "gamma": 0.5,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "cnn_resnet_multistage_cross_attention": {
        "model": DualViewResNetMultiStageCrossAttentionSparseCNN(
            in_channels=1,
            conv_channels=[32, 64],
            fc_dims=[16],
            dropout=0.1,
            num_heads=8,
        ),
        "loss": nn.CrossEntropyLoss(),
        "model_type": "cnn_resnet_multistage_cross_attention",
        "model_name": "DualViewResNetMultiStageCrossAttentionSparseCNN",
        "num_epochs": 15,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 4,
        "gamma": 0.5,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    # ── Graph Neural Networks ────────────────────────────────────────

    "gnn": {
        "model_factory": lambda metadata, in_channels: MinimumViableMINOSGNN(
            metadata=metadata,
            in_channels=in_channels,
            hidden_channels=24,
            num_layers=4,
            dropout=0.1,
        ),
        "loss": nn.CrossEntropyLoss(),
        "is_gnn": True,
        "model_type": "gnn",
        "model_name": "MinimumViableMINOSGNN",
        "num_epochs": 12,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 3,
        "gamma": 0.3,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "gnn_nugraph": {
        "model_factory": lambda metadata, in_channels: NuGraphInspiredBinaryGNN(
            metadata=metadata,
            in_channels=in_channels,
            hidden_channels=32,
            num_layers=4,
            dropout=0.1,
        ),
        "loss": nn.CrossEntropyLoss(),
        "is_gnn": True,
        "model_type": "gnn_nugraph",
        "model_name": "NuGraphInspiredBinaryGNN",
        "num_epochs": 15,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 4,
        "gamma": 0.5,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "sum",
        "max_events": 12000,
    },

    "gnn_timing": {
        "model_factory": lambda metadata, in_channels: TimingAwareGNN(
            in_channels=in_channels,
            edge_dim=6,
            hidden_channels=48,
            num_layers=3,
            dropout=0.15,
        ),
        "loss": nn.CrossEntropyLoss(),
        "is_gnn": True,
        "model_type": "gnn_timing",
        "model_name": "TimingAwareGNN",
        "num_epochs": 20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 5,
        "gamma": 0.5,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "timing",
        "cache_path": "data/cache/minos_uview_multi_view_graph_timing.pt",
        "max_events": 12000,
    },

    # ── Timing ablation study (added for diagnostics, does not touch gnn_timing) ──
    # Both configs below use TimingAwareGNNV2 (correctness-fixed copy of
    # TimingAwareGNN, see src/models.py) with identical hyperparameters, so
    # comparing them isolates the effect of the timing-derived feature values
    # themselves: same architecture, same graph topology, same event subset.
    "gnn_timing_v2": {
        "model_factory": lambda metadata, in_channels: TimingAwareGNNV2(
            in_channels=in_channels,
            edge_dim=6,
            hidden_channels=48,
            num_layers=3,
            dropout=0.15,
        ),
        "loss": nn.CrossEntropyLoss(),
        "is_gnn": True,
        "model_type": "gnn_timing_v2",
        "model_name": "TimingAwareGNNV2",
        "num_epochs": 20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 5,
        "gamma": 0.5,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "timing",
        "ablate_timing": False,
        "cache_path": "data/cache/minos_uview_multi_view_graph_timing.pt",
        "max_events": 12000,
    },

    "gnn_timing_v2_ablated": {
        "model_factory": lambda metadata, in_channels: TimingAwareGNNV2(
            in_channels=in_channels,
            edge_dim=6,
            hidden_channels=48,
            num_layers=3,
            dropout=0.15,
        ),
        "loss": nn.CrossEntropyLoss(),
        "is_gnn": True,
        "model_type": "gnn_timing_v2_ablated",
        "model_name": "TimingAwareGNNV2",
        "num_epochs": 20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 5,
        "gamma": 0.5,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "timing",
        "ablate_timing": True,
        "cache_path": "data/cache/minos_uview_multi_view_graph_timing_ablated.pt",
        "max_events": 12000,
    },

    # ── v3: >=1-hit event recovery + view-sign bug fix in dataset.py's ──
    # _build_timing_graph, same TimingAwareGNNV2 architecture/size as v2 (no
    # capacity increase) so the comparison isolates the effect of the two
    # dataset-level fixes. New cache paths only -- gnn_timing/gnn_timing_v2/
    # gnn_timing_v2_ablated and their cache files are untouched.
    "gnn_timing_v3": {
        "model_factory": lambda metadata, in_channels: TimingAwareGNNV2(
            in_channels=in_channels,
            edge_dim=6,
            hidden_channels=48,
            num_layers=3,
            dropout=0.15,
        ),
        "loss": nn.CrossEntropyLoss(),
        "is_gnn": True,
        "model_type": "gnn_timing_v3",
        "model_name": "TimingAwareGNNV2",
        "num_epochs": 20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 5,
        "gamma": 0.5,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "timing",
        "ablate_timing": False,
        "cache_path": "data/cache/minos_uview_multi_view_graph_timing_v3.pt",
        "max_events": 12000,
    },

    "gnn_timing_v3_ablated": {
        "model_factory": lambda metadata, in_channels: TimingAwareGNNV2(
            in_channels=in_channels,
            edge_dim=6,
            hidden_channels=48,
            num_layers=3,
            dropout=0.15,
        ),
        "loss": nn.CrossEntropyLoss(),
        "is_gnn": True,
        "model_type": "gnn_timing_v3_ablated",
        "model_name": "TimingAwareGNNV2",
        "num_epochs": 20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 5,
        "gamma": 0.5,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "timing",
        "ablate_timing": True,
        "cache_path": "data/cache/minos_uview_multi_view_graph_timing_v3_ablated.pt",
        "max_events": 12000,
    },

    # ── Champion architecture, re-run on the corrected event set ──────
    # TimingAwareGNN (the v1 architecture) tops the leaderboard at 0.9396, but
    # that run used the pre-v3 dataset builder, which required BOTH views to be
    # non-empty and so trained/validated on 11,694 events. The v3 fix and the
    # hitset mode keep any event with >=1 hit -> 11,925 events, and the 231
    # extra events are sparse single-view ones: low-energy and hard. The smaller
    # set is therefore systematically easier, and 0.9396 is not comparable with
    # anything in the 11,925-event family.
    #
    # This config is the same architecture pointed at the already-built v3 cache,
    # purely so the smallest model on the board (25,802 effective parameters --
    # its logged 53,738 includes a dead `convs` ModuleList that forward() never
    # calls) can be compared like-for-like against the hitset transformer.
    # No new model code and no cache rebuild.
    "gnn_timing_v1_on_v3": {
        "model_factory": lambda metadata, in_channels: TimingAwareGNN(
            in_channels=in_channels,
            edge_dim=6,
            hidden_channels=48,
            num_layers=3,
            dropout=0.15,
        ),
        "loss": nn.CrossEntropyLoss(),
        "is_gnn": True,
        "model_type": "gnn_timing_v1_on_v3",
        "model_name": "TimingAwareGNN",
        "num_epochs": 20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 5,
        "gamma": 0.5,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "timing",
        "ablate_timing": False,
        "cache_path": "data/cache/minos_uview_multi_view_graph_timing_v3.pt",
        "max_events": 12000,
    },

    # ── Aux-supervised head on the smallest model ─────────────────────
    # Same TimingAwareGNN architecture and hyperparameters as
    # gnn_timing_v1_on_v3, differing ONLY in aux_weight and the cache (which
    # carries per-hit labels but otherwise produces bit-identical graphs).
    #
    # Because the hit head is constructed last and only when aux_weight > 0,
    # every shared layer initialises identically to the control -- so the
    # already-trained gnn_timing_v1_on_v3 (0.9324, seed 42) IS the control for
    # this run. Train with --seed 42 to keep that true.
    #
    # aux_weight=0.5 matches transformer_hitset_aux, where dense per-hit
    # supervision was worth a measured +0.0047 (95% CI [+0.0016, +0.0080],
    # p=0.0028) -- the only statistically significant result on this dataset.
    "gnn_timing_aux": {
        "model_factory": lambda metadata, in_channels: TimingAwareGNN(
            in_channels=in_channels,
            edge_dim=6,
            hidden_channels=48,
            num_layers=3,
            dropout=0.15,
            aux_weight=0.5,
        ),
        "loss": nn.CrossEntropyLoss(),
        "is_gnn": True,
        "model_type": "gnn_timing_aux",
        "model_name": "TimingAwareGNN",
        "num_epochs": 20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "step_size": 5,
        "gamma": 0.5,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "timing_aux",
        "ablate_timing": False,
        "cache_path": "data/cache/minos_timing_aux.pt",
        "max_events": 12000,
    },

    # ── Hit-level set transformer + per-hit primary-lepton supervision ──
    # Motivated by the plateau across the 25 rows above (ROC-AUC 0.924-0.940
    # spanning CNNs, cross-attention and GNNs, 23k-988k params) and by the
    # gnn_timing_v3 vs. _ablated pair showing timing contributes ~nothing.
    # The new ingredient is dense supervision, not the architecture: thstp
    # gives a soft per-strip primary-lepton charge fraction, and an oracle that
    # merely counts truth-tagged hits scores ROC-AUC 0.9901 (measured), so the
    # separation is present at strip level and is not being extracted.
    #
    # The two configs differ ONLY in aux_weight and share a cache, so the
    # comparison isolates the supervision -- same architecture, same features,
    # same event subset. train.py keys datasets by cache_path, so running them
    # together loads the data once. Sized for CPU (no GPU on this machine).
    "transformer_hitset": {
        "model_factory": lambda metadata, in_channels: HitSetTransformer(
            in_channels=in_channels,
            d_model=64,
            num_layers=4,
            num_heads=4,
            dropout=0.1,
            max_hits=256,
            use_pair_bias=True,
            aux_weight=0.0,
        ),
        "loss": nn.CrossEntropyLoss(),
        "is_gnn": True,
        "model_type": "transformer_hitset",
        "model_name": "HitSetTransformer",
        "num_epochs": 20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "step_size": 5,
        "gamma": 0.5,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "hitset",
        "cache_path": "data/cache/minos_hitset.pt",
        "max_events": 12000,
    },

    "transformer_hitset_aux": {
        "model_factory": lambda metadata, in_channels: HitSetTransformer(
            in_channels=in_channels,
            d_model=64,
            num_layers=4,
            num_heads=4,
            dropout=0.1,
            max_hits=256,
            use_pair_bias=True,
            aux_weight=0.5,
        ),
        "loss": nn.CrossEntropyLoss(),
        "is_gnn": True,
        "model_type": "transformer_hitset_aux",
        "model_name": "HitSetTransformer",
        "num_epochs": 20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "step_size": 5,
        "gamma": 0.5,
        "use_class_weights": False,
        "use_energy_weights": False,
        "feature_mode": "hitset",
        "cache_path": "data/cache/minos_hitset.pt",
        "max_events": 12000,
    },
}


def get_config(name):
    """Look up a model config dict by name."""
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model config '{name}'. Available: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[name]
