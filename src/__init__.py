"""
MINOS U-View Sparse CNN Package (MIT TorchSparse Engine)
---------------------------------------------------------
Streamlined dataset parsing, TorchSparse SparseTensor convolutions, and evaluation for MINOS U-view event displays.
"""

from .torchsparse import SparseTensor
from .dataset import (
    MINOSSingleViewDataset,
    sparse_uview_collate_fn,
    create_uview_dataloaders,
    MINOSMultiViewGraphDataset,
    create_multiview_gnn_dataloaders,
)
from .models import (
    SimpleUViewSparseCNN,
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
    SparseCrossGate,
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
)
from .model_configs import (
    MODEL_CONFIGS,
    DATASET_CONFIG,
    get_config,
)
from .trainer import (
    train_epoch,
    validate_epoch,
    compute_metrics,
    train_model,
    auto_commit_and_get_hash,
    save_model_checkpoint,
    log_experiment,
    display_leaderboard,
)

__all__ = [
    'SparseTensor',
    'MINOSSingleViewDataset',
    'sparse_uview_collate_fn',
    'create_uview_dataloaders',
    'MINOSMultiViewGraphDataset',
    'create_multiview_gnn_dataloaders',
    'SimpleUViewSparseCNN',
    'DualViewSparseCNN',
    'SimplifiedDualViewSparseCNN',
    'DualViewDenseCNN',
    'DualViewPlaneSummarySparseCNN',
    'DualViewCrossAttentionSparseCNN',
    'DualViewDeepCrossAttentionSparseCNN',
    'DualViewResNetCrossAttentionSparseCNN',
    'DualViewTransformerCrossAttnSparseCNN',
    'SimplifiedDualViewCrossAttentionSparseCNN',
    'DualViewPositionalCrossAttentionSparseCNN',
    'DualViewMultiStageCrossAttentionSparseCNN',
    'DualView3DIntersectionSparseCNN',
    'SparseCrossGate',
    'DualViewCrossGateSparseCNN',
    'DualViewHybridTransformerSparseCNN',
    'DualViewMultiLayerTransformerSparseCNN',
    'DualViewZipperSparseCNN',
    'MinimumViableMINOSGNN',
    'NuGraphInspiredBinaryGNN',
    'DualViewDeepResNetCrossAttentionSparseCNN',
    'DualViewResNetDualPoolCrossAttentionSparseCNN',
    'DualViewResNetMultiStageCrossAttentionSparseCNN',
    'TimingAwareGNN',
    'MODEL_CONFIGS',
    'DATASET_CONFIG',
    'get_config',
    'train_epoch',
    'validate_epoch',
    'compute_metrics',
    'train_model',
    'auto_commit_and_get_hash',
    'save_model_checkpoint',
    'log_experiment',
    'display_leaderboard',
]







