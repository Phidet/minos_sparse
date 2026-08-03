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
    DualViewPlaneSummarySparseCNN,
    DualViewCrossAttentionSparseCNN,
    DualView3DIntersectionSparseCNN,
    SparseCrossGate,
    DualViewCrossGateSparseCNN,
    DualViewHybridTransformerSparseCNN,
    DualViewMultiLayerTransformerSparseCNN,
    MinimumViableMINOSGNN,
    NuGraphInspiredBinaryGNN,
)
from .trainer import (
    train_epoch,
    validate_epoch,
    compute_metrics,
    train_model
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
    'DualViewPlaneSummarySparseCNN',
    'DualViewCrossAttentionSparseCNN',
    'DualView3DIntersectionSparseCNN',
    'SparseCrossGate',
    'DualViewCrossGateSparseCNN',
    'DualViewHybridTransformerSparseCNN',
    'DualViewMultiLayerTransformerSparseCNN',
    'MinimumViableMINOSGNN',
    'NuGraphInspiredBinaryGNN',
    'train_epoch',
    'validate_epoch',
    'compute_metrics',
    'train_model'
]
