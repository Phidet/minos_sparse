"""
MINOS U-View Sparse CNN Package (MIT TorchSparse Engine)
---------------------------------------------------------
Streamlined dataset parsing, TorchSparse SparseTensor convolutions, and evaluation for MINOS U-view event displays.
"""

from .torchsparse import SparseTensor
from .dataset import (
    MINOSSingleViewDataset,
    sparse_uview_collate_fn,
    create_uview_dataloaders
)
from .models import (
    SimpleUViewSparseCNN
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
    'SimpleUViewSparseCNN',
    'train_epoch',
    'validate_epoch',
    'compute_metrics',
    'train_model'
]
