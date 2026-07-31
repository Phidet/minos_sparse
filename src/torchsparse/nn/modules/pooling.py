import torch
import torch.nn as nn
from src.torchsparse.sparse_tensor import SparseTensor


class GlobalAvgPooling(nn.Module):
    """
    TorchSparse Global Average Pooling module (MIT HAN Lab API).
    Aggregates point features per batch index, returning a dense 2D tensor [batch_size, num_channels].
    """

    def forward(self, input_tensor: SparseTensor) -> torch.Tensor:
        coords = input_tensor.coords
        feats = input_tensor.feats
        batch_idx = coords[:, 0].long()
        num_channels = feats.shape[1]
        device = feats.device

        batch_size = int(batch_idx.max().item()) + 1 if coords.numel() > 0 else 1

        pooled = torch.zeros(batch_size, num_channels, device=device, dtype=feats.dtype)
        counts = torch.zeros(batch_size, 1, device=device, dtype=feats.dtype)

        pooled.index_add_(0, batch_idx, feats)
        counts.index_add_(0, batch_idx, torch.ones_like(feats[:, :1]))
        counts = torch.clamp(counts, min=1.0)
        return pooled / counts


class GlobalMaxPooling(nn.Module):
    """
    TorchSparse Global Max Pooling module (MIT HAN Lab API).
    Aggregates peak point features per batch index, returning a dense 2D tensor [batch_size, num_channels].
    """

    def forward(self, input_tensor: SparseTensor) -> torch.Tensor:
        coords = input_tensor.coords
        feats = input_tensor.feats
        batch_idx = coords[:, 0].long()
        num_channels = feats.shape[1]
        device = feats.device

        batch_size = int(batch_idx.max().item()) + 1 if coords.numel() > 0 else 1

        idx_exp = batch_idx.unsqueeze(1).expand_as(feats)
        max_pooled = torch.full((batch_size, num_channels), float("-inf"), device=device, dtype=feats.dtype)
        max_pooled.scatter_reduce_(0, idx_exp, feats, reduce="amax", include_self=False)
        return torch.where(torch.isinf(max_pooled), torch.zeros_like(max_pooled), max_pooled)
