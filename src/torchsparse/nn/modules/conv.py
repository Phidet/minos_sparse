from typing import Tuple, Optional
import torch
import torch.nn as nn
from src.torchsparse.sparse_tensor import SparseTensor


class SubMConv2d(nn.Module):
    """
    TorchSparse 2D Submanifold Sparse Convolution (MIT HAN Lab API).
    Accepts SparseTensor(feats, coords) and returns a new SparseTensor with updated features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.spatial_shape = spatial_shape

        k = kernel_size // 2
        offsets = []
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                offsets.append([dy, dx])
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.weight = nn.Parameter(torch.randn(len(offsets), in_channels, out_channels) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, input_tensor: SparseTensor) -> SparseTensor:
        coords = input_tensor.coords
        feats = input_tensor.feats
        N = coords.size(0)
        device = feats.device
        H, W = self.spatial_shape

        linear_idx = coords[:, 0].long() * (H * W) + coords[:, 1].long() * W + coords[:, 2].long()
        max_idx = linear_idx.max().item() + 1
        lookup = torch.full((max_idx,), -1, dtype=torch.long, device=device)
        lookup[linear_idx] = torch.arange(N, device=device)

        out_feats = torch.zeros(N, self.out_channels, device=device, dtype=feats.dtype)

        for k_idx, off in enumerate(self.offsets):
            dy, dx = off[0].item(), off[1].item()
            np_plane = coords[:, 1].long() + dy
            np_strip = coords[:, 2].long() + dx

            valid = (np_plane >= 0) & (np_plane < H) & (np_strip >= 0) & (np_strip < W)
            neighbor_linear = coords[:, 0].long() * (H * W) + np_plane * W + np_strip
            valid = valid & (neighbor_linear < max_idx)

            valid_indices = torch.where(valid)[0]
            if len(valid_indices) == 0:
                continue

            n_linear = neighbor_linear[valid_indices]
            src_hit_idx = lookup[n_linear]
            matched_mask = src_hit_idx >= 0

            if matched_mask.any():
                dst_hits = valid_indices[matched_mask]
                src_hits = src_hit_idx[matched_mask]
                gathered = feats[src_hits]
                contrib = torch.matmul(gathered, self.weight[k_idx])
                out_feats.index_add_(0, dst_hits, contrib)

        if self.bias is not None:
            out_feats = out_feats + self.bias

        return SparseTensor(feats=out_feats, coords=coords, spatial_range=input_tensor.spatial_range)
