import torch
import torch.nn as nn
from src.torchsparse.sparse_tensor import SparseTensor


class BatchNorm(nn.Module):
    """
    TorchSparse BatchNorm module (MIT HAN Lab API).
    Applies BatchNorm1d to input.feats of SparseTensor.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, input_tensor: SparseTensor) -> SparseTensor:
        out_feats = self.bn(input_tensor.feats)
        return SparseTensor(feats=out_feats, coords=input_tensor.coords, spatial_range=input_tensor.spatial_range)
