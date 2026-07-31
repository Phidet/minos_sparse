import torch
import torch.nn as nn
from src.torchsparse.sparse_tensor import SparseTensor


class ReLU(nn.Module):
    """
    TorchSparse ReLU activation (MIT HAN Lab API).
    Applies ReLU elementwise to input.feats of SparseTensor.
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, input_tensor: SparseTensor) -> SparseTensor:
        out_feats = self.relu(input_tensor.feats)
        return SparseTensor(feats=out_feats, coords=input_tensor.coords, spatial_range=input_tensor.spatial_range)
