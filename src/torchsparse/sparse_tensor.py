from typing import Optional, Union, Tuple
import torch


class SparseTensor:
    """
    TorchSparse SparseTensor data structure (MIT HAN Lab API).
    Encapsulates sparse hit features (.feats / .F) and coordinates (.coords / .C).
    coords format: [N, 3] tensor containing [batch_idx, plane, strip].
    """

    def __init__(self, feats: torch.Tensor, coords: torch.Tensor, spatial_range: Optional[Tuple[int, int]] = (486, 192)):
        self.feats = feats
        self.coords = coords
        self.spatial_range = spatial_range

    @property
    def F(self) -> torch.Tensor:
        return self.feats

    @F.setter
    def F(self, val: torch.Tensor):
        self.feats = val

    @property
    def C(self) -> torch.Tensor:
        return self.coords

    @C.setter
    def C(self, val: torch.Tensor):
        self.coords = val

    def to(self, device: Union[str, torch.device]):
        self.feats = self.feats.to(device)
        self.coords = self.coords.to(device)
        return self

    def __repr__(self) -> str:
        return f"SparseTensor(feats={tuple(self.feats.shape)}, coords={tuple(self.coords.shape)})"
