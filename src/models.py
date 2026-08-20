from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn

from src.torchsparse import SparseTensor
from src.dataset import get_hit_lepton_frac, get_has_lepton_truth
import src.torchsparse.nn as spnn

try:
    from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool, global_max_pool, EdgeConv, GraphNorm
    from torch_geometric.data import HeteroData
    from torch_geometric.utils import to_dense_batch
except ImportError:
    HeteroConv = None
    SAGEConv = None
    global_mean_pool = None
    global_max_pool = None
    EdgeConv = None
    GraphNorm = None
    HeteroData = None
    to_dense_batch = None



class SimpleUViewSparseCNN(nn.Module):
    """
    Simple, Standard U-View Sparse CNN built 100% on spnn modules.
    Uses spnn.SubMConv2d, spnn.BatchNorm, spnn.ReLU, and spnn.GlobalAvgPooling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32],
        fc_dims: List[int] = [16],
        num_classes: int = 2,
        dropout: float = 0.1,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels

        self.net = nn.ModuleList()
        prev_c = in_channels
        for out_c in conv_channels:
            self.net.append(spnn.SubMConv2d(prev_c, out_c, kernel_size=3, spatial_shape=spatial_shape))
            self.net.append(spnn.BatchNorm(out_c))
            self.net.append(spnn.ReLU())
            prev_c = out_c

        self.pool = spnn.GlobalAvgPooling()
        pooled_dim = conv_channels[-1] if len(conv_channels) > 0 else in_channels

        classifier_layers = []
        current_dim = pooled_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, input_tensor: SparseTensor) -> torch.Tensor:
        x = input_tensor
        for layer in self.net:
            x = layer(x)

        pooled = self.pool(x)
        return self.classifier(pooled)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualViewSparseCNN(nn.Module):
    """
    Dual-View Sparse CNN for MINOS binary classification.

    Processes two detector views (e.g., U-view and V-view) using two separate
    TorchSparse SubMConv2d feature extraction backbones, pools each view's spatial
    features via GlobalAvgPooling, and combines both perspective embeddings in a
    shared fully connected classification readout.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32],
        fc_dims: List[int] = [16],
        num_classes: int = 2,
        dropout: float = 0.1,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels

        self.net_a = nn.ModuleList()
        prev_c = in_channels
        for out_c in conv_channels:
            self.net_a.append(spnn.SubMConv2d(prev_c, out_c, kernel_size=3, spatial_shape=spatial_shape))
            self.net_a.append(spnn.BatchNorm(out_c))
            self.net_a.append(spnn.ReLU())
            prev_c = out_c

        self.net_b = nn.ModuleList()
        prev_c = in_channels
        for out_c in conv_channels:
            self.net_b.append(spnn.SubMConv2d(prev_c, out_c, kernel_size=3, spatial_shape=spatial_shape))
            self.net_b.append(spnn.BatchNorm(out_c))
            self.net_b.append(spnn.ReLU())
            prev_c = out_c

        self.pool = spnn.GlobalAvgPooling()
        pooled_dim = conv_channels[-1] if len(conv_channels) > 0 else in_channels
        fusion_dim = pooled_dim * 4

        classifier_layers = []
        current_dim = fusion_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_sparse_tensors(self, data) -> Tuple[SparseTensor, SparseTensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError("Unsupported data format for DualViewSparseCNN. Expected HeteroData or tuple of SparseTensors.")

    def forward(self, data) -> torch.Tensor:
        tensor_a, tensor_b = self._extract_sparse_tensors(data)

        x_a = tensor_a
        for layer in self.net_a:
            x_a = layer(x_a)
        pooled_a = self.pool(x_a)

        x_b = tensor_b
        for layer in self.net_b:
            x_b = layer(x_b)
        pooled_b = self.pool(x_b)

        fused = torch.cat([
            pooled_a,
            pooled_b,
            torch.abs(pooled_a - pooled_b),
            pooled_a * pooled_b,
        ], dim=-1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualViewTransformerCrossAttnSparseCNN(nn.Module):
    """
    Dual-View Sparse CNN with Full 1D Transformer Encoder Blocks (Self-Attn + Cross-Attn + FFN + PosEmbed).

    Incorporates 1D positional plane embeddings, 1D self-attention, 1D cross-attention,
    and Feed-Forward Networks (FFN with GELU activations) across Z-axis plane summaries.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [32, 64],
        fc_dims: List[int] = [16],
        num_classes: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.spatial_shape = spatial_shape

        c1 = conv_channels[0] if len(conv_channels) > 0 else 32
        c2 = conv_channels[1] if len(conv_channels) > 1 else c1

        self.block1_a = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )
        self.block1_b = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )

        # 1D Positional Embedding for plane index z in [0, spatial_shape[0]-1]
        self.pos_embed = nn.Parameter(torch.zeros(1, spatial_shape[0], c1))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Self-Attention modules
        self.self_attn_a = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.self_attn_b = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.norm_self_a = nn.LayerNorm(c1)
        self.norm_self_b = nn.LayerNorm(c1)

        # Cross-Attention modules
        self.cross_attn_a = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.cross_attn_b = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.norm_cross_a = nn.LayerNorm(c1)
        self.norm_cross_b = nn.LayerNorm(c1)

        # Feed-Forward Networks (FFN)
        self.ffn_a = nn.Sequential(
            nn.Linear(c1, c1 * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c1 * 4, c1),
            nn.Dropout(dropout),
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(c1, c1 * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c1 * 4, c1),
            nn.Dropout(dropout),
        )
        self.norm_ffn_a = nn.LayerNorm(c1)
        self.norm_ffn_b = nn.LayerNorm(c1)

        self.block2_a = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )
        self.block2_b = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )

        self.pool = spnn.GlobalAvgPooling()
        pooled_dim = c2
        fusion_dim = pooled_dim * 4

        classifier_layers = []
        current_dim = fusion_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_sparse_tensors(self, data) -> Tuple[SparseTensor, SparseTensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError("Unsupported data format for DualViewTransformerCrossAttnSparseCNN.")

    def forward(self, data) -> torch.Tensor:
        tensor_a, tensor_b = self._extract_sparse_tensors(data)

        x_a = self.block1_a(tensor_a)
        x_b = self.block1_b(tensor_b)

        batch_size = int(max(
            x_a.C[:, 0].max().item() if x_a.C.numel() > 0 else 0,
            x_b.C[:, 0].max().item() if x_b.C.numel() > 0 else 0
        )) + 1

        summary_a, keys_a = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], self.spatial_shape[0], num_batch=batch_size)
        summary_b, keys_b = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], self.spatial_shape[0], num_batch=batch_size)

        max_planes = self.spatial_shape[0]

        seq_a = summary_a.view(batch_size, max_planes, -1) + self.pos_embed
        seq_b = summary_b.view(batch_size, max_planes, -1) + self.pos_embed

        # 1D Self-Attention
        self_out_a, _ = self.self_attn_a(query=seq_a, key=seq_a, value=seq_a)
        self_out_b, _ = self.self_attn_b(query=seq_b, key=seq_b, value=seq_b)
        seq_a = self.norm_self_a(seq_a + self_out_a)
        seq_b = self.norm_self_b(seq_b + self_out_b)

        # 1D Cross-Attention
        cross_out_a, _ = self.cross_attn_a(query=seq_a, key=seq_b, value=seq_b)
        cross_out_b, _ = self.cross_attn_b(query=seq_b, key=seq_a, value=seq_a)
        seq_a = self.norm_cross_a(seq_a + cross_out_a)
        seq_b = self.norm_cross_b(seq_b + cross_out_b)

        # Feed-Forward Network (FFN)
        ffn_out_a = self.ffn_a(seq_a)
        ffn_out_b = self.ffn_b(seq_b)
        seq_a = self.norm_ffn_a(seq_a + ffn_out_a).view(-1, seq_a.size(-1))
        seq_b = self.norm_ffn_b(seq_b + ffn_out_b).view(-1, seq_b.size(-1))

        mod_F_a = x_a.F + seq_a[keys_a]
        mod_F_b = x_b.F + seq_b[keys_b]

        x_a = SparseTensor(feats=mod_F_a, coords=x_a.C)
        x_b = SparseTensor(feats=mod_F_b, coords=x_b.C)

        x_a = self.block2_a(x_a)
        x_b = self.block2_b(x_b)

        pooled_a = self.pool(x_a)
        pooled_b = self.pool(x_b)

        fused = torch.cat([
            pooled_a,
            pooled_b,
            torch.abs(pooled_a - pooled_b),
            pooled_a * pooled_b,
        ], dim=-1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimplifiedDualViewSparseCNN(nn.Module):
    """
    Simplified Dual-View Sparse CNN for MINOS binary classification.

    Processes two detector views (e.g., U-view and V-view) using a single shared
    TorchSparse SubMConv2d feature extraction backbone (`self.net`). Each view's
    spatial features are extracted by `self.net` and pooled via `self.pool` (GlobalAvgPooling).
    The resulting view embeddings are combined in a shared fully connected classification readout.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32],
        fc_dims: List[int] = [16],
        num_classes: int = 2,
        dropout: float = 0.1,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels

        self.net = nn.ModuleList()
        prev_c = in_channels
        for out_c in conv_channels:
            self.net.append(spnn.SubMConv2d(prev_c, out_c, kernel_size=3, spatial_shape=spatial_shape))
            self.net.append(spnn.BatchNorm(out_c))
            self.net.append(spnn.ReLU())
            prev_c = out_c

        self.pool = spnn.GlobalAvgPooling()
        pooled_dim = conv_channels[-1] if len(conv_channels) > 0 else in_channels
        fusion_dim = pooled_dim * 4

        classifier_layers = []
        current_dim = fusion_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_sparse_tensors(self, data) -> Tuple[SparseTensor, SparseTensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError("Unsupported data format for SimplifiedDualViewSparseCNN. Expected HeteroData or tuple of SparseTensors.")

    def forward(self, data) -> torch.Tensor:
        tensor_a, tensor_b = self._extract_sparse_tensors(data)

        x_a = tensor_a
        for layer in self.net:
            x_a = layer(x_a)
        pooled_a = self.pool(x_a)

        x_b = tensor_b
        for layer in self.net:
            x_b = layer(x_b)
        pooled_b = self.pool(x_b)

        fused = torch.cat([
            pooled_a,
            pooled_b,
            torch.abs(pooled_a - pooled_b),
            pooled_a * pooled_b,
        ], dim=-1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualViewDenseCNN(nn.Module):
    """
    Dual-View Dense 2D CNN for MINOS binary classification.

    Rasterizes MINOS detector hits into dense 2D image tensors [Batch, Channels, 486, 192]
    for both detector views (e.g., U-view and V-view). Each view is processed by a standard
    2D CNN backbone (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d), pooled via
    nn.AdaptiveAvgPool2d((1, 1)), and fused in a fully connected classification readout.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32],
        fc_dims: List[int] = [16],
        num_classes: int = 2,
        dropout: float = 0.1,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.spatial_shape = spatial_shape

        # View A 2D CNN backbone
        net_a_layers = []
        prev_c = in_channels
        for out_c in conv_channels:
            net_a_layers.append(nn.Conv2d(prev_c, out_c, kernel_size=3, padding=1))
            net_a_layers.append(nn.BatchNorm2d(out_c))
            net_a_layers.append(nn.ReLU())
            net_a_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_c = out_c
        self.net_a = nn.Sequential(*net_a_layers)

        # View B 2D CNN backbone
        net_b_layers = []
        prev_c = in_channels
        for out_c in conv_channels:
            net_b_layers.append(nn.Conv2d(prev_c, out_c, kernel_size=3, padding=1))
            net_b_layers.append(nn.BatchNorm2d(out_c))
            net_b_layers.append(nn.ReLU())
            net_b_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_c = out_c
        self.net_b = nn.Sequential(*net_b_layers)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        pooled_dim = conv_channels[-1] if len(conv_channels) > 0 else in_channels
        fusion_dim = pooled_dim * 4

        classifier_layers = []
        current_dim = fusion_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_dense_images(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts view_a and view_b coordinates and features, returning dense
        2D image tensors of shape [Batch, Channels, H, W].
        """
        H, W = self.spatial_shape

        if isinstance(data, (tuple, list)) and len(data) == 2:
            tensor_a, tensor_b = data[0], data[1]
            coords_a, feats_a = tensor_a.C, tensor_a.F
            coords_b, feats_b = tensor_b.C, tensor_b.F
            batch_size = int(max(coords_a[:, 0].max().item(), coords_b[:, 0].max().item())) + 1 if coords_a.numel() > 0 or coords_b.numel() > 0 else 1
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                tensor_a, tensor_b = data["view_a"], data["view_b"]
                coords_a, feats_a = tensor_a.C, tensor_a.F
                coords_b, feats_b = tensor_b.C, tensor_b.F
                batch_size = int(max(coords_a[:, 0].max().item(), coords_b[:, 0].max().item())) + 1 if coords_a.numel() > 0 or coords_b.numel() > 0 else 1
            else:
                raise TypeError("Unsupported dict input type for DualViewDenseCNN.")
        elif HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]

            if hasattr(data, "num_graphs") and data.num_graphs is not None:
                batch_size = data.num_graphs
            elif hasattr(data, "y") and data.y is not None:
                batch_size = data.y.size(0)
            else:
                max_a = int(batch_a.max().item()) + 1 if batch_a.numel() > 0 else 1
                max_b = int(batch_b.max().item()) + 1 if batch_b.numel() > 0 else 1
                batch_size = max(max_a, max_b)
        else:
            raise TypeError("Unsupported data format for DualViewDenseCNN. Expected HeteroData or tuple/dict of SparseTensors.")

        dense_a = self._coords_to_dense(coords_a, feats_a, batch_size, H, W)
        dense_b = self._coords_to_dense(coords_b, feats_b, batch_size, H, W)
        return dense_a, dense_b

    def _coords_to_dense(self, coords: torch.Tensor, feats: torch.Tensor, batch_size: int, H: int, W: int) -> torch.Tensor:
        dense = torch.zeros((batch_size, self.in_channels, H, W), dtype=feats.dtype, device=feats.device)
        if coords.numel() > 0 and feats.numel() > 0:
            b_idx = coords[:, 0]
            h_idx = torch.clamp(coords[:, 1], 0, H - 1)
            w_idx = torch.clamp(coords[:, 2], 0, W - 1)
            dense[b_idx, :, h_idx, w_idx] = feats[:, :self.in_channels]
        return dense

    def forward(self, data) -> torch.Tensor:
        dense_a, dense_b = self._extract_dense_images(data)

        x_a = self.net_a(dense_a)
        pooled_a = self.pool(x_a).flatten(1)

        x_b = self.net_b(dense_b)
        pooled_b = self.pool(x_b).flatten(1)

        fused = torch.cat([
            pooled_a,
            pooled_b,
            torch.abs(pooled_a - pooled_b),
            pooled_a * pooled_b,
        ], dim=-1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _scatter_plane_summary(feats: torch.Tensor, batch_idx: torch.Tensor, plane_idx: torch.Tensor, max_planes: int = 486, num_batch: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes a per-(batch, plane) mean feature vector without external dependencies."""
    plane_keys = batch_idx * max_planes + plane_idx
    if num_batch is not None:
        batch_size = max(1, num_batch)
    else:
        batch_size = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 1
    total_planes = batch_size * max_planes

    sum_feats = torch.zeros((total_planes, feats.size(-1)), dtype=feats.dtype, device=feats.device)
    sum_feats.index_add_(0, plane_keys, feats)

    counts = torch.zeros((total_planes, 1), dtype=feats.dtype, device=feats.device)
    counts.index_add_(0, plane_keys, torch.ones((feats.size(0), 1), dtype=feats.dtype, device=feats.device))

    mean_feats = sum_feats / torch.clamp(counts, min=1.0)
    return mean_feats, plane_keys


class DualViewPlaneSummarySparseCNN(nn.Module):
    """
    Dual-View Sparse CNN with Intermediate Plane-Wise Summary Feature Injection.

    At intermediate conv stages, strip features per plane z are collapsed into
    1D plane summaries. Cross-view gating signals are computed per plane and injected
    back into the sparse hit features of the opposite view.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32],
        fc_dims: List[int] = [16],
        num_classes: int = 2,
        dropout: float = 0.1,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.spatial_shape = spatial_shape

        c1 = conv_channels[0] if len(conv_channels) > 0 else 16
        c2 = conv_channels[1] if len(conv_channels) > 1 else c1

        self.block1_a = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )
        self.block1_b = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )

        self.gate_a = nn.Sequential(
            nn.Linear(2 * c1, c1),
            nn.ReLU(),
            nn.Linear(c1, c1),
            nn.Sigmoid(),
        )
        self.gate_b = nn.Sequential(
            nn.Linear(2 * c1, c1),
            nn.ReLU(),
            nn.Linear(c1, c1),
            nn.Sigmoid(),
        )

        self.block2_a = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )
        self.block2_b = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )

        self.pool = spnn.GlobalAvgPooling()
        pooled_dim = c2
        fusion_dim = pooled_dim * 4

        classifier_layers = []
        current_dim = fusion_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_sparse_tensors(self, data) -> Tuple[SparseTensor, SparseTensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError("Unsupported data format for DualViewPlaneSummarySparseCNN.")

    def forward(self, data) -> torch.Tensor:
        tensor_a, tensor_b = self._extract_sparse_tensors(data)

        x_a = self.block1_a(tensor_a)
        x_b = self.block1_b(tensor_b)

        batch_size = int(max(
            x_a.C[:, 0].max().item() if x_a.C.numel() > 0 else 0,
            x_b.C[:, 0].max().item() if x_b.C.numel() > 0 else 0
        )) + 1

        summary_a, keys_a = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], self.spatial_shape[0], num_batch=batch_size)
        summary_b, keys_b = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], self.spatial_shape[0], num_batch=batch_size)

        plane_concat_a = torch.cat([summary_a, summary_b], dim=-1)
        plane_concat_b = torch.cat([summary_b, summary_a], dim=-1)

        gate_a = self.gate_a(plane_concat_a)
        gate_b = self.gate_b(plane_concat_b)

        mod_F_a = x_a.F + x_a.F * gate_a[keys_a]
        mod_F_b = x_b.F + x_b.F * gate_b[keys_b]

        x_a = SparseTensor(feats=mod_F_a, coords=x_a.C)
        x_b = SparseTensor(feats=mod_F_b, coords=x_b.C)

        x_a = self.block2_a(x_a)
        x_b = self.block2_b(x_b)

        pooled_a = self.pool(x_a)
        pooled_b = self.pool(x_b)

        fused = torch.cat([
            pooled_a,
            pooled_b,
            torch.abs(pooled_a - pooled_b),
            pooled_a * pooled_b,
        ], dim=-1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualViewCrossAttentionSparseCNN(nn.Module):
    """
    Dual-View Sparse CNN with Intermediate 1D Plane-Wise Cross-Attention.

    At intermediate conv stages, 1D plane summary sequences are extracted along plane axis Z.
    Multi-Head Attention (nn.MultiheadAttention) allows View A to attend to View B's
    features across current and neighboring planes before continuing to stage 2.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32],
        fc_dims: List[int] = [16],
        num_classes: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.spatial_shape = spatial_shape

        c1 = conv_channels[0] if len(conv_channels) > 0 else 16
        c2 = conv_channels[1] if len(conv_channels) > 1 else c1

        self.block1_a = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )
        self.block1_b = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )

        self.cross_attn_a = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.cross_attn_b = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.norm_a = nn.LayerNorm(c1)
        self.norm_b = nn.LayerNorm(c1)

        self.block2_a = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )
        self.block2_b = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )

        self.pool = spnn.GlobalAvgPooling()
        pooled_dim = c2
        fusion_dim = pooled_dim * 4

        classifier_layers = []
        current_dim = fusion_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_sparse_tensors(self, data) -> Tuple[SparseTensor, SparseTensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError("Unsupported data format for DualViewCrossAttentionSparseCNN.")

    def forward(self, data) -> torch.Tensor:
        tensor_a, tensor_b = self._extract_sparse_tensors(data)

        x_a = self.block1_a(tensor_a)
        x_b = self.block1_b(tensor_b)

        batch_size = int(max(
            x_a.C[:, 0].max().item() if x_a.C.numel() > 0 else 0,
            x_b.C[:, 0].max().item() if x_b.C.numel() > 0 else 0
        )) + 1

        summary_a, keys_a = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], self.spatial_shape[0], num_batch=batch_size)
        summary_b, keys_b = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], self.spatial_shape[0], num_batch=batch_size)

        max_planes = self.spatial_shape[0]

        seq_a = summary_a.view(batch_size, max_planes, -1)
        seq_b = summary_b.view(batch_size, max_planes, -1)

        attn_out_a, _ = self.cross_attn_a(query=seq_a, key=seq_b, value=seq_b)
        attn_out_b, _ = self.cross_attn_b(query=seq_b, key=seq_a, value=seq_a)

        seq_a = self.norm_a(seq_a + attn_out_a).view(-1, seq_a.size(-1))
        seq_b = self.norm_b(seq_b + attn_out_b).view(-1, seq_b.size(-1))

        mod_F_a = x_a.F + seq_a[keys_a]
        mod_F_b = x_b.F + seq_b[keys_b]

        x_a = SparseTensor(feats=mod_F_a, coords=x_a.C)
        x_b = SparseTensor(feats=mod_F_b, coords=x_b.C)

        x_a = self.block2_a(x_a)
        x_b = self.block2_b(x_b)

        pooled_a = self.pool(x_a)
        pooled_b = self.pool(x_b)

        fused = torch.cat([
            pooled_a,
            pooled_b,
            torch.abs(pooled_a - pooled_b),
            pooled_a * pooled_b,
        ], dim=-1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SparseResBlock(nn.Module):
    """SubMConv2d Residual Block that preserves spatial active site coordinates."""
    def __init__(self, channels: int, spatial_shape: Tuple[int, int]):
        super().__init__()
        self.conv1 = spnn.SubMConv2d(channels, channels, kernel_size=3, spatial_shape=spatial_shape)
        self.bn1 = spnn.BatchNorm(channels)
        self.relu = spnn.ReLU()
        self.conv2 = spnn.SubMConv2d(channels, channels, kernel_size=3, spatial_shape=spatial_shape)
        self.bn2 = spnn.BatchNorm(channels)
        self.out_relu = spnn.ReLU()

    def forward(self, x: SparseTensor) -> SparseTensor:
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.bn2(res)
        mod_F = x.F + res.F
        out = SparseTensor(feats=mod_F, coords=x.C)
        return self.out_relu(out)


class DualViewResNetCrossAttentionSparseCNN(nn.Module):
    """
    Dual-View Sparse ResNet with Intermediate 1D Plane Cross-Attention.

    Uses Sparse Residual Blocks (SubMConv2d skip connections) in Stage 1 and Stage 2 feature backbones,
    enabling deeper, gradient-stable feature extraction paired with 1D cross-attention along the Z-axis.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [32, 64],
        fc_dims: List[int] = [16],
        num_classes: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.spatial_shape = spatial_shape

        c1 = conv_channels[0] if len(conv_channels) > 0 else 32
        c2 = conv_channels[1] if len(conv_channels) > 1 else c1

        self.block1_a = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
            SparseResBlock(c1, spatial_shape=spatial_shape)
        )
        self.block1_b = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
            SparseResBlock(c1, spatial_shape=spatial_shape)
        )

        self.cross_attn_a = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.cross_attn_b = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.norm_a = nn.LayerNorm(c1)
        self.norm_b = nn.LayerNorm(c1)

        self.block2_a = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
            SparseResBlock(c2, spatial_shape=spatial_shape)
        )
        self.block2_b = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
            SparseResBlock(c2, spatial_shape=spatial_shape)
        )

        self.pool = spnn.GlobalAvgPooling()
        pooled_dim = c2
        fusion_dim = pooled_dim * 4

        classifier_layers = []
        current_dim = fusion_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_sparse_tensors(self, data) -> Tuple[SparseTensor, SparseTensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError("Unsupported data format for DualViewResNetCrossAttentionSparseCNN.")

    def forward(self, data) -> torch.Tensor:
        tensor_a, tensor_b = self._extract_sparse_tensors(data)

        x_a = self.block1_a(tensor_a)
        x_b = self.block1_b(tensor_b)

        batch_size = int(max(
            x_a.C[:, 0].max().item() if x_a.C.numel() > 0 else 0,
            x_b.C[:, 0].max().item() if x_b.C.numel() > 0 else 0
        )) + 1

        summary_a, keys_a = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], self.spatial_shape[0], num_batch=batch_size)
        summary_b, keys_b = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], self.spatial_shape[0], num_batch=batch_size)

        max_planes = self.spatial_shape[0]

        seq_a = summary_a.view(batch_size, max_planes, -1)
        seq_b = summary_b.view(batch_size, max_planes, -1)

        attn_out_a, _ = self.cross_attn_a(query=seq_a, key=seq_b, value=seq_b)
        attn_out_b, _ = self.cross_attn_b(query=seq_b, key=seq_a, value=seq_a)

        seq_a = self.norm_a(seq_a + attn_out_a).view(-1, seq_a.size(-1))
        seq_b = self.norm_b(seq_b + attn_out_b).view(-1, seq_b.size(-1))

        mod_F_a = x_a.F + seq_a[keys_a]
        mod_F_b = x_b.F + seq_b[keys_b]

        x_a = SparseTensor(feats=mod_F_a, coords=x_a.C)
        x_b = SparseTensor(feats=mod_F_b, coords=x_b.C)

        x_a = self.block2_a(x_a)
        x_b = self.block2_b(x_b)

        pooled_a = self.pool(x_a)
        pooled_b = self.pool(x_b)

        fused = torch.cat([
            pooled_a,
            pooled_b,
            torch.abs(pooled_a - pooled_b),
            pooled_a * pooled_b,
        ], dim=-1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



class DualViewDeepCrossAttentionSparseCNN(nn.Module):
    """
    3-Stage Deep Dual-View Sparse CNN with Intermediate 1D Cross-Attention.

    Features a 3-stage SubMConv2d feature backbone (e.g. 32 -> 64 -> 128) with intermediate
    Z-plane cross-attention after Stage 1, followed by multi-resolution Conv Stage 2 and Stage 3.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [32, 64, 128],
        fc_dims: List[int] = [32, 16],
        num_classes: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.spatial_shape = spatial_shape

        c1 = conv_channels[0] if len(conv_channels) > 0 else 32
        c2 = conv_channels[1] if len(conv_channels) > 1 else c1
        c3 = conv_channels[2] if len(conv_channels) > 2 else c2

        # Stage 1
        self.block1_a = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )
        self.block1_b = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )

        # Cross-Attention Stage 1
        self.cross_attn_a = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.cross_attn_b = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.norm_a = nn.LayerNorm(c1)
        self.norm_b = nn.LayerNorm(c1)

        # Stage 2
        self.block2_a = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )
        self.block2_b = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )

        # Stage 3
        self.block3_a = nn.Sequential(
            spnn.SubMConv2d(c2, c3, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c3),
            spnn.ReLU(),
        )
        self.block3_b = nn.Sequential(
            spnn.SubMConv2d(c2, c3, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c3),
            spnn.ReLU(),
        )

        self.pool = spnn.GlobalAvgPooling()
        pooled_dim = c3
        fusion_dim = pooled_dim * 4

        classifier_layers = []
        current_dim = fusion_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_sparse_tensors(self, data) -> Tuple[SparseTensor, SparseTensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError("Unsupported data format for DualViewDeepCrossAttentionSparseCNN.")

    def forward(self, data) -> torch.Tensor:
        tensor_a, tensor_b = self._extract_sparse_tensors(data)

        # Stage 1 Conv
        x_a = self.block1_a(tensor_a)
        x_b = self.block1_b(tensor_b)

        batch_size = int(max(
            x_a.C[:, 0].max().item() if x_a.C.numel() > 0 else 0,
            x_b.C[:, 0].max().item() if x_b.C.numel() > 0 else 0
        )) + 1

        max_planes = self.spatial_shape[0]

        # Stage 1 Cross Attention
        summary_a, keys_a = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], max_planes, num_batch=batch_size)
        summary_b, keys_b = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], max_planes, num_batch=batch_size)

        seq_a = summary_a.view(batch_size, max_planes, -1)
        seq_b = summary_b.view(batch_size, max_planes, -1)

        attn_out_a, _ = self.cross_attn_a(query=seq_a, key=seq_b, value=seq_b)
        attn_out_b, _ = self.cross_attn_b(query=seq_b, key=seq_a, value=seq_a)

        seq_a = self.norm_a(seq_a + attn_out_a).view(-1, seq_a.size(-1))
        seq_b = self.norm_b(seq_b + attn_out_b).view(-1, seq_b.size(-1))

        mod_F_a = x_a.F + seq_a[keys_a]
        mod_F_b = x_b.F + seq_b[keys_b]

        x_a = SparseTensor(feats=mod_F_a, coords=x_a.C)
        x_b = SparseTensor(feats=mod_F_b, coords=x_b.C)

        # Stage 2 & 3 Conv
        x_a = self.block2_a(x_a)
        x_b = self.block2_b(x_b)

        x_a = self.block3_a(x_a)
        x_b = self.block3_b(x_b)

        pooled_a = self.pool(x_a)
        pooled_b = self.pool(x_b)

        fused = torch.cat([
            pooled_a,
            pooled_b,
            torch.abs(pooled_a - pooled_b),
            pooled_a * pooled_b,
        ], dim=-1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



class SimplifiedDualViewCrossAttentionSparseCNN(nn.Module):
    """
    Simplified Dual-View Sparse CNN with Intermediate 1D Plane-Wise Cross-Attention.

    Like DualViewCrossAttentionSparseCNN, but the SubMConv2d feature extraction backbones
    (`self.block1` and `self.block2`) are shared across both input detector views (U-view and V-view).
    Views are processed separately through the shared conv backbones, followed by 1D cross-attention
    along the Z-axis, pooling, and a shared fully connected classifier.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32],
        fc_dims: List[int] = [16],
        num_classes: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.spatial_shape = spatial_shape

        c1 = conv_channels[0] if len(conv_channels) > 0 else 16
        c2 = conv_channels[1] if len(conv_channels) > 1 else c1

        # Shared Stage 1 Conv backbone
        self.block1 = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )

        # Cross-Attention modules
        self.cross_attn_a = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.cross_attn_b = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.norm_a = nn.LayerNorm(c1)
        self.norm_b = nn.LayerNorm(c1)

        # Shared Stage 2 Conv backbone
        self.block2 = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )

        self.pool = spnn.GlobalAvgPooling()
        pooled_dim = c2
        fusion_dim = pooled_dim * 4

        classifier_layers = []
        current_dim = fusion_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_sparse_tensors(self, data) -> Tuple[SparseTensor, SparseTensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError("Unsupported data format for SimplifiedDualViewCrossAttentionSparseCNN.")

    def forward(self, data) -> torch.Tensor:
        tensor_a, tensor_b = self._extract_sparse_tensors(data)

        # Stage 1 using shared block1
        x_a = self.block1(tensor_a)
        x_b = self.block1(tensor_b)

        batch_size = int(max(
            x_a.C[:, 0].max().item() if x_a.C.numel() > 0 else 0,
            x_b.C[:, 0].max().item() if x_b.C.numel() > 0 else 0
        )) + 1

        summary_a, keys_a = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], self.spatial_shape[0], num_batch=batch_size)
        summary_b, keys_b = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], self.spatial_shape[0], num_batch=batch_size)

        max_planes = self.spatial_shape[0]

        seq_a = summary_a.view(batch_size, max_planes, -1)
        seq_b = summary_b.view(batch_size, max_planes, -1)

        attn_out_a, _ = self.cross_attn_a(query=seq_a, key=seq_b, value=seq_b)
        attn_out_b, _ = self.cross_attn_b(query=seq_b, key=seq_a, value=seq_a)

        seq_a = self.norm_a(seq_a + attn_out_a).view(-1, seq_a.size(-1))
        seq_b = self.norm_b(seq_b + attn_out_b).view(-1, seq_b.size(-1))

        mod_F_a = x_a.F + seq_a[keys_a]
        mod_F_b = x_b.F + seq_b[keys_b]

        x_a = SparseTensor(feats=mod_F_a, coords=x_a.C)
        x_b = SparseTensor(feats=mod_F_b, coords=x_b.C)

        # Stage 2 using shared block2
        x_a = self.block2(x_a)
        x_b = self.block2(x_b)

        pooled_a = self.pool(x_a)
        pooled_b = self.pool(x_b)

        fused = torch.cat([
            pooled_a,
            pooled_b,
            torch.abs(pooled_a - pooled_b),
            pooled_a * pooled_b,
        ], dim=-1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualViewPositionalCrossAttentionSparseCNN(nn.Module):
    """
    Dual-View Sparse CNN with 1D Learned Positional Encodings, Self-Attention, and Cross-Attention.

    At intermediate conv stages, 1D plane summary sequences are extracted along plane axis Z.
    Learned 1D positional embeddings (for plane depth z) are added to the sequence embeddings.
    1D Self-Attention models longitudinal track continuity within each view, followed by
    Multi-Head Cross-Attention between View A and View B before continuing to stage 2.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32],
        fc_dims: List[int] = [16],
        num_classes: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.spatial_shape = spatial_shape

        c1 = conv_channels[0] if len(conv_channels) > 0 else 16
        c2 = conv_channels[1] if len(conv_channels) > 1 else c1

        self.block1_a = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )
        self.block1_b = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )

        # 1D Positional Embedding for plane index z in [0, spatial_shape[0]-1]
        self.pos_embed = nn.Parameter(torch.zeros(1, spatial_shape[0], c1))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Self-Attention modules
        self.self_attn_a = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.self_attn_b = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.norm_self_a = nn.LayerNorm(c1)
        self.norm_self_b = nn.LayerNorm(c1)

        # Cross-Attention modules
        self.cross_attn_a = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.cross_attn_b = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.norm_cross_a = nn.LayerNorm(c1)
        self.norm_cross_b = nn.LayerNorm(c1)

        self.block2_a = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )
        self.block2_b = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )

        self.pool = spnn.GlobalAvgPooling()
        pooled_dim = c2
        fusion_dim = pooled_dim * 4

        classifier_layers = []
        current_dim = fusion_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_sparse_tensors(self, data) -> Tuple[SparseTensor, SparseTensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError("Unsupported data format for DualViewPositionalCrossAttentionSparseCNN.")

    def forward(self, data) -> torch.Tensor:
        tensor_a, tensor_b = self._extract_sparse_tensors(data)

        x_a = self.block1_a(tensor_a)
        x_b = self.block1_b(tensor_b)

        batch_size = int(max(
            x_a.C[:, 0].max().item() if x_a.C.numel() > 0 else 0,
            x_b.C[:, 0].max().item() if x_b.C.numel() > 0 else 0
        )) + 1

        summary_a, keys_a = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], self.spatial_shape[0], num_batch=batch_size)
        summary_b, keys_b = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], self.spatial_shape[0], num_batch=batch_size)

        max_planes = self.spatial_shape[0]

        seq_a = summary_a.view(batch_size, max_planes, -1) + self.pos_embed
        seq_b = summary_b.view(batch_size, max_planes, -1) + self.pos_embed

        # 1D Self-Attention within each view
        self_out_a, _ = self.self_attn_a(query=seq_a, key=seq_a, value=seq_a)
        self_out_b, _ = self.self_attn_b(query=seq_b, key=seq_b, value=seq_b)
        seq_a = self.norm_self_a(seq_a + self_out_a)
        seq_b = self.norm_self_b(seq_b + self_out_b)

        # 1D Cross-Attention between views
        cross_out_a, _ = self.cross_attn_a(query=seq_a, key=seq_b, value=seq_b)
        cross_out_b, _ = self.cross_attn_b(query=seq_b, key=seq_a, value=seq_a)

        seq_a = self.norm_cross_a(seq_a + cross_out_a).view(-1, seq_a.size(-1))
        seq_b = self.norm_cross_b(seq_b + cross_out_b).view(-1, seq_b.size(-1))

        mod_F_a = x_a.F + seq_a[keys_a]
        mod_F_b = x_b.F + seq_b[keys_b]

        x_a = SparseTensor(feats=mod_F_a, coords=x_a.C)
        x_b = SparseTensor(feats=mod_F_b, coords=x_b.C)

        x_a = self.block2_a(x_a)
        x_b = self.block2_b(x_b)

        pooled_a = self.pool(x_a)
        pooled_b = self.pool(x_b)

        fused = torch.cat([
            pooled_a,
            pooled_b,
            torch.abs(pooled_a - pooled_b),
            pooled_a * pooled_b,
        ], dim=-1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualViewMultiStageCrossAttentionSparseCNN(nn.Module):
    """
    Dual-View Sparse CNN with Multi-Stage 1D Plane-Wise Cross-Attention.

    Applies cross-attention at multiple intermediate convolutional stages (after Block 1 and after Block 2),
    enabling deep multi-scale cross-view feature interaction across plane levels before final pooling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32],
        fc_dims: List[int] = [16],
        num_classes: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.spatial_shape = spatial_shape

        c1 = conv_channels[0] if len(conv_channels) > 0 else 16
        c2 = conv_channels[1] if len(conv_channels) > 1 else c1

        # Stage 1
        self.block1_a = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )
        self.block1_b = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )

        # Cross-Attention Stage 1
        self.cross_attn1_a = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.cross_attn1_b = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.norm1_a = nn.LayerNorm(c1)
        self.norm1_b = nn.LayerNorm(c1)

        # Stage 2
        self.block2_a = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )
        self.block2_b = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )

        # Cross-Attention Stage 2
        self.cross_attn2_a = nn.MultiheadAttention(embed_dim=c2, num_heads=num_heads, batch_first=True)
        self.cross_attn2_b = nn.MultiheadAttention(embed_dim=c2, num_heads=num_heads, batch_first=True)
        self.norm2_a = nn.LayerNorm(c2)
        self.norm2_b = nn.LayerNorm(c2)

        self.pool = spnn.GlobalAvgPooling()
        pooled_dim = c2
        fusion_dim = pooled_dim * 4

        classifier_layers = []
        current_dim = fusion_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_sparse_tensors(self, data) -> Tuple[SparseTensor, SparseTensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError("Unsupported data format for DualViewMultiStageCrossAttentionSparseCNN.")

    def forward(self, data) -> torch.Tensor:
        tensor_a, tensor_b = self._extract_sparse_tensors(data)

        # --- Stage 1 Conv ---
        x_a = self.block1_a(tensor_a)
        x_b = self.block1_b(tensor_b)

        batch_size = int(max(
            x_a.C[:, 0].max().item() if x_a.C.numel() > 0 else 0,
            x_b.C[:, 0].max().item() if x_b.C.numel() > 0 else 0
        )) + 1

        max_planes = self.spatial_shape[0]

        # --- Stage 1 Cross-Attention ---
        summary1_a, keys1_a = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], max_planes, num_batch=batch_size)
        summary1_b, keys1_b = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], max_planes, num_batch=batch_size)

        seq1_a = summary1_a.view(batch_size, max_planes, -1)
        seq1_b = summary1_b.view(batch_size, max_planes, -1)

        attn1_out_a, _ = self.cross_attn1_a(query=seq1_a, key=seq1_b, value=seq1_b)
        attn1_out_b, _ = self.cross_attn1_b(query=seq1_b, key=seq1_a, value=seq1_a)

        seq1_a = self.norm1_a(seq1_a + attn1_out_a).view(-1, seq1_a.size(-1))
        seq1_b = self.norm1_b(seq1_b + attn1_out_b).view(-1, seq1_b.size(-1))

        mod_F1_a = x_a.F + seq1_a[keys1_a]
        mod_F1_b = x_b.F + seq1_b[keys1_b]

        x_a = SparseTensor(feats=mod_F1_a, coords=x_a.C)
        x_b = SparseTensor(feats=mod_F1_b, coords=x_b.C)

        # --- Stage 2 Conv ---
        x_a = self.block2_a(x_a)
        x_b = self.block2_b(x_b)

        # --- Stage 2 Cross-Attention ---
        summary2_a, keys2_a = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], max_planes, num_batch=batch_size)
        summary2_b, keys2_b = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], max_planes, num_batch=batch_size)

        seq2_a = summary2_a.view(batch_size, max_planes, -1)
        seq2_b = summary2_b.view(batch_size, max_planes, -1)

        attn2_out_a, _ = self.cross_attn2_a(query=seq2_a, key=seq2_b, value=seq2_b)
        attn2_out_b, _ = self.cross_attn2_b(query=seq2_b, key=seq2_a, value=seq2_a)

        seq2_a = self.norm2_a(seq2_a + attn2_out_a).view(-1, seq2_a.size(-1))
        seq2_b = self.norm2_b(seq2_b + attn2_out_b).view(-1, seq2_b.size(-1))

        mod_F2_a = x_a.F + seq2_a[keys2_a]
        mod_F2_b = x_b.F + seq2_b[keys2_b]

        x_a = SparseTensor(feats=mod_F2_a, coords=x_a.C)
        x_b = SparseTensor(feats=mod_F2_b, coords=x_b.C)

        # --- Pooling & Fusion ---
        pooled_a = self.pool(x_a)
        pooled_b = self.pool(x_b)

        fused = torch.cat([
            pooled_a,
            pooled_b,
            torch.abs(pooled_a - pooled_b),
            pooled_a * pooled_b,
        ], dim=-1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualView3DIntersectionSparseCNN(nn.Module):
    """
    Dual-View Sparse CNN with Physics-Aware 3D Candidate Space Point Scoring.

    At intermediate conv stages, active plane overlaps between View A and View B
    are detected to form candidate 3D space-point interactions. Learned 3D plane
    intersection weights gate and focus 2D sparse convolution features.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32],
        fc_dims: List[int] = [16],
        num_classes: int = 2,
        dropout: float = 0.1,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.spatial_shape = spatial_shape

        c1 = conv_channels[0] if len(conv_channels) > 0 else 16
        c2 = conv_channels[1] if len(conv_channels) > 1 else c1

        self.block1_a = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )
        self.block1_b = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )

        self.inter_scorer = nn.Sequential(
            nn.Linear(2 * c1, c1),
            nn.SiLU(),
            nn.Linear(c1, c1),
            nn.Sigmoid(),
        )

        self.block2_a = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )
        self.block2_b = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )

        self.pool = spnn.GlobalAvgPooling()
        pooled_dim = c2
        fusion_dim = pooled_dim * 4

        classifier_layers = []
        current_dim = fusion_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_sparse_tensors(self, data) -> Tuple[SparseTensor, SparseTensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError("Unsupported data format for DualView3DIntersectionSparseCNN.")

    def forward(self, data) -> torch.Tensor:
        tensor_a, tensor_b = self._extract_sparse_tensors(data)

        x_a = self.block1_a(tensor_a)
        x_b = self.block1_b(tensor_b)

        batch_size = int(max(
            x_a.C[:, 0].max().item() if x_a.C.numel() > 0 else 0,
            x_b.C[:, 0].max().item() if x_b.C.numel() > 0 else 0
        )) + 1

        summary_a, keys_a = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], self.spatial_shape[0], num_batch=batch_size)
        summary_b, keys_b = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], self.spatial_shape[0], num_batch=batch_size)

        inter_weights = self.inter_scorer(torch.cat([summary_a, summary_b], dim=-1))

        mod_F_a = x_a.F + x_a.F * inter_weights[keys_a]
        mod_F_b = x_b.F + x_b.F * inter_weights[keys_b]

        x_a = SparseTensor(feats=mod_F_a, coords=x_a.C)
        x_b = SparseTensor(feats=mod_F_b, coords=x_b.C)

        x_a = self.block2_a(x_a)
        x_b = self.block2_b(x_b)

        pooled_a = self.pool(x_a)
        pooled_b = self.pool(x_b)

        fused = torch.cat([
            pooled_a,
            pooled_b,
            torch.abs(pooled_a - pooled_b),
            pooled_a * pooled_b,
        ], dim=-1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SparseCrossGate(nn.Module):
    """
    Cross-view feature gating.

    Each view learns a channel-wise modulation from the other view.
    The interaction happens before global pooling, while preserving spatial
    information in the sparse tensor.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()

        hidden = max(channels // reduction, 4)

        self.gate_a = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

        self.gate_b = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(
        self,
        x_a: SparseTensor,
        x_b: SparseTensor
    ):

        # Global context vectors
        mean_a = torch.mean(x_a.F, dim=0)
        mean_b = torch.mean(x_b.F, dim=0)

        gate_a = self.gate_a(mean_b)
        gate_b = self.gate_b(mean_a)

        x_a = SparseTensor(
            feats=x_a.F * gate_a,
            coords=x_a.C
        )

        x_b = SparseTensor(
            feats=x_b.F * gate_b,
            coords=x_b.C
        )

        return x_a, x_b



class DualViewCrossGateSparseCNN(nn.Module):
    r"""
    Dual-view sparse CNN for MINOS U/V event classification.

    Architecture:

        U view                 V view

        Sparse CNN             Sparse CNN
             |                      |
             |                      |
             +---- Cross Gate -----+
             |                      |
        Sparse CNN             Sparse CNN
             |                      |
          Pool                  Pool
             \                  /
              Feature fusion
                    |
              classifier

    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32],
        fc_dims: List[int] = [32],
        num_classes: int = 2,
        dropout: float = 0.1,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()

        self.in_channels = in_channels
        c1 = conv_channels[0]
        c2 = conv_channels[1]

        self.spatial_shape = spatial_shape


        # -------------------------
        # First sparse feature stage
        # -------------------------

        self.block1_a = nn.Sequential(
            spnn.SubMConv2d(
                in_channels,
                c1,
                kernel_size=3,
                spatial_shape=spatial_shape
            ),
            spnn.BatchNorm(c1),
            spnn.ReLU()
        )

        self.block1_b = nn.Sequential(
            spnn.SubMConv2d(
                in_channels,
                c1,
                kernel_size=3,
                spatial_shape=spatial_shape
            ),
            spnn.BatchNorm(c1),
            spnn.ReLU()
        )


        # Cross-view interaction
        self.cross_gate = SparseCrossGate(c1)


        # -------------------------
        # Second feature stage
        # -------------------------

        self.block2_a = nn.Sequential(
            spnn.SubMConv2d(
                c1,
                c2,
                kernel_size=3,
                spatial_shape=spatial_shape
            ),
            spnn.BatchNorm(c2),
            spnn.ReLU()
        )


        self.block2_b = nn.Sequential(
            spnn.SubMConv2d(
                c1,
                c2,
                kernel_size=3,
                spatial_shape=spatial_shape
            ),
            spnn.BatchNorm(c2),
            spnn.ReLU()
        )


        self.pool = spnn.GlobalAvgPooling()


        # Final fusion:
        #
        # A
        # B
        # |A-B|
        # A*B
        #
        fusion_dim = 4 * c2


        layers = []

        current = fusion_dim

        for dim in fc_dims:

            layers.append(
                nn.Linear(current, dim)
            )

            layers.append(
                nn.ReLU()
            )

            if dropout > 0:
                layers.append(
                    nn.Dropout(dropout)
                )

            current = dim


        layers.append(
            nn.Linear(current, num_classes)
        )

        self.classifier = nn.Sequential(*layers)



    def _extract_sparse_tensors(self, data):

        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]

        if isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError(
            "Unsupported input format"
        )


    def forward(self, data):

        x_a, x_b = self._extract_sparse_tensors(data)


        # First view-specific processing

        x_a = self.block1_a(x_a)
        x_b = self.block1_b(x_b)


        # Exchange information

        x_a, x_b = self.cross_gate(
            x_a,
            x_b
        )


        # More view-specific processing

        x_a = self.block2_a(x_a)
        x_b = self.block2_b(x_b)


        # Global representation

        pooled_a = self.pool(x_a)
        pooled_b = self.pool(x_b)


        # Explicit interaction features

        fused = torch.cat(
            [
                pooled_a,
                pooled_b,
                torch.abs(pooled_a - pooled_b),
                pooled_a * pooled_b
            ],
            dim=1
        )


        return self.classifier(fused)



    def get_num_params(self):

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )


class DualViewHybridTransformerSparseCNN(nn.Module):
    r"""
    Dual-View Hybrid Conv-Transformer for MINOS U/V event classification.

    Combines 2D Sparse Convolutions for local spatial feature extraction
    with interleaved Multi-Head Cross-Attention Transformer blocks across
    the shared plane Z axis at multiple intermediate stages.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32],
        fc_dims: List[int] = [16],
        num_classes: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.spatial_shape = spatial_shape
        max_planes = spatial_shape[0]

        c1 = conv_channels[0] if len(conv_channels) > 0 else 16
        c2 = conv_channels[1] if len(conv_channels) > 1 else c1

        # Stage 1 Sparse Convolutions
        self.block1_a = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )
        self.block1_b = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )

        # Stage 1 Interleaved Transformer Cross-Attention & Positional Embedding
        self.pos_embed1 = nn.Parameter(torch.randn(1, max_planes, c1) * 0.02)
        self.cross_attn1_a = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.cross_attn1_b = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.norm1_a = nn.LayerNorm(c1)
        self.norm1_b = nn.LayerNorm(c1)

        # Stage 2 Sparse Convolutions
        self.block2_a = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )
        self.block2_b = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )

        # Stage 2 Interleaved Transformer Cross-Attention & Positional Embedding
        self.pos_embed2 = nn.Parameter(torch.randn(1, max_planes, c2) * 0.02)
        self.cross_attn2_a = nn.MultiheadAttention(embed_dim=c2, num_heads=num_heads, batch_first=True)
        self.cross_attn2_b = nn.MultiheadAttention(embed_dim=c2, num_heads=num_heads, batch_first=True)
        self.norm2_a = nn.LayerNorm(c2)
        self.norm2_b = nn.LayerNorm(c2)

        self.pool = spnn.GlobalAvgPooling()
        fusion_dim = c2 * 4

        classifier_layers = []
        current_dim = fusion_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_sparse_tensors(self, data) -> Tuple[SparseTensor, SparseTensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError("Unsupported data format for DualViewHybridTransformerSparseCNN.")

    def forward(self, data) -> torch.Tensor:
        tensor_a, tensor_b = self._extract_sparse_tensors(data)

        # Stage 1 Sparse Convolutions
        x_a = self.block1_a(tensor_a)
        x_b = self.block1_b(tensor_b)

        batch_size = int(max(
            x_a.C[:, 0].max().item() if x_a.C.numel() > 0 else 0,
            x_b.C[:, 0].max().item() if x_b.C.numel() > 0 else 0
        )) + 1
        max_planes = self.spatial_shape[0]

        # Stage 1 Cross-Attention along Z
        sum1_a, keys1_a = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], max_planes, num_batch=batch_size)
        sum1_b, keys1_b = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], max_planes, num_batch=batch_size)

        seq1_a = sum1_a.view(batch_size, max_planes, -1) + self.pos_embed1
        seq1_b = sum1_b.view(batch_size, max_planes, -1) + self.pos_embed1

        attn1_a, _ = self.cross_attn1_a(query=seq1_a, key=seq1_b, value=seq1_b)
        attn1_b, _ = self.cross_attn1_b(query=seq1_b, key=seq1_a, value=seq1_a)

        seq1_a = self.norm1_a(seq1_a + attn1_a).view(-1, seq1_a.size(-1))
        seq1_b = self.norm1_b(seq1_b + attn1_b).view(-1, seq1_b.size(-1))

        x_a = SparseTensor(feats=x_a.F + seq1_a[keys1_a], coords=x_a.C)
        x_b = SparseTensor(feats=x_b.F + seq1_b[keys1_b], coords=x_b.C)

        # Stage 2 Sparse Convolutions
        x_a = self.block2_a(x_a)
        x_b = self.block2_b(x_b)

        # Stage 2 Cross-Attention along Z
        sum2_a, keys2_a = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], max_planes, num_batch=batch_size)
        sum2_b, keys2_b = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], max_planes, num_batch=batch_size)

        seq2_a = sum2_a.view(batch_size, max_planes, -1) + self.pos_embed2
        seq2_b = sum2_b.view(batch_size, max_planes, -1) + self.pos_embed2

        attn2_a, _ = self.cross_attn2_a(query=seq2_a, key=seq2_b, value=seq2_b)
        attn2_b, _ = self.cross_attn2_b(query=seq2_b, key=seq2_a, value=seq2_a)

        seq2_a = self.norm2_a(seq2_a + attn2_a).view(-1, seq2_a.size(-1))
        seq2_b = self.norm2_b(seq2_b + attn2_b).view(-1, seq2_b.size(-1))

        x_a = SparseTensor(feats=x_a.F + seq2_a[keys2_a], coords=x_a.C)
        x_b = SparseTensor(feats=x_b.F + seq2_b[keys2_b], coords=x_b.C)

        pooled_a = self.pool(x_a)
        pooled_b = self.pool(x_b)

        fused = torch.cat([
            pooled_a,
            pooled_b,
            torch.abs(pooled_a - pooled_b),
            pooled_a * pooled_b,
        ], dim=-1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualViewTransformerLayer(nn.Module):
    """
    Dual-view Transformer layer with Self-Attention and Cross-Attention
    over 1D plane sequences along plane axis Z.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Self-Attention
        self.self_attn_a = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.self_attn_b = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_sa_a = nn.LayerNorm(embed_dim)
        self.norm_sa_b = nn.LayerNorm(embed_dim)

        # Cross-Attention
        self.cross_attn_a = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_b = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_ca_a = nn.LayerNorm(embed_dim)
        self.norm_ca_b = nn.LayerNorm(embed_dim)

        # Feed-Forward Network (FFN)
        self.ffn_a = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn_a = nn.LayerNorm(embed_dim)
        self.norm_ffn_b = nn.LayerNorm(embed_dim)

    def forward(self, seq_a: torch.Tensor, seq_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Self-Attention
        sa_a, _ = self.self_attn_a(query=seq_a, key=seq_a, value=seq_a)
        sa_b, _ = self.self_attn_b(query=seq_b, key=seq_b, value=seq_b)
        seq_a = self.norm_sa_a(seq_a + sa_a)
        seq_b = self.norm_sa_b(seq_b + sa_b)

        # 2. Cross-Attention
        ca_a, _ = self.cross_attn_a(query=seq_a, key=seq_b, value=seq_b)
        ca_b, _ = self.cross_attn_b(query=seq_b, key=seq_a, value=seq_a)
        seq_a = self.norm_ca_a(seq_a + ca_a)
        seq_b = self.norm_ca_b(seq_b + ca_b)

        # 3. Feed-Forward
        seq_a = self.norm_ffn_a(seq_a + self.ffn_a(seq_a))
        seq_b = self.norm_ffn_b(seq_b + self.ffn_b(seq_b))

        return seq_a, seq_b


class DualViewMultiLayerTransformerSparseCNN(nn.Module):
    r"""
    Dual-View Sparse CNN with a Multi-Layer Transformer Encoder Stack along Z.

    Extracts 1D plane summary sequences along the shared plane axis Z, adds learned
    1D positional encodings, processes plane sequences through multiple stacked Transformer
    layers (self-attention + cross-attention + FFN), and modulates sparse hit features.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32],
        fc_dims: List[int] = [16],
        num_classes: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
        num_transformer_layers: int = 3,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.spatial_shape = spatial_shape
        max_planes = spatial_shape[0]

        c1 = conv_channels[0] if len(conv_channels) > 0 else 16
        c2 = conv_channels[1] if len(conv_channels) > 1 else c1

        self.block1_a = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )
        self.block1_b = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )

        # 1D Positional Encodings along Z
        self.pos_embed = nn.Parameter(torch.randn(1, max_planes, c1) * 0.02)

        # Multi-layer Transformer Stack
        self.transformer_layers = nn.ModuleList([
            DualViewTransformerLayer(embed_dim=c1, num_heads=num_heads, dropout=dropout)
            for _ in range(num_transformer_layers)
        ])

        self.block2_a = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )
        self.block2_b = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )

        self.pool = spnn.GlobalAvgPooling()
        fusion_dim = c2 * 4

        classifier_layers = []
        current_dim = fusion_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_sparse_tensors(self, data) -> Tuple[SparseTensor, SparseTensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError("Unsupported data format for DualViewMultiLayerTransformerSparseCNN.")

    def forward(self, data) -> torch.Tensor:
        tensor_a, tensor_b = self._extract_sparse_tensors(data)

        x_a = self.block1_a(tensor_a)
        x_b = self.block1_b(tensor_b)

        batch_size = int(max(
            x_a.C[:, 0].max().item() if x_a.C.numel() > 0 else 0,
            x_b.C[:, 0].max().item() if x_b.C.numel() > 0 else 0
        )) + 1
        max_planes = self.spatial_shape[0]

        summary_a, keys_a = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], max_planes, num_batch=batch_size)
        summary_b, keys_b = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], max_planes, num_batch=batch_size)

        seq_a = summary_a.view(batch_size, max_planes, -1) + self.pos_embed
        seq_b = summary_b.view(batch_size, max_planes, -1) + self.pos_embed

        for layer in self.transformer_layers:
            seq_a, seq_b = layer(seq_a, seq_b)

        seq_a_flat = seq_a.view(-1, seq_a.size(-1))
        seq_b_flat = seq_b.view(-1, seq_b.size(-1))

        mod_F_a = x_a.F + seq_a_flat[keys_a]
        mod_F_b = x_b.F + seq_b_flat[keys_b]

        x_a = SparseTensor(feats=mod_F_a, coords=x_a.C)
        x_b = SparseTensor(feats=mod_F_b, coords=x_b.C)

        x_a = self.block2_a(x_a)
        x_b = self.block2_b(x_b)

        pooled_a = self.pool(x_a)
        pooled_b = self.pool(x_b)

        fused = torch.cat([
            pooled_a,
            pooled_b,
            torch.abs(pooled_a - pooled_b),
            pooled_a * pooled_b,
        ], dim=-1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualViewZipperSparseCNN(nn.Module):
    r"""
    Dual-View Zipper Sparse CNN for MINOS U/V event classification.

    Processes U and V views using 2D SubMConv2d feature extraction backbones.
    Per-plane 1D summaries along Z are extracted for View A and View B, and
    zipper-interleaved into a single 1D plane sequence along Z of length 2 * max_planes:
      [A(0), B(0), A(1), B(1), A(2), B(2), ..., A(M-1), B(M-1)]

    1D Convolutions (Conv1d) slide along this physical plane sequence to pick up
    longitudinal 3D energy deposition patterns, track slopes, and shower development.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [16, 32],
        conv1d_channels: List[int] = [32],
        fc_dims: List[int] = [8, 4, 2],
        num_classes: int = 2,
        dropout: float = 0.1,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.conv1d_channels = conv1d_channels
        self.spatial_shape = spatial_shape
        max_planes = spatial_shape[0]

        c1 = conv_channels[0] if len(conv_channels) > 0 else 16
        c2 = conv_channels[1] if len(conv_channels) > 1 else c1

        # 2D Sparse CNN backbones for View A and View B
        self.block1_a = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )
        self.block1_b = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
        )

        self.block2_a = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )
        self.block2_b = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
        )

        # 1D Positional Embedding for plane Z
        self.pos_embed = nn.Parameter(torch.randn(1, max_planes, c2) * 0.02)

        # 1D Convolution Stack over Zippered Sequence
        conv1d_layers = []
        prev_c = c2
        for i, out_c in enumerate(conv1d_channels):
            conv1d_layers.append(
                nn.Conv1d(prev_c, out_c, kernel_size=3, padding=1)
            )
            conv1d_layers.append(nn.BatchNorm1d(out_c))
            conv1d_layers.append(nn.SiLU())
            if i % 2 == 1:
                conv1d_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if dropout > 0.0:
                conv1d_layers.append(nn.Dropout(dropout))
            prev_c = out_c

        self.conv1d_stack = nn.Sequential(*conv1d_layers)

        # Classifier
        classifier_layers = []
        current_dim = prev_c * 2  # Mean pooling + Max pooling across 1D sequence
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.SiLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_sparse_tensors(self, data) -> Tuple[SparseTensor, SparseTensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError("Unsupported data format for DualViewZipperSparseCNN.")

    def forward(self, data) -> torch.Tensor:
        tensor_a, tensor_b = self._extract_sparse_tensors(data)

        x_a = self.block1_a(tensor_a)
        x_b = self.block1_b(tensor_b)

        x_a = self.block2_a(x_a)
        x_b = self.block2_b(x_b)

        batch_size = int(max(
            x_a.C[:, 0].max().item() if x_a.C.numel() > 0 else 0,
            x_b.C[:, 0].max().item() if x_b.C.numel() > 0 else 0
        )) + 1
        max_planes = self.spatial_shape[0]

        # Extract per-plane 1D summaries along Z
        summary_a, _ = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], max_planes, num_batch=batch_size)
        summary_b, _ = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], max_planes, num_batch=batch_size)

        seq_a = summary_a.view(batch_size, max_planes, -1) + self.pos_embed
        seq_b = summary_b.view(batch_size, max_planes, -1) + self.pos_embed

        # Zipper Interleave along Z: [A(0), B(0), A(1), B(1), ..., A(M-1), B(M-1)]
        # Shape: (batch_size, 2 * max_planes, C)
        zipped = torch.stack([seq_a, seq_b], dim=2).view(batch_size, 2 * max_planes, -1)

        # Transpose to (batch_size, channels, length=2*max_planes) for Conv1d
        zipped_t = zipped.transpose(1, 2)

        # Pass through 1D Convolution Stack
        conv1d_out = self.conv1d_stack(zipped_t)  # (batch_size, C_out, L_out)

        # Global Pooling (Mean + Max across length)
        mean_pooled = torch.mean(conv1d_out, dim=2)
        max_pooled, _ = torch.max(conv1d_out, dim=2)
        fused = torch.cat([mean_pooled, max_pooled], dim=1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MinimumViableMINOSGNN(nn.Module):
    """
    Small hetero-GNN for MINOS binary classification.

    The model encodes two view-specific node sets plus shared nexus nodes,
    applies a few HeteroConv layers with SAGEConv message passing, pools each
    node type to a graph embedding, and classifies the event.
    """

    def __init__(
        self,
        metadata,
        in_channels: int = 4,
        hidden_channels: int = 32,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        if HeteroConv is None or SAGEConv is None or global_mean_pool is None:
            raise ImportError(
                "torch_geometric is required for MinimumViableMINOSGNN. "
                "Install PyTorch Geometric to use the GNN path."
            )

        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.dropout = dropout

        self.node_encoders = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(in_channels, hidden_channels),
                    nn.ReLU(),
                )
                for node_type in self.node_types
            }
        )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                edge_type: SAGEConv(hidden_channels, hidden_channels)
                for edge_type in self.edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        pooled_dim = hidden_channels * len(self.node_types)
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, data) -> torch.Tensor:
        x_dict = {
            node_type: self.node_encoders[node_type](data[node_type].x)
            for node_type in self.node_types
        }
        edge_index_dict = {
            edge_type: data[edge_type].edge_index
            for edge_type in self.edge_types
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                node_type: torch.relu(x)
                for node_type, x in x_dict.items()
            }
            x_dict = {
                node_type: nn.functional.dropout(x, p=self.dropout, training=self.training)
                for node_type, x in x_dict.items()
            }

        pooled = []
        for node_type in self.node_types:
            batch = getattr(data[node_type], "batch", None)
            if batch is None:
                batch = torch.zeros(x_dict[node_type].size(0), dtype=torch.long, device=x_dict[node_type].device)
            pooled.append(global_mean_pool(x_dict[node_type], batch))

        graph_embedding = torch.cat(pooled, dim=-1)
        return self.classifier(graph_embedding)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NuGraphInspiredBinaryGNN(nn.Module):
    """
    NuGraph-inspired binary classifier for the MINOS two-view problem.

    The model keeps the existing graph representation but changes the update
    pattern to better reflect NuGraph-style reasoning:
    - local per-view message passing,
    - shared nexus mixing,
    - residual gated updates,
    - explicit fusion of both view embeddings for the final binary decision.

    Note:
        Evaluated variations including Delaunay/KNN mesh edge construction ("gnn_nugraph_delaunay")
        and Gated Attention Pooling ("gnn_nugraph_attention") did not yield performance improvements
        over this base NuGraph architecture.
    """

    def __init__(
        self,
        metadata,
        in_channels: int = 4,
        hidden_channels: int = 48,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()

        if HeteroConv is None or SAGEConv is None or global_mean_pool is None:
            raise ImportError(
                "torch_geometric is required for NuGraphInspiredBinaryGNN. "
                "Install PyTorch Geometric to use the improved GNN path."
            )

        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.dropout = dropout

        self.node_encoders = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(in_channels, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                for node_type in self.node_types
            }
        )

        self.local_relations = [
            edge_type for edge_type in self.edge_types if edge_type[0] == edge_type[2] and edge_type[1] == "same_view"
        ]
        self.cross_relations = [
            edge_type for edge_type in self.edge_types if edge_type[1] in {"to_nexus", "rev_to_view_a", "rev_to_view_b"}
        ]
        if not self.local_relations:
            self.local_relations = [edge_type for edge_type in self.edge_types if edge_type[0] == edge_type[2]]
        if not self.cross_relations:
            self.cross_relations = [edge_type for edge_type in self.edge_types if edge_type[0] != edge_type[2]]

        self.local_convs = nn.ModuleList()
        self.cross_convs = nn.ModuleList()
        self.local_norms = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_channels) for node_type in self.node_types}
        )
        self.cross_norms = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_channels) for node_type in self.node_types}
        )
        self.gates = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(2 * hidden_channels, hidden_channels),
                    nn.Sigmoid(),
                )
                for node_type in self.node_types
            }
        )

        for _ in range(num_layers):
            self.local_convs.append(
                HeteroConv(
                    {
                        edge_type: SAGEConv(hidden_channels, hidden_channels)
                        for edge_type in self.local_relations
                    },
                    aggr="sum",
                )
            )
            self.cross_convs.append(
                HeteroConv(
                    {
                        edge_type: SAGEConv(hidden_channels, hidden_channels)
                        for edge_type in self.cross_relations
                    },
                    aggr="sum",
                )
            )

        fusion_dim = hidden_channels * 5
        self.readout = nn.Sequential(
            nn.Linear(fusion_dim, hidden_channels * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def _pool(self, x_dict: Dict[str, torch.Tensor], data) -> Dict[str, torch.Tensor]:
        pooled = {}
        for node_type in self.node_types:
            batch = getattr(data[node_type], "batch", None)
            if batch is None:
                batch = torch.zeros(
                    x_dict[node_type].size(0),
                    dtype=torch.long,
                    device=x_dict[node_type].device,
                )
            pooled[node_type] = global_mean_pool(x_dict[node_type], batch)
        return pooled

    def forward(self, data) -> torch.Tensor:
        x_dict = {
            node_type: self.node_encoders[node_type](data[node_type].x)
            for node_type in self.node_types
        }

        edge_index_dict = {
            edge_type: data[edge_type].edge_index
            for edge_type in self.edge_types
        }

        for local_conv, cross_conv in zip(self.local_convs, self.cross_convs):
            local_update = local_conv(x_dict, edge_index_dict)
            for node_type, update in local_update.items():
                update = self.local_norms[node_type](torch.relu(update))
                gate = self.gates[node_type](torch.cat([x_dict[node_type], update], dim=-1))
                x_dict[node_type] = x_dict[node_type] + gate * nn.functional.dropout(update, p=self.dropout, training=self.training)

            cross_update = cross_conv(x_dict, edge_index_dict)
            for node_type, update in cross_update.items():
                update = self.cross_norms[node_type](torch.relu(update))
                gate = self.gates[node_type](torch.cat([x_dict[node_type], update], dim=-1))
                x_dict[node_type] = x_dict[node_type] + gate * nn.functional.dropout(update, p=self.dropout, training=self.training)

        pooled = self._pool(x_dict, data)
        view_a = pooled["view_a"]
        view_b = pooled["view_b"]
        nexus = pooled["nexus"]
        fused = torch.cat([
            view_a,
            view_b,
            nexus,
            torch.abs(view_a - view_b),
            view_a * view_b,
        ], dim=-1)
        return self.readout(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _sparse_global_max_pooling(x: SparseTensor) -> torch.Tensor:
    batch_idx = x.C[:, 0]
    batch_size = int(batch_idx.max().item()) + 1 if x.C.numel() > 0 else 1
    num_feats = x.F.size(1)
    max_pooled = torch.full((batch_size, num_feats), fill_value=-1e9, device=x.F.device, dtype=x.F.dtype)
    max_pooled.scatter_reduce_(0, batch_idx.unsqueeze(1).expand(-1, num_feats), x.F, reduce="max")
    max_pooled = torch.where(max_pooled == -1e9, torch.zeros_like(max_pooled), max_pooled)
    return max_pooled


class DualViewDeepResNetCrossAttentionSparseCNN(nn.Module):
    """
    3-Stage Deep Dual-View Sparse ResNet with Intermediate 1D Cross-Attention.
    Combines 3-stage feature depth [32, 64, 128] with residual skip-connections (SparseResBlock)
    in all 3 stages for deep gradient stability and rich spatial representations.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [32, 64, 128],
        fc_dims: List[int] = [32, 16],
        num_classes: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.spatial_shape = spatial_shape

        c1 = conv_channels[0] if len(conv_channels) > 0 else 32
        c2 = conv_channels[1] if len(conv_channels) > 1 else c1
        c3 = conv_channels[2] if len(conv_channels) > 2 else c2

        # Stage 1 + ResBlock
        self.block1_a = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
            SparseResBlock(c1, spatial_shape=spatial_shape)
        )
        self.block1_b = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
            SparseResBlock(c1, spatial_shape=spatial_shape)
        )

        # Cross-Attention Stage 1
        self.cross_attn_a = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.cross_attn_b = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.norm_a = nn.LayerNorm(c1)
        self.norm_b = nn.LayerNorm(c1)

        # Stage 2 + ResBlock
        self.block2_a = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
            SparseResBlock(c2, spatial_shape=spatial_shape)
        )
        self.block2_b = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
            SparseResBlock(c2, spatial_shape=spatial_shape)
        )

        # Stage 3 + ResBlock
        self.block3_a = nn.Sequential(
            spnn.SubMConv2d(c2, c3, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c3),
            spnn.ReLU(),
            SparseResBlock(c3, spatial_shape=spatial_shape)
        )
        self.block3_b = nn.Sequential(
            spnn.SubMConv2d(c2, c3, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c3),
            spnn.ReLU(),
            SparseResBlock(c3, spatial_shape=spatial_shape)
        )

        self.pool = spnn.GlobalAvgPooling()
        pooled_dim = c3
        fusion_dim = pooled_dim * 4

        classifier_layers = []
        current_dim = fusion_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_sparse_tensors(self, data) -> Tuple[SparseTensor, SparseTensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError("Unsupported data format for DualViewDeepResNetCrossAttentionSparseCNN.")

    def forward(self, data) -> torch.Tensor:
        tensor_a, tensor_b = self._extract_sparse_tensors(data)

        # Stage 1 Conv + ResBlock
        x_a = self.block1_a(tensor_a)
        x_b = self.block1_b(tensor_b)

        batch_size = int(max(
            x_a.C[:, 0].max().item() if x_a.C.numel() > 0 else 0,
            x_b.C[:, 0].max().item() if x_b.C.numel() > 0 else 0
        )) + 1

        max_planes = self.spatial_shape[0]

        # Stage 1 Cross Attention
        summary_a, keys_a = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], max_planes, num_batch=batch_size)
        summary_b, keys_b = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], max_planes, num_batch=batch_size)

        seq_a = summary_a.view(batch_size, max_planes, -1)
        seq_b = summary_b.view(batch_size, max_planes, -1)

        attn_out_a, _ = self.cross_attn_a(query=seq_a, key=seq_b, value=seq_b)
        attn_out_b, _ = self.cross_attn_b(query=seq_b, key=seq_a, value=seq_a)

        seq_a = self.norm_a(seq_a + attn_out_a).view(-1, seq_a.size(-1))
        seq_b = self.norm_b(seq_b + attn_out_b).view(-1, seq_b.size(-1))

        mod_F_a = x_a.F + seq_a[keys_a]
        mod_F_b = x_b.F + seq_b[keys_b]

        x_a = SparseTensor(feats=mod_F_a, coords=x_a.C)
        x_b = SparseTensor(feats=mod_F_b, coords=x_b.C)

        # Stage 2 Conv + ResBlock
        x_a = self.block2_a(x_a)
        x_b = self.block2_b(x_b)

        # Stage 3 Conv + ResBlock
        x_a = self.block3_a(x_a)
        x_b = self.block3_b(x_b)

        pooled_a = self.pool(x_a)
        pooled_b = self.pool(x_b)

        fused = torch.cat([
            pooled_a,
            pooled_b,
            torch.abs(pooled_a - pooled_b),
            pooled_a * pooled_b,
        ], dim=-1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualViewResNetDualPoolCrossAttentionSparseCNN(nn.Module):
    """
    Dual-View Sparse ResNet with Intermediate Cross-Attention and Dual Pooling (Average + Max).
    Combines Global Average Pooling (average shower density) and Global Max Pooling (peak track energy).
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [32, 64],
        fc_dims: List[int] = [32, 16],
        num_classes: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.spatial_shape = spatial_shape

        c1 = conv_channels[0] if len(conv_channels) > 0 else 32
        c2 = conv_channels[1] if len(conv_channels) > 1 else c1

        self.block1_a = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
            SparseResBlock(c1, spatial_shape=spatial_shape)
        )
        self.block1_b = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
            SparseResBlock(c1, spatial_shape=spatial_shape)
        )

        self.cross_attn_a = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.cross_attn_b = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.norm_a = nn.LayerNorm(c1)
        self.norm_b = nn.LayerNorm(c1)

        self.block2_a = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
            SparseResBlock(c2, spatial_shape=spatial_shape)
        )
        self.block2_b = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
            SparseResBlock(c2, spatial_shape=spatial_shape)
        )

        self.avg_pool = spnn.GlobalAvgPooling()
        fusion_dim = c2 * 8  # avg_a, max_a, avg_b, max_b, abs(avg_diff), avg_prod, abs(max_diff), max_prod

        classifier_layers = []
        current_dim = fusion_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_sparse_tensors(self, data) -> Tuple[SparseTensor, SparseTensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError("Unsupported data format for DualViewResNetDualPoolCrossAttentionSparseCNN.")

    def forward(self, data) -> torch.Tensor:
        tensor_a, tensor_b = self._extract_sparse_tensors(data)

        x_a = self.block1_a(tensor_a)
        x_b = self.block1_b(tensor_b)

        batch_size = int(max(
            x_a.C[:, 0].max().item() if x_a.C.numel() > 0 else 0,
            x_b.C[:, 0].max().item() if x_b.C.numel() > 0 else 0
        )) + 1

        max_planes = self.spatial_shape[0]

        summary_a, keys_a = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], max_planes, num_batch=batch_size)
        summary_b, keys_b = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], max_planes, num_batch=batch_size)

        seq_a = summary_a.view(batch_size, max_planes, -1)
        seq_b = summary_b.view(batch_size, max_planes, -1)

        attn_out_a, _ = self.cross_attn_a(query=seq_a, key=seq_b, value=seq_b)
        attn_out_b, _ = self.cross_attn_b(query=seq_b, key=seq_a, value=seq_a)

        seq_a = self.norm_a(seq_a + attn_out_a).view(-1, seq_a.size(-1))
        seq_b = self.norm_b(seq_b + attn_out_b).view(-1, seq_b.size(-1))

        mod_F_a = x_a.F + seq_a[keys_a]
        mod_F_b = x_b.F + seq_b[keys_b]

        x_a = SparseTensor(feats=mod_F_a, coords=x_a.C)
        x_b = SparseTensor(feats=mod_F_b, coords=x_b.C)

        x_a = self.block2_a(x_a)
        x_b = self.block2_b(x_b)

        avg_a = self.avg_pool(x_a)
        avg_b = self.avg_pool(x_b)
        max_a = _sparse_global_max_pooling(x_a)
        max_b = _sparse_global_max_pooling(x_b)

        fused = torch.cat([
            avg_a,
            max_a,
            avg_b,
            max_b,
            torch.abs(avg_a - avg_b),
            avg_a * avg_b,
            torch.abs(max_a - max_b),
            max_a * max_b,
        ], dim=-1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualViewResNetMultiStageCrossAttentionSparseCNN(nn.Module):
    """
    Dual-View Sparse ResNet with Multi-Stage (Stage 1 and Stage 2) Cross-Attention.
    Exchanges information between U and V views at both fine spatial resolution (Stage 1)
    and broader region resolution (Stage 2).
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: List[int] = [32, 64],
        fc_dims: List[int] = [16],
        num_classes: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
        spatial_shape: Tuple[int, int] = (486, 192)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.spatial_shape = spatial_shape

        c1 = conv_channels[0] if len(conv_channels) > 0 else 32
        c2 = conv_channels[1] if len(conv_channels) > 1 else c1

        # Stage 1
        self.block1_a = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
            SparseResBlock(c1, spatial_shape=spatial_shape)
        )
        self.block1_b = nn.Sequential(
            spnn.SubMConv2d(in_channels, c1, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c1),
            spnn.ReLU(),
            SparseResBlock(c1, spatial_shape=spatial_shape)
        )

        # Cross-Attention 1
        self.cross_attn1_a = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.cross_attn1_b = nn.MultiheadAttention(embed_dim=c1, num_heads=num_heads, batch_first=True)
        self.norm1_a = nn.LayerNorm(c1)
        self.norm1_b = nn.LayerNorm(c1)

        # Stage 2
        self.block2_a = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
            SparseResBlock(c2, spatial_shape=spatial_shape)
        )
        self.block2_b = nn.Sequential(
            spnn.SubMConv2d(c1, c2, kernel_size=3, spatial_shape=spatial_shape),
            spnn.BatchNorm(c2),
            spnn.ReLU(),
            SparseResBlock(c2, spatial_shape=spatial_shape)
        )

        # Cross-Attention 2
        self.cross_attn2_a = nn.MultiheadAttention(embed_dim=c2, num_heads=num_heads, batch_first=True)
        self.cross_attn2_b = nn.MultiheadAttention(embed_dim=c2, num_heads=num_heads, batch_first=True)
        self.norm2_a = nn.LayerNorm(c2)
        self.norm2_b = nn.LayerNorm(c2)

        self.pool = spnn.GlobalAvgPooling()
        pooled_dim = c2
        fusion_dim = pooled_dim * 4

        classifier_layers = []
        current_dim = fusion_dim
        for fc_dim in fc_dims:
            classifier_layers.append(nn.Linear(current_dim, fc_dim))
            classifier_layers.append(nn.ReLU())
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            current_dim = fc_dim

        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _extract_sparse_tensors(self, data) -> Tuple[SparseTensor, SparseTensor]:
        if isinstance(data, (tuple, list)) and len(data) == 2:
            return data[0], data[1]
        elif isinstance(data, dict) and "view_a" in data and "view_b" in data:
            if isinstance(data["view_a"], SparseTensor):
                return data["view_a"], data["view_b"]

        if HeteroData is not None and isinstance(data, HeteroData):
            batch_a = getattr(data["view_a"], "batch", None)
            if batch_a is None:
                batch_a = torch.zeros(data["view_a"].x.size(0), dtype=torch.long, device=data["view_a"].x.device)
            coords_a = torch.cat([batch_a.unsqueeze(1), data["view_a"].pos.long()], dim=1)
            feats_a = data["view_a"].x[:, :self.in_channels]
            tensor_a = SparseTensor(feats=feats_a, coords=coords_a)

            batch_b = getattr(data["view_b"], "batch", None)
            if batch_b is None:
                batch_b = torch.zeros(data["view_b"].x.size(0), dtype=torch.long, device=data["view_b"].x.device)
            coords_b = torch.cat([batch_b.unsqueeze(1), data["view_b"].pos.long()], dim=1)
            feats_b = data["view_b"].x[:, :self.in_channels]
            tensor_b = SparseTensor(feats=feats_b, coords=coords_b)

            return tensor_a, tensor_b

        raise TypeError("Unsupported data format for DualViewResNetMultiStageCrossAttentionSparseCNN.")

    def forward(self, data) -> torch.Tensor:
        tensor_a, tensor_b = self._extract_sparse_tensors(data)

        # Stage 1
        x_a = self.block1_a(tensor_a)
        x_b = self.block1_b(tensor_b)

        batch_size = int(max(
            x_a.C[:, 0].max().item() if x_a.C.numel() > 0 else 0,
            x_b.C[:, 0].max().item() if x_b.C.numel() > 0 else 0
        )) + 1

        max_planes = self.spatial_shape[0]

        # Stage 1 Cross Attention
        summary1_a, keys1_a = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], max_planes, num_batch=batch_size)
        summary1_b, keys1_b = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], max_planes, num_batch=batch_size)

        seq1_a = summary1_a.view(batch_size, max_planes, -1)
        seq1_b = summary1_b.view(batch_size, max_planes, -1)

        attn1_a, _ = self.cross_attn1_a(query=seq1_a, key=seq1_b, value=seq1_b)
        attn1_b, _ = self.cross_attn1_b(query=seq1_b, key=seq1_a, value=seq1_a)

        seq1_a = self.norm1_a(seq1_a + attn1_a).view(-1, seq1_a.size(-1))
        seq1_b = self.norm1_b(seq1_b + attn1_b).view(-1, seq1_b.size(-1))

        x_a = SparseTensor(feats=x_a.F + seq1_a[keys1_a], coords=x_a.C)
        x_b = SparseTensor(feats=x_b.F + seq1_b[keys1_b], coords=x_b.C)

        # Stage 2
        x_a = self.block2_a(x_a)
        x_b = self.block2_b(x_b)

        # Stage 2 Cross Attention
        summary2_a, keys2_a = _scatter_plane_summary(x_a.F, x_a.C[:, 0], x_a.C[:, 1], max_planes, num_batch=batch_size)
        summary2_b, keys2_b = _scatter_plane_summary(x_b.F, x_b.C[:, 0], x_b.C[:, 1], max_planes, num_batch=batch_size)

        seq2_a = summary2_a.view(batch_size, max_planes, -1)
        seq2_b = summary2_b.view(batch_size, max_planes, -1)

        attn2_a, _ = self.cross_attn2_a(query=seq2_a, key=seq2_b, value=seq2_b)
        attn2_b, _ = self.cross_attn2_b(query=seq2_b, key=seq2_a, value=seq2_a)

        seq2_a = self.norm2_a(seq2_a + attn2_a).view(-1, seq2_a.size(-1))
        seq2_b = self.norm2_b(seq2_b + attn2_b).view(-1, seq2_b.size(-1))

        x_a = SparseTensor(feats=x_a.F + seq2_a[keys2_a], coords=x_a.C)
        x_b = SparseTensor(feats=x_b.F + seq2_b[keys2_b], coords=x_b.C)

        pooled_a = self.pool(x_a)
        pooled_b = self.pool(x_b)

        fused = torch.cat([
            pooled_a,
            pooled_b,
            torch.abs(pooled_a - pooled_b),
            pooled_a * pooled_b,
        ], dim=-1)

        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TimingAwareGNN(nn.Module):
    """
    Timing-aware GNN for MINOS CC/NC classification.

    Exploits dual-ended strip readout timing (East/West) alongside charge and
    spatial features. Operates on a merged homogeneous graph containing hits
    from both U and V views, with cross-view edges and spacetime-kNN edges.

    Node features (9-dim):
        PE_east_log, PE_west_log, t_scaled, dt_scaled, tpos_norm, z_norm,
        view_flag, readout_valid_east, readout_valid_west

    Edge features (6-dim):
        Δz, Δtpos, Δt_scaled, ||r||, causal_flag, same_view_flag

    Architecture:
        - Node encoder MLP: in_channels → hidden_channels
        - Edge encoder MLP: edge_dim → hidden_channels
        - N stacked EdgeConv-style message passing layers with:
          * Edge-feature-aware aggregation
          * Residual gated updates
          * GraphNorm normalization
        - Concat(global_mean_pool, global_max_pool) readout
        - Classification MLP head with dropout
    """

    def __init__(
        self,
        in_channels: int = 9,
        edge_dim: int = 6,
        hidden_channels: int = 48,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.15,
        aux_weight: float = 0.0,
    ):
        super().__init__()

        if EdgeConv is None or global_mean_pool is None or global_max_pool is None or GraphNorm is None:
            raise ImportError(
                "torch_geometric is required for TimingAwareGNN. "
                "Install PyTorch Geometric to use this model."
            )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # Node encoder: raw features → hidden_channels
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Edge encoder: raw edge features → hidden_channels
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Message passing layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.edge_mlps = nn.ModuleList()

        for _ in range(num_layers):
            # EdgeConv-style MLP: takes concatenated [x_i, x_j, e_ij] → hidden
            edge_mlp = nn.Sequential(
                nn.Linear(2 * hidden_channels + hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.edge_mlps.append(edge_mlp)

            self.convs.append(
                EdgeConv(nn=edge_mlp, aggr="max")
            )
            self.norms.append(GraphNorm(hidden_channels))
            self.gates.append(nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.Sigmoid(),
            ))

        # Readout: concat(mean_pool, max_pool) → 2 * hidden_channels
        pooled_dim = hidden_channels * 2
        self.readout = nn.Sequential(
            nn.Linear(pooled_dim, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

        # Optional auxiliary per-hit primary-lepton head.
        #
        # Constructed LAST and only when enabled, which matters for two reasons:
        #   * With aux_weight=0.0 (the default) the module is bit-identical to the
        #     original: same state_dict keys, same 53,738 params, same init. The
        #     existing gnn_timing / gnn_timing_v1_on_v3 configs and their saved
        #     checkpoints keep working untouched.
        #   * Because nothing is drawn from the RNG before this point that wasn't
        #     drawn before, every shared layer initialises identically with and
        #     without the head. So the already-trained gnn_timing_v1_on_v3
        #     (0.9324, seed 42) is an exactly matched control for the aux run --
        #     no separate control training needed.
        #
        # Note the `convs` ModuleList above is never called in forward(), but its
        # construction is NOT removable: EdgeConv.__init__ calls reset_parameters()
        # on the MLP handed to it, so `convs[i].nn IS edge_mlps[i]` (they share
        # parameters -- all 53,738 are live) and dropping it would re-roll the
        # initialisation of every later layer.
        self.aux_weight = float(aux_weight)
        if self.aux_weight > 0.0:
            self.hit_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.SiLU(),
                nn.Linear(hidden_channels // 2, 1),
            )

    def _encode_nodes(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Message-passing body, shared by forward() and predict_hit_lepton_prob()."""
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Encode nodes and edges
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr) if edge_attr is not None else None

        # Message passing with residual gated updates
        for i, (conv, norm, gate) in enumerate(zip(self.convs, self.norms, self.gates)):
            # Build edge-feature-enriched node pairs for EdgeConv
            # EdgeConv expects nn(cat(x_i, x_j - x_i)) by default,
            # but we override by augmenting node features with edge features
            # via a custom approach: we pre-concatenate edge features to target nodes
            src, dst = edge_index[0], edge_index[1]

            # Construct input for conv: cat(h[src], h[dst], edge_feat)
            if e is not None:
                msg_input = torch.cat([h[src], h[dst], e], dim=-1)
            else:
                msg_input = torch.cat([h[src], h[dst], torch.zeros(src.size(0), self.hidden_channels, device=h.device)], dim=-1)

            # Run the edge MLP manually and aggregate
            msg = self.edge_mlps[i](msg_input)

            # Aggregate messages (max aggregation, matching EdgeConv)
            agg = torch.zeros_like(h)
            agg.scatter_reduce_(0, dst.unsqueeze(1).expand(-1, self.hidden_channels), msg, reduce="amax", include_self=False)

            # Normalize
            agg = norm(torch.relu(agg), batch)

            # Gated residual update
            gate_val = gate(torch.cat([h, agg], dim=-1))
            h = h + gate_val * nn.functional.dropout(agg, p=self.dropout, training=self.training)

        return h, batch

    def forward(self, data):
        h, batch = self._encode_nodes(data)

        # Global pooling: concat(mean, max)
        pooled_mean = global_mean_pool(h, batch)
        pooled_max = global_max_pool(h, batch)
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
        logits = self.readout(pooled)

        if self.training and self.aux_weight > 0.0:
            return logits, self.aux_weight * self._aux_loss(h, batch, data)
        return logits

    def _aux_loss(self, h: torch.Tensor, batch: torch.Tensor, data) -> torch.Tensor:
        """
        Masked soft-target BCE over nodes, averaged per event then across events.

        Simpler than the transformer's equivalent: PyG concatenates node
        attributes, so ``hit_lepton_frac`` is already aligned with ``h`` -- no dense
        padding, no mask, no truncation. Only ``has_lepton_truth`` (per graph) needs
        broadcasting to nodes.
        """
        target = get_hit_lepton_frac(data).view(-1)
        has_truth = get_has_lepton_truth(data, required=False)
        if has_truth is None:
            node_ok = torch.ones_like(target, dtype=torch.bool)
        else:
            node_ok = has_truth.view(-1).bool()[batch]

        if not bool(node_ok.any()):
            return h.sum() * 0.0

        hit_logits = self.hit_head(h).squeeze(-1)
        per_hit = nn.functional.binary_cross_entropy_with_logits(
            hit_logits, target, reduction="none"
        ) * node_ok.to(h.dtype)

        n_graphs = int(batch.max()) + 1
        sums = torch.zeros(n_graphs, device=h.device, dtype=h.dtype).index_add_(0, batch, per_hit)
        counts = torch.zeros(n_graphs, device=h.device, dtype=h.dtype).index_add_(
            0, batch, node_ok.to(h.dtype)
        )
        keep = counts > 0
        return (sums[keep] / counts[keep]).mean()

    @torch.no_grad()
    def predict_hit_lepton_prob(self, data) -> torch.Tensor:
        """
        Per-hit primary-lepton probability, node-indexed (aligned with ``data.x``).

        This is the output that makes the model checkable against real data --
        it can be validated on FD cosmic and rock muons and on muon-removed
        control samples, which an event-level score cannot.
        """
        if self.aux_weight <= 0.0:
            raise RuntimeError(
                "This model was built with aux_weight=0.0 and has no hit head."
            )
        self.eval()
        h, _ = self._encode_nodes(data)
        return torch.sigmoid(self.hit_head(h).squeeze(-1))

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TimingAwareGNNV2(nn.Module):
    """
    Correctness-fixed copy of TimingAwareGNN, added as a separate class rather
    than editing TimingAwareGNN in place so the original (currently the #1
    leaderboard entry) stays reproducible against its logged checkpoint/row.

    Two fixes relative to TimingAwareGNN:
      1. Removes the dead `self.convs` ModuleList of `EdgeConv` layers, which
         was constructed in TimingAwareGNN.__init__ but never called in
         forward() (message passing was already done by a hand-rolled
         duplicate of the same computation).
      2. Edge embeddings are now refined every layer via an `edge_update` MLP
         that recomputes each edge's embedding from the freshly updated
         endpoint node states, instead of being computed once before the
         layer loop and reused unchanged across all `num_layers` "stacked"
         layers.

    Node/edge feature layout is unchanged from TimingAwareGNN (see that
    class's docstring).
    """

    def __init__(
        self,
        in_channels: int = 9,
        edge_dim: int = 6,
        hidden_channels: int = 48,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()

        if global_mean_pool is None or global_max_pool is None or GraphNorm is None:
            raise ImportError(
                "torch_geometric is required for TimingAwareGNNV2. "
                "Install PyTorch Geometric to use this model."
            )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.norms = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.edge_mlps = nn.ModuleList()
        self.edge_updates = nn.ModuleList()

        for _ in range(num_layers):
            self.edge_mlps.append(nn.Sequential(
                nn.Linear(3 * hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ))
            self.norms.append(GraphNorm(hidden_channels))
            self.gates.append(nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.Sigmoid(),
            ))
            self.edge_updates.append(nn.Sequential(
                nn.Linear(3 * hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
            ))

        pooled_dim = hidden_channels * 2
        self.readout = nn.Sequential(
            nn.Linear(pooled_dim, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr) if edge_attr is not None else torch.zeros(
            edge_index.size(1), self.hidden_channels, device=h.device
        )

        src, dst = edge_index[0], edge_index[1]

        for i in range(self.num_layers):
            msg_input = torch.cat([h[src], h[dst], e], dim=-1)
            msg = self.edge_mlps[i](msg_input)

            agg = torch.zeros_like(h)
            agg.scatter_reduce_(0, dst.unsqueeze(1).expand(-1, self.hidden_channels), msg, reduce="amax", include_self=False)

            agg = self.norms[i](torch.relu(agg), batch)

            gate_val = self.gates[i](torch.cat([h, agg], dim=-1))
            h = h + gate_val * nn.functional.dropout(agg, p=self.dropout, training=self.training)

            # Refine edge embeddings using the updated endpoint states so edge
            # (i.e. timing-pair) representations evolve with depth like node
            # features do, instead of staying frozen at their layer-0 encoding.
            e = self.edge_updates[i](torch.cat([h[src], h[dst], e], dim=-1))

        pooled_mean = global_mean_pool(h, batch)
        pooled_max = global_max_pool(h, batch)
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)

        return self.readout(pooled)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



# ── Hit-level set transformer (feature_mode='hitset') ────────────────────

class _HitSetBlock(nn.Module):
    """Pre-LN encoder block over strip tokens, with an additive attention bias."""

    def __init__(self, d_model: int, num_heads: int, dropout: float, ff_mult: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.ff(self.norm2(x))
        return x


class HitSetTransformer(nn.Module):
    """
    Transformer over the set of calibrated strips in a Far Detector snarl,
    consuming ``feature_mode='hitset'`` events (edgeless homogeneous ``Data``,
    14 features per hit -- see ``MINOSMultiViewGraphDataset``).

    Motivation. 25 prior leaderboard runs spanning sparse CNNs, dense CNNs,
    cross-attention and two GNN families all land in ROC-AUC 0.924-0.940, and
    the gnn_timing_v3 ablation shows the timing features contribute ~nothing.
    Architecture and statistics are both exhausted, so the point of this model
    is not the architecture -- it is the auxiliary head. ``thstp`` supplies a
    soft per-strip primary-lepton charge fraction, turning one bit of
    supervision per event into 50-400 targets per event. A diagnostic oracle
    that simply counts truth-tagged hits reaches ROC-AUC 0.9901, versus 0.9396
    for the best current model, so the information is present at strip level
    and is not being extracted.

    ``aux_weight`` switches the segmentation head:
      * ``0.0`` -- event head only; the architecture-swap control.
      * ``> 0`` -- adds ``aux_weight * BCE(hit_logits, hit_lepton_frac)``.

    The aux loss is computed only in ``training`` mode, so ``val_loss`` stays
    the event loss alone and remains comparable with every other leaderboard
    row.

    Two design choices are worth flagging:

    * **Pairwise attention bias** (``use_pair_bias``). Muon-vs-shower is a
      statement about *relative* structure -- a muon is a chain of hits with
      small plane-to-plane steps in ``tpos`` and a consistent dt/dz. A bias
      built from (dz, dtpos, dt, same_view, dplane, |dr|) is added to the
      attention logits so that relation is directly available rather than
      having to be inferred from absolute coordinates. It is computed once and
      reused across all blocks (as in Particle Transformer), which keeps it
      affordable on CPU.
    * **Truncation.** Events longer than ``max_hits`` keep their first
      ``max_hits`` tokens. The dataset sorts hits by (z, tpos), so this retains
      the upstream vertex region and the start of the track. At the default 256
      this affects ~13% of events (longest observed: 1008 hits).
    """

    # Column indices into the 14-feature hit vector built by
    # MINOSMultiViewGraphDataset._build_hitset_event.
    _IDX_T = 2
    _IDX_TPOS = 4
    _IDX_Z = 5
    _IDX_VIEW = 6

    # z_norm = (z_metres - 15) / 15 and the MINOS plane pitch is 5.94 cm, so
    # one plane step is 0.0594 / 15 in normalised units. Converting dz back to
    # an approximate plane count keeps the pair-bias MLP well conditioned --
    # adjacent planes differ by ~0.004 in z_norm, which would otherwise be
    # numerically indistinguishable from zero at initialisation.
    _PLANES_PER_ZNORM = 15.0 / 0.0594

    def __init__(
        self,
        in_channels: int = 14,
        d_model: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_hits: int = 256,
        use_pair_bias: bool = True,
        pair_hidden: int = 8,
        num_freqs: int = 4,
        aux_weight: float = 0.0,
        num_classes: int = 2,
    ):
        super().__init__()

        if to_dense_batch is None:
            raise ImportError(
                "torch_geometric is required for HitSetTransformer. "
                "Install PyTorch Geometric to use this model."
            )

        self.in_channels = in_channels
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_hits = max_hits
        self.use_pair_bias = use_pair_bias
        self.num_freqs = num_freqs
        self.aux_weight = float(aux_weight)

        # Fourier features for z and tpos (attention handles geometry better in
        # a periodic basis than from raw scalars).
        self.register_buffer(
            "_freqs", (2.0 ** torch.arange(num_freqs)) * torch.pi, persistent=False
        )
        input_dim = in_channels + 4 * num_freqs

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        if use_pair_bias:
            self.pair_mlp = nn.Sequential(
                nn.Linear(6, pair_hidden),
                nn.GELU(),
                nn.Linear(pair_hidden, num_heads),
            )

        self.blocks = nn.ModuleList([
            _HitSetBlock(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

        # CLS + masked mean + masked max, mirroring the dual-pool readout used
        # by TimingAwareGNNV2.
        self.event_head = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        self.hit_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def _fourier(self, v: torch.Tensor) -> torch.Tensor:
        """[B, N] -> [B, N, 2 * num_freqs]."""
        a = v.unsqueeze(-1) * self._freqs
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)

    def _pair_bias(self, dense: torch.Tensor) -> torch.Tensor:
        """[B, N, F] -> [B, heads, N, N] additive attention bias."""
        z = dense[..., self._IDX_Z]
        tp = dense[..., self._IDX_TPOS]
        t = dense[..., self._IDX_T]
        view = dense[..., self._IDX_VIEW]

        dz = z.unsqueeze(-1) - z.unsqueeze(-2)
        dtp = tp.unsqueeze(-1) - tp.unsqueeze(-2)
        dt = t.unsqueeze(-1) - t.unsqueeze(-2)
        same_view = (view.unsqueeze(-1) == view.unsqueeze(-2)).float()
        dplane = torch.clamp(dz * self._PLANES_PER_ZNORM, -20.0, 20.0) / 20.0
        dr = torch.sqrt(dz * dz + dtp * dtp + 1e-12)

        pair = torch.stack([dz, dtp, dt, same_view, dplane, dr], dim=-1)
        return self.pair_mlp(pair).permute(0, 3, 1, 2)

    def _aux_loss(
        self,
        hit_logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        has_truth: torch.Tensor,
    ) -> torch.Tensor:
        """Masked soft-target BCE, averaged per event then across events."""
        valid = mask & has_truth.view(-1, 1)
        counts = valid.sum(dim=1)
        keep = counts > 0
        if not bool(keep.any()):
            return hit_logits.sum() * 0.0

        per_hit = nn.functional.binary_cross_entropy_with_logits(
            hit_logits, target, reduction="none"
        ) * valid.float()
        per_event = per_hit.sum(dim=1)[keep] / counts[keep].float()
        return per_event.mean()

    def forward(self, data):
        x = data.x
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        dense, mask = to_dense_batch(x, batch, max_num_nodes=self.max_hits)
        B, N, _ = dense.shape

        feats = torch.cat(
            [
                dense,
                self._fourier(dense[..., self._IDX_Z]),
                self._fourier(dense[..., self._IDX_TPOS]),
            ],
            dim=-1,
        )
        tokens = self.input_proj(feats)
        seq = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)

        # The CLS token is never padding, so no query row is fully masked and
        # the softmax cannot produce NaN.
        seq_mask = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1
        )

        L = N + 1
        bias = torch.zeros(B, self.num_heads, L, L, device=dense.device, dtype=dense.dtype)
        if self.use_pair_bias:
            bias[:, :, 1:, 1:] = self._pair_bias(dense)
        # Fold padding into the float bias rather than passing a separate bool
        # key_padding_mask (mixing mask dtypes is deprecated in torch>=2.1).
        bias = bias.masked_fill(~seq_mask[:, None, None, :], float("-inf"))
        attn_mask = bias.reshape(B * self.num_heads, L, L)

        for blk in self.blocks:
            seq = blk(seq, attn_mask)
        seq = self.norm_out(seq)

        cls_out = seq[:, 0]
        tok = seq[:, 1:]
        m = mask.unsqueeze(-1).to(tok.dtype)
        pooled_mean = (tok * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        pooled_max = tok.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        logits = self.event_head(torch.cat([cls_out, pooled_mean, pooled_max], dim=-1))

        if self.training and self.aux_weight > 0.0:
            target, _ = to_dense_batch(
                get_hit_lepton_frac(data).unsqueeze(-1), batch, max_num_nodes=self.max_hits
            )
            has_truth = get_has_lepton_truth(data, required=False)
            if has_truth is None:
                has_truth = torch.ones(B, dtype=torch.bool, device=x.device)
            aux = self._aux_loss(
                self.hit_head(tok).squeeze(-1),
                target.squeeze(-1),
                mask,
                has_truth.view(-1).bool(),
            )
            return logits, self.aux_weight * aux

        return logits

    @torch.no_grad()
    def predict_hit_lepton_prob(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Per-hit primary-lepton probability, for validating the segmentation
        head against control samples and for deriving interpretable
        muon-chain variables. Returns ``(probs [B, N], mask [B, N])``.
        """
        self.eval()
        x = data.x
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        dense, mask = to_dense_batch(x, batch, max_num_nodes=self.max_hits)
        B, N, _ = dense.shape
        feats = torch.cat(
            [
                dense,
                self._fourier(dense[..., self._IDX_Z]),
                self._fourier(dense[..., self._IDX_TPOS]),
            ],
            dim=-1,
        )
        seq = torch.cat(
            [self.cls_token.expand(B, -1, -1), self.input_proj(feats)], dim=1
        )
        seq_mask = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1
        )
        L = N + 1
        bias = torch.zeros(B, self.num_heads, L, L, device=dense.device, dtype=dense.dtype)
        if self.use_pair_bias:
            bias[:, :, 1:, 1:] = self._pair_bias(dense)
        bias = bias.masked_fill(~seq_mask[:, None, None, :], float("-inf"))
        attn_mask = bias.reshape(B * self.num_heads, L, L)

        for blk in self.blocks:
            seq = blk(seq, attn_mask)
        seq = self.norm_out(seq)
        return torch.sigmoid(self.hit_head(seq[:, 1:]).squeeze(-1)), mask

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
