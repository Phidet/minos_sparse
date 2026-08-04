from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn

from src.torchsparse import SparseTensor
import src.torchsparse.nn as spnn

try:
    from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroConv = None
    SAGEConv = None
    global_mean_pool = None
    HeteroData = None



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




