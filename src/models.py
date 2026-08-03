from typing import List, Tuple, Dict
import torch
import torch.nn as nn

from src.torchsparse import SparseTensor
import src.torchsparse.nn as spnn

try:
    from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
except ImportError:
    HeteroConv = None
    SAGEConv = None
    global_mean_pool = None


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


