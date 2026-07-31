from typing import List, Tuple
import torch
import torch.nn as nn

from src.torchsparse import SparseTensor
import src.torchsparse.nn as spnn


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
