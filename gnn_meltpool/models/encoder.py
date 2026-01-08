"""
Node and Edge Encoders for MeshGraphNet
Encodes raw features into latent representations
"""

import torch
import torch.nn as nn
from typing import Optional


class MLP(nn.Module):
    """Multi-layer Perceptron with LayerNorm and residual connections."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = 'relu',
        use_layer_norm: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function ('relu', 'elu', 'gelu')
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        return self.network(x)


class NodeEncoder(nn.Module):
    """Encodes node features into latent space."""

    def __init__(
        self,
        node_feature_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = 'relu'
    ):
        """
        Args:
            node_feature_dim: Dimension of input node features
            latent_dim: Dimension of latent representation
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function
        """
        super().__init__()

        self.mlp = MLP(
            input_dim=node_feature_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            use_layer_norm=True
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Encode node features.

        Args:
            node_features: Node features [num_nodes, node_feature_dim]
                          Features: [x, y, z, T, phase, u, v, w, phi, type]

        Returns:
            Encoded node features [num_nodes, latent_dim]
        """
        return self.mlp(node_features)


class EdgeEncoder(nn.Module):
    """Encodes edge features into latent space."""

    def __init__(
        self,
        edge_feature_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = 'relu'
    ):
        """
        Args:
            edge_feature_dim: Dimension of input edge features
            latent_dim: Dimension of latent representation
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function
        """
        super().__init__()

        self.mlp = MLP(
            input_dim=edge_feature_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            use_layer_norm=True
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Encode edge features.

        Args:
            edge_features: Edge features [num_edges, edge_feature_dim]
                          Features: [dx, dy, dz, distance]

        Returns:
            Encoded edge features [num_edges, latent_dim]
        """
        return self.mlp(edge_features)


def build_edge_features(
    pos: torch.Tensor,
    edge_index: torch.Tensor
) -> torch.Tensor:
    """
    Build edge features from node positions.

    Args:
        pos: Node positions [num_nodes, 3] (x, y, z)
        edge_index: Edge connectivity [2, num_edges]

    Returns:
        Edge features [num_edges, 4] (dx, dy, dz, distance)
    """
    row, col = edge_index

    # Relative position
    relative_pos = pos[col] - pos[row]  # [num_edges, 3]

    # Euclidean distance
    distance = torch.norm(relative_pos, dim=-1, keepdim=True)  # [num_edges, 1]

    # Concatenate [dx, dy, dz, distance]
    edge_features = torch.cat([relative_pos, distance], dim=-1)

    return edge_features
