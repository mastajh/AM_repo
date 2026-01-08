"""
Node and Edge Encoders for MeshGraphNet
Transforms input features into latent space
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer Perceptron with LayerNorm and ReLU activation"""

    def __init__(self, in_dim, out_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NodeEncoder(nn.Module):
    """
    Encodes node features into latent space

    Args:
        node_dim: Input node feature dimension (11)
            - Position: x, y, z (3)
            - Velocity: u, v, w (3)
            - Pressure: p (1)
            - Wall distance: d_wall (1)
            - Node type: one-hot (3)
        latent_dim: Latent dimension (128)
    """

    def __init__(self, node_dim=11, latent_dim=128, hidden_dim=128, num_layers=2):
        super().__init__()
        self.encoder = MLP(node_dim, latent_dim, hidden_dim, num_layers)

    def forward(self, x):
        """
        Args:
            x: Node features [num_nodes, node_dim]

        Returns:
            Node embeddings [num_nodes, latent_dim]
        """
        return self.encoder(x)


class EdgeEncoder(nn.Module):
    """
    Encodes edge features into latent space

    Args:
        edge_dim: Input edge feature dimension (4)
            - Relative position: dx, dy, dz (3)
            - Euclidean distance: ||r|| (1)
        latent_dim: Latent dimension (128)
    """

    def __init__(self, edge_dim=4, latent_dim=128, hidden_dim=128, num_layers=2):
        super().__init__()
        self.encoder = MLP(edge_dim, latent_dim, hidden_dim, num_layers)

    def forward(self, edge_attr):
        """
        Args:
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Edge embeddings [num_edges, latent_dim]
        """
        return self.encoder(edge_attr)
