"""
Graph Processor with Message Passing
Implements the "Process" step of Encode-Process-Decode
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from .encoder import MLP


class GraphNetBlock(MessagePassing):
    """
    Single message passing layer with residual connections

    Message passing:
    1. Edge update: e'_ij = e_ij + EdgeMLP([h_i, h_j, e_ij])
    2. Node update: h'_i = h_i + NodeMLP([h_i, Σ_j e'_ij])

    Args:
        latent_dim: Latent space dimension (128)
        hidden_dim: Hidden layer dimension (128)
        num_mlp_layers: Number of MLP layers (2)
    """

    def __init__(self, latent_dim=128, hidden_dim=128, num_mlp_layers=2):
        super().__init__(aggr='sum')  # Aggregate messages by summation

        # Edge update MLP: [sender, receiver, edge_attr] -> edge_attr
        self.edge_mlp = MLP(
            in_dim=3 * latent_dim,
            out_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers
        )

        # Node update MLP: [node, aggregated_messages] -> node
        self.node_mlp = MLP(
            in_dim=2 * latent_dim,
            out_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [num_nodes, latent_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, latent_dim]

        Returns:
            x: Updated node features [num_nodes, latent_dim]
            edge_attr: Updated edge features [num_edges, latent_dim]
        """
        row, col = edge_index  # Source and target nodes

        # Edge update with residual connection
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_attr = edge_attr + self.edge_mlp(edge_input)

        # Node update with residual connection
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        node_input = torch.cat([x, aggr_out], dim=-1)
        x = x + self.node_mlp(node_input)

        return x, edge_attr

    def message(self, edge_attr):
        """
        Construct messages from edge attributes
        Called internally by propagate()
        """
        return edge_attr


class GraphProcessor(nn.Module):
    """
    Stack of message passing layers

    Args:
        latent_dim: Latent space dimension (128)
        num_layers: Number of message passing layers (15)
        hidden_dim: Hidden layer dimension (128)
        num_mlp_layers: Number of MLP layers per block (2)
    """

    def __init__(self, latent_dim=128, num_layers=15, hidden_dim=128, num_mlp_layers=2):
        super().__init__()

        self.layers = nn.ModuleList([
            GraphNetBlock(latent_dim, hidden_dim, num_mlp_layers)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x, edge_index, edge_attr):
        """
        Apply multiple rounds of message passing

        Args:
            x: Node features [num_nodes, latent_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, latent_dim]

        Returns:
            x: Processed node features [num_nodes, latent_dim]
            edge_attr: Processed edge features [num_edges, latent_dim]
        """
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)

        return x, edge_attr
