"""
Message Passing Processor for MeshGraphNet
Implements graph neural network layers for information propagation
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from typing import Optional

from .encoder import MLP


class MessagePassingLayer(MessagePassing):
    """
    Single message passing layer.
    Updates both edge and node features through message passing.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = 'relu',
        aggr: str = 'sum'
    ):
        """
        Args:
            latent_dim: Latent feature dimension
            hidden_dim: Hidden layer dimension for MLPs
            num_layers: Number of layers in MLPs
            activation: Activation function
            aggr: Aggregation function ('sum', 'mean', 'max')
        """
        super().__init__(aggr=aggr)

        self.latent_dim = latent_dim

        # Edge update MLP: [sender, receiver, edge] -> edge
        self.edge_mlp = MLP(
            input_dim=3 * latent_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            use_layer_norm=True
        )

        # Node update MLP: [node, aggregated_messages] -> node
        self.node_mlp = MLP(
            input_dim=2 * latent_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            use_layer_norm=True
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through message passing layer.

        Args:
            x: Node features [num_nodes, latent_dim]
            edge_attr: Edge features [num_edges, latent_dim]
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            Updated node features [num_nodes, latent_dim]
            Updated edge features [num_edges, latent_dim]
        """
        row, col = edge_index

        # Update edges
        edge_features = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_attr_new = edge_attr + self.edge_mlp(edge_features)  # Residual

        # Update nodes via message passing
        x_new = self.propagate(edge_index, x=x, edge_attr=edge_attr_new)

        return x_new, edge_attr_new

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Construct messages from neighbors.

        Args:
            x_j: Features of neighbor nodes [num_edges, latent_dim]
            edge_attr: Edge features [num_edges, latent_dim]

        Returns:
            Messages [num_edges, latent_dim]
        """
        # Use edge features as messages
        return edge_attr

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Update node features with aggregated messages.

        Args:
            aggr_out: Aggregated messages [num_nodes, latent_dim]
            x: Current node features [num_nodes, latent_dim]

        Returns:
            Updated node features [num_nodes, latent_dim]
        """
        node_input = torch.cat([x, aggr_out], dim=-1)
        x_new = x + self.node_mlp(node_input)  # Residual connection
        return x_new


class Processor(nn.Module):
    """
    Graph Network Processor.
    Stacks multiple message passing layers.
    """

    def __init__(
        self,
        latent_dim: int,
        num_layers: int = 15,
        hidden_dim: int = 128,
        mlp_layers: int = 2,
        activation: str = 'relu',
        aggr: str = 'sum'
    ):
        """
        Args:
            latent_dim: Latent feature dimension
            num_layers: Number of message passing layers
            hidden_dim: Hidden dimension for MLPs
            mlp_layers: Number of layers in MLPs
            activation: Activation function
            aggr: Aggregation function
        """
        super().__init__()

        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            MessagePassingLayer(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_layers=mlp_layers,
                activation=activation,
                aggr=aggr
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process graph through multiple message passing steps.

        Args:
            x: Node features [num_nodes, latent_dim]
            edge_attr: Edge features [num_edges, latent_dim]
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            Processed node features [num_nodes, latent_dim]
            Processed edge features [num_edges, latent_dim]
        """
        for layer in self.layers:
            x, edge_attr = layer(x, edge_attr, edge_index)

        return x, edge_attr


class GraphNetworkBlock(nn.Module):
    """
    Complete graph network block with edge and node updates.
    Alternative implementation with separate edge and node networks.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = 'relu'
    ):
        """
        Args:
            latent_dim: Latent feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            activation: Activation function
        """
        super().__init__()

        # Edge network
        self.edge_network = MLP(
            input_dim=3 * latent_dim,  # [sender, receiver, edge]
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation
        )

        # Node network
        self.node_network = MLP(
            input_dim=2 * latent_dim,  # [node, aggregated]
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, latent_dim]
            edge_attr: Edge features [num_edges, latent_dim]
            edge_index: Edge connectivity [2, num_edges]

        Returns:
            Updated node and edge features
        """
        row, col = edge_index

        # Update edges
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        edge_attr_new = edge_attr + self.edge_network(edge_input)

        # Aggregate edge features to nodes
        num_nodes = x.size(0)
        aggregated = torch.zeros(num_nodes, edge_attr_new.size(1), device=x.device)
        aggregated.index_add_(0, col, edge_attr_new)

        # Update nodes
        node_input = torch.cat([x, aggregated], dim=-1)
        x_new = x + self.node_network(node_input)

        return x_new, edge_attr_new
