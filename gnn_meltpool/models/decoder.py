"""
Output Decoder for MeshGraphNet
Decodes latent node features to predict state changes
"""

import torch
import torch.nn as nn
from typing import Optional

from .encoder import MLP


class Decoder(nn.Module):
    """
    Decodes latent node features to output predictions.
    Predicts the change in node features (delta).
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = 'relu',
        predict_delta: bool = True
    ):
        """
        Args:
            latent_dim: Latent feature dimension
            output_dim: Output feature dimension (number of predicted features)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function
            predict_delta: If True, predict state changes; if False, predict absolute state
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.predict_delta = predict_delta

        self.mlp = MLP(
            input_dim=latent_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            use_layer_norm=True
        )

    def forward(self, node_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent features to predictions.

        Args:
            node_latent: Latent node features [num_nodes, latent_dim]

        Returns:
            Predicted output [num_nodes, output_dim]
            If predict_delta=True, returns state changes
            If predict_delta=False, returns absolute state
        """
        output = self.mlp(node_latent)
        return output


class MultiTaskDecoder(nn.Module):
    """
    Multi-task decoder for predicting different physical quantities.
    Can predict position, velocity, temperature, phase, etc. separately.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dims: dict[str, int],
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = 'relu'
    ):
        """
        Args:
            latent_dim: Latent feature dimension
            output_dims: Dictionary mapping task names to output dimensions
                        e.g., {'velocity': 3, 'temperature': 1, 'phase': 1}
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function
        """
        super().__init__()

        self.output_dims = output_dims
        self.decoders = nn.ModuleDict()

        for task_name, out_dim in output_dims.items():
            self.decoders[task_name] = MLP(
                input_dim=latent_dim,
                output_dim=out_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                activation=activation,
                use_layer_norm=True
            )

    def forward(self, node_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Decode latent features to multiple task outputs.

        Args:
            node_latent: Latent node features [num_nodes, latent_dim]

        Returns:
            Dictionary of predictions for each task
        """
        outputs = {}
        for task_name, decoder in self.decoders.items():
            outputs[task_name] = decoder(node_latent)

        return outputs


class AdaptiveDecoder(nn.Module):
    """
    Adaptive decoder with node-type-specific outputs.
    Different node types (interior, boundary, free surface) may need different predictions.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        num_node_types: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = 'relu'
    ):
        """
        Args:
            latent_dim: Latent feature dimension
            output_dim: Output feature dimension
            num_node_types: Number of different node types
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function
        """
        super().__init__()

        self.num_node_types = num_node_types

        # Shared decoder
        self.shared_mlp = MLP(
            input_dim=latent_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers - 1,
            activation=activation,
            use_layer_norm=True
        )

        # Type-specific heads
        self.type_heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim)
            for _ in range(num_node_types)
        ])

    def forward(
        self,
        node_latent: torch.Tensor,
        node_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode with node-type-specific outputs.

        Args:
            node_latent: Latent node features [num_nodes, latent_dim]
            node_types: Node type indices [num_nodes] (0, 1, 2, ...)

        Returns:
            Predicted output [num_nodes, output_dim]
        """
        # Shared processing
        shared_features = self.shared_mlp(node_latent)

        # Type-specific predictions
        output = torch.zeros(
            node_latent.size(0),
            self.type_heads[0].out_features,
            device=node_latent.device
        )

        for type_idx in range(self.num_node_types):
            mask = (node_types == type_idx)
            if mask.any():
                output[mask] = self.type_heads[type_idx](shared_features[mask])

        return output


class ResidualDecoder(nn.Module):
    """
    Decoder with explicit residual connection to input features.
    Useful when predictions are small perturbations of current state.
    """

    def __init__(
        self,
        latent_dim: int,
        input_feature_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = 'relu'
    ):
        """
        Args:
            latent_dim: Latent feature dimension
            input_feature_dim: Dimension of original input features
            output_dim: Output feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function
        """
        super().__init__()

        # Combine latent and input features
        self.mlp = MLP(
            input_dim=latent_dim + input_feature_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            use_layer_norm=True
        )

    def forward(
        self,
        node_latent: torch.Tensor,
        input_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode with residual connection.

        Args:
            node_latent: Latent node features [num_nodes, latent_dim]
            input_features: Original input features [num_nodes, input_feature_dim]

        Returns:
            Predicted output [num_nodes, output_dim]
        """
        combined = torch.cat([node_latent, input_features], dim=-1)
        output = self.mlp(combined)
        return output
