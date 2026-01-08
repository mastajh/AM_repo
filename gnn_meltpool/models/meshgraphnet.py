"""
MeshGraphNet: Complete model integrating encoder, processor, and decoder
Physics-informed graph neural network for melt pool simulation
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Optional, Tuple

from .encoder import NodeEncoder, EdgeEncoder, build_edge_features
from .processor import Processor
from .decoder import Decoder


class MeshGraphNet(nn.Module):
    """
    MeshGraphNet model for predicting dynamics of melt pool and free surface.

    Architecture:
        1. Encoder: Maps raw node/edge features to latent space
        2. Processor: Message passing to propagate information
        3. Decoder: Maps latent features to output predictions

    Reference:
        Pfaff et al. "Learning Mesh-Based Simulation with Graph Networks" (2021)
    """

    def __init__(
        self,
        node_feature_dim: int = 10,
        edge_feature_dim: int = 4,
        output_dim: int = 7,  # [delta_T, delta_phase, delta_u, delta_v, delta_w, delta_phi, delta_pos]
        latent_dim: int = 128,
        hidden_dim: int = 128,
        num_message_passing: int = 15,
        num_mlp_layers: int = 2,
        activation: str = 'relu',
        aggr: str = 'sum',
        predict_delta: bool = True
    ):
        """
        Args:
            node_feature_dim: Input node feature dimension
                             [x, y, z, T, phase, u, v, w, phi, type]
            edge_feature_dim: Input edge feature dimension
                             [dx, dy, dz, distance]
            output_dim: Output prediction dimension
                       [delta_T, delta_phase, delta_u, delta_v, delta_w, delta_phi]
            latent_dim: Latent representation dimension
            hidden_dim: Hidden layer dimension for MLPs
            num_message_passing: Number of message passing layers
            num_mlp_layers: Number of layers in MLPs
            activation: Activation function ('relu', 'elu', 'gelu')
            aggr: Message aggregation function ('sum', 'mean', 'max')
            predict_delta: If True, predict state changes; else predict absolute state
        """
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.predict_delta = predict_delta

        # Encoder
        self.node_encoder = NodeEncoder(
            node_feature_dim=node_feature_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            activation=activation
        )

        self.edge_encoder = EdgeEncoder(
            edge_feature_dim=edge_feature_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            activation=activation
        )

        # Processor
        self.processor = Processor(
            latent_dim=latent_dim,
            num_layers=num_message_passing,
            hidden_dim=hidden_dim,
            mlp_layers=num_mlp_layers,
            activation=activation,
            aggr=aggr
        )

        # Decoder
        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mlp_layers,
            activation=activation,
            predict_delta=predict_delta
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            data: PyG Data object with attributes:
                - x: Node features [num_nodes, node_feature_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_feature_dim]

        Returns:
            Predictions [num_nodes, output_dim]
            If predict_delta=True, returns state changes
            If predict_delta=False, returns absolute state
        """
        # Extract data
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Encode
        node_latent = self.node_encoder(x)
        edge_latent = self.edge_encoder(edge_attr)

        # Process (message passing)
        node_latent, edge_latent = self.processor(node_latent, edge_latent, edge_index)

        # Decode
        output = self.decoder(node_latent)

        return output

    def predict_next_state(
        self,
        data: Data,
        dt: float = 1e-5,
        apply_constraints: bool = True
    ) -> Data:
        """
        Predict next state given current state.

        Args:
            data: Current state as PyG Data object
            dt: Time step (used for scaling predictions)
            apply_constraints: Whether to apply physical constraints

        Returns:
            Next state as PyG Data object
        """
        # Get predictions (state changes)
        delta = self.forward(data)

        # Update state
        next_data = data.clone()

        if self.predict_delta:
            # Update dynamic features: [T, phase, u, v, w, phi]
            # Assuming node features are [x, y, z, T, phase, u, v, w, phi, type]
            next_data.x[:, 3:9] = data.x[:, 3:9] + delta[:, :6]
        else:
            # Direct prediction of next state
            next_data.x[:, 3:9] = delta[:, :6]

        # Apply physical constraints if requested
        if apply_constraints:
            next_data = self._apply_constraints(next_data)

        return next_data

    def _apply_constraints(self, data: Data) -> Data:
        """
        Apply physical constraints to predictions.

        Args:
            data: PyG Data object

        Returns:
            Data with constraints applied
        """
        # Extract features
        x = data.x.clone()

        # Temperature constraint (T >= 0)
        x[:, 3] = torch.clamp(x[:, 3], min=0.0)

        # Phase constraint (0 <= phase <= 2)
        x[:, 4] = torch.clamp(x[:, 4], min=0.0, max=2.0)

        # Level-set normalization (keep bounded)
        x[:, 8] = torch.clamp(x[:, 8], min=-1.0, max=1.0)

        data.x = x
        return data

    def rollout(
        self,
        initial_data: Data,
        num_steps: int,
        dt: float = 1e-5,
        apply_constraints: bool = True
    ) -> list[Data]:
        """
        Perform autoregressive rollout prediction.

        Args:
            initial_data: Initial state
            num_steps: Number of time steps to predict
            dt: Time step
            apply_constraints: Whether to apply physical constraints

        Returns:
            List of predicted states over time
        """
        trajectory = [initial_data.clone()]
        current_data = initial_data.clone()

        for step in range(num_steps):
            # Predict next state
            current_data = self.predict_next_state(
                current_data,
                dt=dt,
                apply_constraints=apply_constraints
            )

            # Store
            trajectory.append(current_data.clone())

        return trajectory

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size(self) -> dict:
        """Get model size information."""
        total_params = self.count_parameters()
        encoder_params = (
            sum(p.numel() for p in self.node_encoder.parameters()) +
            sum(p.numel() for p in self.edge_encoder.parameters())
        )
        processor_params = sum(p.numel() for p in self.processor.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        return {
            'total': total_params,
            'encoder': encoder_params,
            'processor': processor_params,
            'decoder': decoder_params,
            'total_mb': total_params * 4 / (1024 ** 2)  # Assuming float32
        }


def build_meshgraphnet(config: dict) -> MeshGraphNet:
    """
    Build MeshGraphNet model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        MeshGraphNet model
    """
    model_config = config.get('model', {})

    model = MeshGraphNet(
        node_feature_dim=model_config.get('node_features', 10),
        edge_feature_dim=model_config.get('edge_features', 4),
        output_dim=6,  # [delta_T, delta_phase, delta_u, delta_v, delta_w, delta_phi]
        latent_dim=model_config.get('latent_dim', 128),
        hidden_dim=model_config.get('hidden_dim', 128),
        num_message_passing=model_config.get('num_message_passing', 15),
        num_mlp_layers=model_config.get('hidden_layers', 2),
        activation=model_config.get('activation', 'relu'),
        predict_delta=True
    )

    return model
