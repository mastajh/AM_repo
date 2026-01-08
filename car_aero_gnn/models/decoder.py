"""
Decoder for MeshGraphNet
Transforms latent node features to physical quantities (velocity, pressure)
"""

import torch
import torch.nn as nn
from .encoder import MLP


class Decoder(nn.Module):
    """
    Decodes latent node features to output predictions

    For steady-state: Predicts velocity (u, v, w) and pressure (p) directly
    For unsteady: Predicts change rate (du/dt, dv/dt, dw/dt, dp/dt)

    Args:
        latent_dim: Latent dimension (128)
        output_dim: Output dimension (4 for velocity + pressure)
        hidden_dim: Hidden layer dimension (128)
        num_layers: Number of MLP layers (2)
    """

    def __init__(self, latent_dim=128, output_dim=4, hidden_dim=128, num_layers=2):
        super().__init__()
        self.decoder = MLP(latent_dim, output_dim, hidden_dim, num_layers)

    def forward(self, x):
        """
        Args:
            x: Latent node features [num_nodes, latent_dim]

        Returns:
            predictions: [num_nodes, output_dim]
                - For steady-state: (u, v, w, p)
                - For unsteady: (du/dt, dv/dt, dw/dt, dp/dt)
        """
        return self.decoder(x)


class ResidualDecoder(nn.Module):
    """
    Decoder with residual prediction
    Predicts delta from current state: y_new = y_current + delta

    Useful for unsteady simulations where changes are small

    Args:
        latent_dim: Latent dimension (128)
        output_dim: Output dimension (4)
        hidden_dim: Hidden layer dimension (128)
        num_layers: Number of MLP layers (2)
    """

    def __init__(self, latent_dim=128, output_dim=4, hidden_dim=128, num_layers=2):
        super().__init__()
        self.decoder = MLP(latent_dim, output_dim, hidden_dim, num_layers)

    def forward(self, x, current_state):
        """
        Args:
            x: Latent node features [num_nodes, latent_dim]
            current_state: Current physical state [num_nodes, output_dim]

        Returns:
            predictions: Updated state [num_nodes, output_dim]
        """
        delta = self.decoder(x)
        return current_state + delta
