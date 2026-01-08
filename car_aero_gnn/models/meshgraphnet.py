"""
MeshGraphNet: Graph Neural Network for Mesh-based Physical Simulation

Implementation of "Learning Mesh-Based Simulation with Graph Networks"
(Pfaff et al., ICML 2021)

Architecture: Encode-Process-Decode
1. Encode: Map input features to latent space
2. Process: Message passing on graph (15 layers)
3. Decode: Map latent features to predictions
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from .encoder import NodeEncoder, EdgeEncoder
from .processor import GraphProcessor
from .decoder import Decoder, ResidualDecoder


class MeshGraphNet(nn.Module):
    """
    Complete MeshGraphNet model for aerodynamic simulation

    Args:
        node_dim: Input node feature dimension (11)
            - Position: x, y, z (3)
            - Velocity: u, v, w (3)
            - Pressure: p (1)
            - Wall distance: d_wall (1)
            - Node type: one-hot (3)
        edge_dim: Input edge feature dimension (4)
            - Relative position: dx, dy, dz (3)
            - Distance: ||r|| (1)
        output_dim: Output dimension (4: u, v, w, p)
        latent_dim: Latent space dimension (128)
        num_message_passing_layers: Number of message passing steps (15)
        num_mlp_layers: Number of layers in each MLP (2)
        use_residual: Whether to predict residuals (True for unsteady)
    """

    def __init__(
        self,
        node_dim=11,
        edge_dim=4,
        output_dim=4,
        latent_dim=128,
        num_message_passing_layers=15,
        num_mlp_layers=2,
        use_residual=False
    ):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.use_residual = use_residual

        # Encoder: Input features -> Latent space
        self.node_encoder = NodeEncoder(
            node_dim=node_dim,
            latent_dim=latent_dim,
            hidden_dim=latent_dim,
            num_layers=num_mlp_layers
        )

        self.edge_encoder = EdgeEncoder(
            edge_dim=edge_dim,
            latent_dim=latent_dim,
            hidden_dim=latent_dim,
            num_layers=num_mlp_layers
        )

        # Processor: Message passing
        self.processor = GraphProcessor(
            latent_dim=latent_dim,
            num_layers=num_message_passing_layers,
            hidden_dim=latent_dim,
            num_mlp_layers=num_mlp_layers
        )

        # Decoder: Latent space -> Output predictions
        if use_residual:
            self.decoder = ResidualDecoder(
                latent_dim=latent_dim,
                output_dim=output_dim,
                hidden_dim=latent_dim,
                num_layers=num_mlp_layers
            )
        else:
            self.decoder = Decoder(
                latent_dim=latent_dim,
                output_dim=output_dim,
                hidden_dim=latent_dim,
                num_layers=num_mlp_layers
            )

    def forward(self, data, return_latent=False):
        """
        Forward pass

        Args:
            data: PyG Data object with attributes:
                - x: Node features [num_nodes, node_dim]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_dim]
            return_latent: Whether to return latent representations

        Returns:
            predictions: [num_nodes, output_dim]
                - For steady-state: (u, v, w, p)
                - For unsteady: (du/dt, dv/dt, dw/dt, dp/dt) or (u, v, w, p)
            latent (optional): Latent node features [num_nodes, latent_dim]
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Encode
        x_latent = self.node_encoder(x)
        edge_attr_latent = self.edge_encoder(edge_attr)

        # Process
        x_latent, edge_attr_latent = self.processor(
            x_latent, edge_index, edge_attr_latent
        )

        # Decode
        if self.use_residual and hasattr(data, 'current_state'):
            predictions = self.decoder(x_latent, data.current_state)
        else:
            predictions = self.decoder(x_latent)

        if return_latent:
            return predictions, x_latent
        return predictions

    def predict_steady_state(self, data):
        """
        Predict steady-state flow field

        Args:
            data: Graph with geometry and boundary conditions

        Returns:
            velocity: [num_nodes, 3] velocity field (u, v, w)
            pressure: [num_nodes, 1] pressure field (p)
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(data)

        velocity = predictions[:, :3]
        pressure = predictions[:, 3:4]

        return velocity, pressure

    def predict_unsteady_step(self, data, dt=0.001):
        """
        Predict next time step for unsteady simulation

        Args:
            data: Current state graph
            dt: Time step size

        Returns:
            Updated data with new velocity and pressure
        """
        self.eval()
        with torch.no_grad():
            if self.use_residual:
                # Predict state directly
                predictions = self.forward(data)
            else:
                # Predict time derivative
                derivatives = self.forward(data)
                # Euler integration
                current_state = data.x[:, 3:7]  # Current (u, v, w, p)
                predictions = current_state + derivatives * dt

        # Update data
        updated_data = data.clone()
        updated_data.x[:, 3:7] = predictions

        return updated_data

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self):
        """Print model summary"""
        print("=" * 70)
        print("MeshGraphNet Model Summary")
        print("=" * 70)
        print(f"Node input dimension:        {self.node_dim}")
        print(f"Edge input dimension:        {self.edge_dim}")
        print(f"Output dimension:            {self.output_dim}")
        print(f"Latent dimension:            {self.latent_dim}")
        print(f"Message passing layers:      {self.processor.num_layers}")
        print(f"Use residual decoder:        {self.use_residual}")
        print(f"Total parameters:            {self.count_parameters():,}")
        print("=" * 70)


def create_meshgraphnet(config):
    """
    Factory function to create MeshGraphNet from config

    Args:
        config: Configuration dictionary

    Returns:
        model: MeshGraphNet instance
    """
    model = MeshGraphNet(
        node_dim=config.get('node_dim', 11),
        edge_dim=config.get('edge_dim', 4),
        output_dim=config.get('output_dim', 4),
        latent_dim=config.get('latent_dim', 128),
        num_message_passing_layers=config.get('num_layers', 15),
        num_mlp_layers=config.get('num_mlp_layers', 2),
        use_residual=config.get('use_residual', False)
    )

    return model


if __name__ == '__main__':
    # Test model creation
    model = MeshGraphNet()
    model.get_model_summary()

    # Test forward pass with dummy data
    num_nodes = 1000
    num_edges = 5000

    dummy_data = Data(
        x=torch.randn(num_nodes, 11),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.randn(num_edges, 4)
    )

    output = model(dummy_data)
    print(f"\nTest forward pass:")
    print(f"Input nodes: {num_nodes}, edges: {num_edges}")
    print(f"Output shape: {output.shape}")
    print("Model test passed!")
