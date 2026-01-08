"""
Free surface physics constraints
Implements VOF/Level-set interface tracking and surface tension
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Optional, Tuple


def compute_gradient(
    field: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute gradient of a scalar field on the graph.

    Args:
        field: Scalar field values [num_nodes]
        pos: Node positions [num_nodes, 3]
        edge_index: Edge connectivity [2, num_edges]
        edge_attr: Optional edge features [num_edges, 4]

    Returns:
        Gradient field [num_nodes, 3]
    """
    row, col = edge_index
    num_nodes = field.size(0)

    # Compute field differences along edges
    field_diff = field[col] - field[row]  # [num_edges]

    # Compute relative positions
    if edge_attr is not None:
        rel_pos = edge_attr[:, :3]  # [num_edges, 3]
    else:
        rel_pos = pos[col] - pos[row]  # [num_edges, 3]

    # Distance
    dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [num_edges, 1]
    dist = torch.clamp(dist, min=1e-8)

    # Gradient approximation: ∇φ ≈ Δφ / Δx
    grad_contribution = (field_diff.unsqueeze(-1) / dist) * (rel_pos / dist)  # [num_edges, 3]

    # Aggregate to nodes
    gradient = torch.zeros(num_nodes, 3, device=field.device)
    gradient.index_add_(0, row, grad_contribution)

    # Average by number of neighbors
    degree = torch.zeros(num_nodes, device=field.device)
    degree.index_add_(0, row, torch.ones(edge_index.size(1), device=field.device))
    degree = torch.clamp(degree, min=1.0)

    gradient = gradient / degree.unsqueeze(-1)

    return gradient


def compute_divergence(
    vector_field: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor
) -> torch.Tensor:
    """
    Compute divergence of a vector field on the graph.

    Args:
        vector_field: Vector field values [num_nodes, 3]
        pos: Node positions [num_nodes, 3]
        edge_index: Edge connectivity [2, num_edges]

    Returns:
        Divergence field [num_nodes]
    """
    row, col = edge_index
    num_nodes = vector_field.size(0)

    # Relative positions
    rel_pos = pos[col] - pos[row]  # [num_edges, 3]
    dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [num_edges, 1]
    dist = torch.clamp(dist, min=1e-8)

    # Vector field differences
    field_diff = vector_field[col] - vector_field[row]  # [num_edges, 3]

    # Divergence approximation: ∇·u ≈ (u_j - u_i)·(x_j - x_i) / |x_j - x_i|^2
    div_contribution = (field_diff * rel_pos).sum(dim=-1) / (dist.squeeze(-1) + 1e-8)  # [num_edges]

    # Aggregate to nodes
    divergence = torch.zeros(num_nodes, device=vector_field.device)
    divergence.index_add_(0, row, div_contribution)

    # Average by number of neighbors
    degree = torch.zeros(num_nodes, device=vector_field.device)
    degree.index_add_(0, row, torch.ones(edge_index.size(1), device=vector_field.device))
    degree = torch.clamp(degree, min=1.0)

    divergence = divergence / degree

    return divergence


def compute_curvature(
    level_set: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor
) -> torch.Tensor:
    """
    Compute mean curvature from level-set field.
    κ = ∇·(∇φ / |∇φ|)

    Args:
        level_set: Level-set field [num_nodes]
        pos: Node positions [num_nodes, 3]
        edge_index: Edge connectivity [2, num_edges]

    Returns:
        Curvature field [num_nodes]
    """
    # Compute gradient
    grad_phi = compute_gradient(level_set, pos, edge_index)  # [num_nodes, 3]

    # Normalize gradient
    grad_norm = torch.norm(grad_phi, dim=-1, keepdim=True)  # [num_nodes, 1]
    grad_norm = torch.clamp(grad_norm, min=1e-8)
    normal = grad_phi / grad_norm  # [num_nodes, 3]

    # Compute divergence of normal (curvature)
    curvature = compute_divergence(normal, pos, edge_index)  # [num_nodes]

    return curvature


class FreeSurfaceLoss(nn.Module):
    """
    Free surface physics loss.
    Enforces interface tracking and surface tension effects.
    """

    def __init__(
        self,
        surface_tension: float = 0.07,  # N/m
        density_liquid: float = 1000.0,  # kg/m^3
        density_gas: float = 1.0,  # kg/m^3
        interface_weight: float = 1.0,
        smoothness_weight: float = 0.1
    ):
        """
        Args:
            surface_tension: Surface tension coefficient
            density_liquid: Liquid density
            density_gas: Gas density
            interface_weight: Weight for interface preservation loss
            smoothness_weight: Weight for interface smoothness loss
        """
        super().__init__()

        self.surface_tension = surface_tension
        self.density_liquid = density_liquid
        self.density_gas = density_gas
        self.interface_weight = interface_weight
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data: Data
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute free surface loss.

        Args:
            pred: Predicted state changes [num_nodes, feature_dim]
            target: Target state changes [num_nodes, feature_dim]
            data: PyG Data object with current state

        Returns:
            Total loss and dictionary of individual loss components
        """
        # Extract level-set field (assuming index 8 in node features)
        phi_current = data.x[:, 8]  # [num_nodes]

        # Predicted next level-set
        phi_next = phi_current + pred[:, 5]  # Assuming pred[:, 5] is delta_phi

        # Target next level-set
        phi_target = phi_current + target[:, 5]

        # 1. Interface preservation: level-set should maintain sign distance property
        # |∇φ| should be close to 1 near interface
        grad_phi = compute_gradient(phi_next, data.pos, data.edge_index, data.edge_attr)
        grad_norm = torch.norm(grad_phi, dim=-1)

        # Loss for nodes near interface (|φ| < threshold)
        interface_mask = torch.abs(phi_next) < 0.3
        if interface_mask.any():
            interface_loss = torch.mean((grad_norm[interface_mask] - 1.0) ** 2)
        else:
            interface_loss = torch.tensor(0.0, device=pred.device)

        # 2. Smoothness: penalize high curvature (Laplacian regularization)
        curvature = compute_curvature(phi_next, data.pos, data.edge_index)
        smoothness_loss = torch.mean(curvature ** 2)

        # 3. Mass conservation: level-set volume should be preserved
        # Approximate volume by counting nodes with φ > 0
        volume_current = torch.sum(phi_current > 0).float()
        volume_next = torch.sum(phi_next > 0).float()
        volume_loss = ((volume_next - volume_current) / (volume_current + 1e-8)) ** 2

        # Total loss
        total_loss = (
            self.interface_weight * interface_loss +
            self.smoothness_weight * smoothness_loss +
            0.01 * volume_loss
        )

        loss_dict = {
            'interface_loss': interface_loss.item(),
            'smoothness_loss': smoothness_loss.item(),
            'volume_loss': volume_loss.item()
        }

        return total_loss, loss_dict


class SurfaceTensionForce(nn.Module):
    """
    Compute surface tension force from level-set field.
    F_st = σκn, where σ is surface tension, κ is curvature, n is normal.
    """

    def __init__(
        self,
        surface_tension: float = 0.07,
        density_liquid: float = 1000.0
    ):
        """
        Args:
            surface_tension: Surface tension coefficient [N/m]
            density_liquid: Liquid density [kg/m^3]
        """
        super().__init__()

        self.surface_tension = surface_tension
        self.density_liquid = density_liquid

    def forward(self, data: Data) -> torch.Tensor:
        """
        Compute surface tension force.

        Args:
            data: PyG Data object with level-set field

        Returns:
            Force field [num_nodes, 3]
        """
        phi = data.x[:, 8]  # Level-set field
        pos = data.pos
        edge_index = data.edge_index

        # Compute gradient (normal direction)
        grad_phi = compute_gradient(phi, pos, edge_index)  # [num_nodes, 3]
        grad_norm = torch.norm(grad_phi, dim=-1, keepdim=True)
        grad_norm = torch.clamp(grad_norm, min=1e-8)
        normal = grad_phi / grad_norm  # [num_nodes, 3]

        # Compute curvature
        curvature = compute_curvature(phi, pos, edge_index)  # [num_nodes]

        # Surface tension force: F = σκn
        force = self.surface_tension * curvature.unsqueeze(-1) * normal

        # Apply only near interface
        interface_mask = torch.abs(phi) < 0.3
        force = force * interface_mask.unsqueeze(-1).float()

        return force


def heaviside_smooth(phi: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
    """
    Smooth Heaviside function for level-set.

    Args:
        phi: Level-set field [num_nodes]
        epsilon: Smoothing parameter

    Returns:
        Heaviside values [num_nodes] (0 for gas, 1 for liquid)
    """
    return 0.5 * (1.0 + torch.tanh(phi / epsilon))


def delta_smooth(phi: torch.Tensor, epsilon: float = 0.1) -> torch.Tensor:
    """
    Smooth delta function for level-set interface.

    Args:
        phi: Level-set field [num_nodes]
        epsilon: Smoothing parameter

    Returns:
        Delta values [num_nodes] (peaked at interface)
    """
    return 0.5 / epsilon * (1.0 / torch.cosh(phi / epsilon)) ** 2
