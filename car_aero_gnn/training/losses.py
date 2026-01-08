"""
Loss Functions for Aerodynamic Simulation

Includes:
1. Data loss (MSE)
2. Physics-informed losses (continuity equation, momentum equation)
3. Combined loss with weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


def data_loss(pred, target, reduction='mean'):
    """
    Mean Squared Error loss

    Args:
        pred: Predictions [N, D]
        target: Ground truth [N, D]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        loss: Scalar loss value
    """
    return F.mse_loss(pred, target, reduction=reduction)


def velocity_loss(pred_vel, target_vel, reduction='mean'):
    """
    Separate loss for velocity field

    Args:
        pred_vel: Predicted velocity [N, 3]
        target_vel: Target velocity [N, 3]
        reduction: Reduction method

    Returns:
        loss: Velocity loss
    """
    return F.mse_loss(pred_vel, target_vel, reduction=reduction)


def pressure_loss(pred_p, target_p, reduction='mean'):
    """
    Separate loss for pressure field

    Args:
        pred_p: Predicted pressure [N, 1]
        target_p: Target pressure [N, 1]
        reduction: Reduction method

    Returns:
        loss: Pressure loss
    """
    return F.mse_loss(pred_p, target_p, reduction=reduction)


def compute_divergence_graph(velocity, edge_index, positions, node_types=None):
    """
    Compute divergence using graph-based finite differences

    Continuity equation: div(u) = 0

    Args:
        velocity: Velocity field [N, 3]
        edge_index: Graph connectivity [2, E]
        positions: Node positions [N, 3]
        node_types: Node type mask (optional, [N])

    Returns:
        divergence: Divergence at each node [N]
    """
    row, col = edge_index

    # Compute velocity differences
    vel_diff = velocity[col] - velocity[row]

    # Compute position differences
    pos_diff = positions[col] - positions[row]

    # Distance between nodes
    dist = torch.norm(pos_diff, dim=1, keepdim=True) + 1e-8

    # Normalized direction vectors
    directions = pos_diff / dist

    # Velocity gradient along edge direction (central difference)
    vel_grad = (vel_diff * directions).sum(dim=1)

    # Aggregate gradients at each node
    divergence = scatter(vel_grad, row, dim=0, reduce='mean', dim_size=velocity.size(0))

    # Only compute for interior nodes if node_types provided
    if node_types is not None:
        interior_mask = (node_types == 0)
        divergence = divergence * interior_mask.float()

    return divergence


def continuity_loss(pred_vel, data, lambda_cont=1.0):
    """
    Continuity equation loss: div(u) = 0

    Args:
        pred_vel: Predicted velocity [N, 3]
        data: PyG Data object with edge_index, pos, and optionally node_types
        lambda_cont: Weight for continuity loss

    Returns:
        loss: Continuity loss
    """
    # Extract node types if available
    node_types = None
    if hasattr(data, 'x') and data.x.size(1) >= 11:
        # Node types are in last 3 dimensions (one-hot)
        node_type_onehot = data.x[:, -3:]
        node_types = torch.argmax(node_type_onehot, dim=1)

    # Compute divergence
    div = compute_divergence_graph(
        pred_vel,
        data.edge_index,
        data.pos,
        node_types
    )

    # L2 penalty on divergence
    loss = lambda_cont * torch.mean(div ** 2)

    return loss


def compute_laplacian_graph(field, edge_index, positions):
    """
    Compute Laplacian using graph-based finite differences

    ∇²φ approximation using neighboring nodes

    Args:
        field: Scalar or vector field [N, D]
        edge_index: Graph connectivity [2, E]
        positions: Node positions [N, 3]

    Returns:
        laplacian: Laplacian at each node [N, D]
    """
    row, col = edge_index

    # Field differences
    field_diff = field[col] - field[row]

    # Position differences
    pos_diff = positions[col] - positions[row]
    dist_sq = torch.sum(pos_diff ** 2, dim=1, keepdim=True) + 1e-8

    # Weighted by inverse squared distance
    weighted_diff = field_diff / dist_sq

    # Aggregate
    laplacian = scatter(weighted_diff, row, dim=0, reduce='mean', dim_size=field.size(0))

    return laplacian


def momentum_residual(pred_vel, pred_p, data, reynolds=1e6, lambda_mom=1.0):
    """
    Momentum equation residual loss

    Steady-state Navier-Stokes:
    (u·∇)u = -∇p + (1/Re)∇²u

    This is a simplified version - full implementation would need
    proper convective term computation

    Args:
        pred_vel: Predicted velocity [N, 3]
        pred_p: Predicted pressure [N, 1]
        data: PyG Data object
        reynolds: Reynolds number
        lambda_mom: Weight for momentum loss

    Returns:
        loss: Momentum residual loss
    """
    # Compute viscous term: (1/Re)∇²u
    laplacian_u = compute_laplacian_graph(pred_vel, data.edge_index, data.pos)
    viscous_term = laplacian_u / reynolds

    # Compute pressure gradient: -∇p
    # Simplified gradient computation
    row, col = data.edge_index
    p_diff = pred_p[col] - pred_p[row]
    pos_diff = data.pos[col] - data.pos[row]
    dist = torch.norm(pos_diff, dim=1, keepdim=True) + 1e-8

    directions = pos_diff / dist
    p_grad_edge = (p_diff / dist) * directions

    pressure_grad = scatter(
        p_grad_edge,
        row,
        dim=0,
        reduce='mean',
        dim_size=pred_vel.size(0)
    )

    # Simplified momentum residual (ignoring convective term for now)
    residual = viscous_term + pressure_grad

    # L2 penalty on residual
    loss = lambda_mom * torch.mean(residual ** 2)

    return loss


class PhysicsInformedLoss(nn.Module):
    """
    Combined physics-informed loss

    L_total = L_data + λ_cont * L_cont + λ_mom * L_mom

    Args:
        lambda_cont: Weight for continuity loss
        lambda_mom: Weight for momentum loss
        reynolds: Reynolds number
        use_separate_losses: Use separate velocity/pressure losses
    """

    def __init__(
        self,
        lambda_cont=0.1,
        lambda_mom=0.01,
        reynolds=1e6,
        use_separate_losses=True
    ):
        super().__init__()
        self.lambda_cont = lambda_cont
        self.lambda_mom = lambda_mom
        self.reynolds = reynolds
        self.use_separate_losses = use_separate_losses

    def forward(self, pred, target, data):
        """
        Compute total loss

        Args:
            pred: Predictions [N, 4] (u, v, w, p)
            target: Ground truth [N, 4]
            data: PyG Data object

        Returns:
            loss: Total loss
            loss_dict: Dictionary with individual loss components
        """
        # Split predictions
        pred_vel = pred[:, :3]
        pred_p = pred[:, 3:4]

        target_vel = target[:, :3]
        target_p = target[:, 3:4]

        # Data loss
        if self.use_separate_losses:
            loss_vel = velocity_loss(pred_vel, target_vel)
            loss_pres = pressure_loss(pred_p, target_p)
            loss_data = loss_vel + loss_pres
        else:
            loss_data = data_loss(pred, target)

        # Physics losses
        if self.lambda_cont > 0:
            loss_cont = continuity_loss(pred_vel, data, lambda_cont=1.0)
        else:
            loss_cont = torch.tensor(0.0, device=pred.device)

        if self.lambda_mom > 0:
            loss_mom = momentum_residual(
                pred_vel, pred_p, data,
                reynolds=self.reynolds,
                lambda_mom=1.0
            )
        else:
            loss_mom = torch.tensor(0.0, device=pred.device)

        # Total loss
        total_loss = (
            loss_data +
            self.lambda_cont * loss_cont +
            self.lambda_mom * loss_mom
        )

        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'data': loss_data.item(),
            'continuity': loss_cont.item(),
            'momentum': loss_mom.item()
        }

        if self.use_separate_losses:
            loss_dict['velocity'] = loss_vel.item()
            loss_dict['pressure'] = loss_pres.item()

        return total_loss, loss_dict


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss with optional masking

    Useful for focusing on specific regions (e.g., wake region)

    Args:
        weight: Weight tensor [N] or callable that computes weights
    """

    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target, data=None):
        """
        Compute weighted MSE

        Args:
            pred: Predictions
            target: Ground truth
            data: Optional data for weight computation

        Returns:
            loss: Weighted MSE loss
        """
        mse = (pred - target) ** 2

        if self.weight is not None:
            if callable(self.weight):
                weight = self.weight(data)
            else:
                weight = self.weight

            mse = mse * weight.unsqueeze(-1)

        return mse.mean()


def create_loss_function(config):
    """
    Factory function to create loss from config

    Args:
        config: Configuration dict

    Returns:
        loss_fn: Loss function
    """
    loss_type = config.get('loss_type', 'physics_informed')

    if loss_type == 'physics_informed':
        loss_fn = PhysicsInformedLoss(
            lambda_cont=config.get('lambda_cont', 0.1),
            lambda_mom=config.get('lambda_mom', 0.01),
            reynolds=config.get('reynolds', 1e6),
            use_separate_losses=config.get('use_separate_losses', True)
        )
    elif loss_type == 'mse':
        loss_fn = lambda pred, target, data: (
            data_loss(pred, target),
            {'total': data_loss(pred, target).item()}
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss_fn


if __name__ == '__main__':
    # Test loss functions
    print("Testing loss functions...")

    # Create dummy data
    from torch_geometric.data import Data

    num_nodes = 100
    num_edges = 500

    dummy_data = Data(
        x=torch.randn(num_nodes, 11),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.randn(num_edges, 4),
        pos=torch.randn(num_nodes, 3)
    )

    pred = torch.randn(num_nodes, 4)
    target = torch.randn(num_nodes, 4)

    # Test physics-informed loss
    loss_fn = PhysicsInformedLoss()
    total_loss, loss_dict = loss_fn(pred, target, dummy_data)

    print(f"Total loss: {total_loss.item():.6f}")
    print(f"Loss components: {loss_dict}")

    print("Loss function test passed!")
