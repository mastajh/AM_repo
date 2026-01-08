"""
Loss functions for training
Combines data loss with physics-informed losses
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Dict, Optional, Tuple

from physics import (
    PhysicsInformedLoss,
    FreeSurfaceLoss,
    HeatTransferLoss,
    PhaseChangeLoss,
    CurvatureRegularization
)


class MeltPoolLoss(nn.Module):
    """
    Combined loss function for melt pool simulation.

    Combines:
    1. Data loss (MSE between prediction and target)
    2. Physics-informed losses (conservation laws)
    3. Free surface losses (interface tracking)
    4. Thermal losses (heat transfer)
    5. Regularization losses
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()

        self.config = config
        physics_config = config.get('physics', {})
        sim_config = config.get('simulation', {})

        # Data loss weight
        self.data_weight = 1.0

        # Physics-informed losses
        self.use_physics = physics_config.get('use_physics_loss', True)

        if self.use_physics:
            self.physics_loss = PhysicsInformedLoss(config)

            self.surface_loss = FreeSurfaceLoss(
                surface_tension=sim_config.get('surface_tension', 0.07),
                density_liquid=sim_config.get('density_liquid', 1000.0),
                density_gas=sim_config.get('density_gas', 1.0),
                interface_weight=physics_config.get('surface_weight', 0.1),
                smoothness_weight=0.05
            )

            self.heat_loss = HeatTransferLoss(
                thermal_conductivity=sim_config.get('thermal_conductivity', 0.6),
                density=sim_config.get('density_liquid', 1000.0),
                specific_heat=sim_config.get('specific_heat', 4186.0),
                dt=config['data'].get('dt', 1e-5),
                weight=0.05
            )

            self.phase_loss = PhaseChangeLoss(
                melting_temp=273.15,
                latent_heat=334000.0,
                weight=0.05
            )

            self.curvature_reg = CurvatureRegularization(
                weight=physics_config.get('curvature_weight', 0.01)
            )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data: Data,
        external_force: Optional[torch.Tensor] = None,
        heat_source: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.

        Args:
            pred: Predicted state changes [num_nodes, feature_dim]
            target: Target state changes [num_nodes, feature_dim]
            data: PyG Data object with current state
            external_force: External forces [num_nodes, 3]
            heat_source: Heat sources [num_nodes]

        Returns:
            Total loss and dictionary of loss components
        """
        loss_dict = {}

        # 1. Data loss (MSE)
        data_loss = nn.functional.mse_loss(pred, target)
        loss_dict['data_loss'] = data_loss.item()

        total_loss = self.data_weight * data_loss

        # 2. Physics losses
        if self.use_physics:
            # Conservation laws
            physics_loss, physics_dict = self.physics_loss(
                pred, target, data, external_force, heat_source
            )
            total_loss = total_loss + physics_loss
            loss_dict.update(physics_dict)

            # Free surface
            surface_loss, surface_dict = self.surface_loss(pred, target, data)
            total_loss = total_loss + surface_loss
            loss_dict.update(surface_dict)

            # Heat transfer
            heat_loss, heat_dict = self.heat_loss(pred, target, data)
            total_loss = total_loss + heat_loss
            loss_dict.update(heat_dict)

            # Phase change
            phase_loss, phase_dict = self.phase_loss(pred, target, data)
            total_loss = total_loss + phase_loss
            loss_dict.update(phase_dict)

            # Curvature regularization
            curv_loss, curv_dict = self.curvature_reg(data)
            total_loss = total_loss + curv_loss
            loss_dict.update(curv_dict)

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


class RolloutLoss(nn.Module):
    """
    Loss for multi-step rollout predictions.
    Penalizes error accumulation over time.
    """

    def __init__(
        self,
        num_steps: int = 10,
        decay: float = 0.9
    ):
        """
        Args:
            num_steps: Number of rollout steps
            decay: Decay factor for later steps
        """
        super().__init__()

        self.num_steps = num_steps
        self.decay = decay

    def forward(
        self,
        model: nn.Module,
        initial_data: Data,
        target_trajectory: list
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute rollout loss.

        Args:
            model: MeshGraphNet model
            initial_data: Initial state
            target_trajectory: List of target states

        Returns:
            Loss and dictionary
        """
        total_loss = 0.0
        current_data = initial_data.clone()

        for step in range(min(self.num_steps, len(target_trajectory))):
            # Predict next state
            delta_pred = model(current_data)

            # Update state
            current_data.x[:, 3:9] = current_data.x[:, 3:9] + delta_pred

            # Target
            target_state = target_trajectory[step].x[:, 3:9]

            # Loss for this step
            step_loss = nn.functional.mse_loss(current_data.x[:, 3:9], target_state)

            # Add to total with decay
            weight = self.decay ** step
            total_loss = total_loss + weight * step_loss

        loss_dict = {'rollout_loss': total_loss.item()}

        return total_loss, loss_dict


class AdaptiveWeightLoss(nn.Module):
    """
    Loss with adaptive weights that balance different components.
    Uses gradient-based balancing.
    """

    def __init__(self, num_losses: int = 3, alpha: float = 0.1):
        """
        Args:
            num_losses: Number of loss components
            alpha: Learning rate for weight updates
        """
        super().__init__()

        self.num_losses = num_losses
        self.alpha = alpha

        # Initialize weights
        self.register_buffer('weights', torch.ones(num_losses))
        self.register_buffer('initial_losses', torch.ones(num_losses))
        self.initialized = False

    def forward(
        self,
        losses: list
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted loss.

        Args:
            losses: List of loss tensors

        Returns:
            Total loss and dictionary
        """
        losses_tensor = torch.stack(losses)

        # Initialize on first call
        if not self.initialized:
            self.initial_losses = losses_tensor.detach()
            self.initialized = True

        # Normalize losses by initial values
        normalized_losses = losses_tensor / (self.initial_losses + 1e-8)

        # Update weights (gradient-based balancing)
        with torch.no_grad():
            # Weights inversely proportional to loss magnitude
            new_weights = 1.0 / (normalized_losses.detach() + 1e-8)
            new_weights = new_weights / new_weights.sum()

            # Exponential moving average
            self.weights = (1 - self.alpha) * self.weights + self.alpha * new_weights

        # Weighted sum
        total_loss = (self.weights * losses_tensor).sum()

        loss_dict = {
            f'loss_{i}': losses[i].item() for i in range(len(losses))
        }
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


def build_loss(config: dict) -> nn.Module:
    """
    Build loss function from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Loss module
    """
    return MeltPoolLoss(config)
