"""
Physics constraints and conservation laws
Implements mass, momentum, and energy conservation losses
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Optional, Tuple, Dict

from .free_surface import compute_divergence, compute_gradient


class MassConservationLoss(nn.Module):
    """
    Mass conservation (continuity equation) loss.
    ∂ρ/∂t + ∇·(ρu) = 0
    """

    def __init__(
        self,
        density_liquid: float = 1000.0,
        density_gas: float = 1.0,
        dt: float = 1e-5,
        weight: float = 1.0
    ):
        """
        Args:
            density_liquid: Liquid density [kg/m^3]
            density_gas: Gas density [kg/m^3]
            dt: Time step [s]
            weight: Loss weight
        """
        super().__init__()

        self.rho_l = density_liquid
        self.rho_g = density_gas
        self.dt = dt
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data: Data
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute mass conservation loss.

        Args:
            pred: Predicted state changes
            target: Target state changes
            data: PyG Data object

        Returns:
            Loss and dictionary of loss components
        """
        # Extract velocity field (assuming indices 5,6,7 are u,v,w)
        velocity = data.x[:, 5:8]  # [num_nodes, 3]

        # Extract phase (to compute density)
        phase = data.x[:, 4]  # [num_nodes]

        # Density from phase (linear interpolation)
        # phase: 0=solid, 1=liquid, 2=gas
        # For simplicity, treat solid as liquid
        rho = torch.where(
            phase < 1.5,
            torch.full_like(phase, self.rho_l),
            torch.full_like(phase, self.rho_g)
        )

        # Compute divergence of momentum (ρu)
        momentum = rho.unsqueeze(-1) * velocity  # [num_nodes, 3]
        div_momentum = compute_divergence(momentum, data.pos, data.edge_index)  # [num_nodes]

        # Continuity equation: ∇·(ρu) ≈ 0 (incompressible)
        mass_loss = torch.mean(div_momentum ** 2)

        loss_dict = {
            'mass_conservation_loss': mass_loss.item()
        }

        total_loss = self.weight * mass_loss

        return total_loss, loss_dict


class MomentumConservationLoss(nn.Module):
    """
    Momentum conservation (Navier-Stokes) loss.
    ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + f
    Simplified for low Reynolds number (Stokes flow).
    """

    def __init__(
        self,
        viscosity: float = 0.001,  # Pa·s
        density: float = 1000.0,  # kg/m^3
        dt: float = 1e-5,
        weight: float = 1.0
    ):
        """
        Args:
            viscosity: Dynamic viscosity [Pa·s]
            density: Density [kg/m^3]
            dt: Time step [s]
            weight: Loss weight
        """
        super().__init__()

        self.mu = viscosity
        self.rho = density
        self.dt = dt
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data: Data,
        external_force: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute momentum conservation loss.

        Args:
            pred: Predicted state changes
            target: Target state changes
            data: PyG Data object
            external_force: External forces [num_nodes, 3] (e.g., gravity, surface tension)

        Returns:
            Loss and dictionary of loss components
        """
        # Extract velocity
        u = data.x[:, 5]
        v = data.x[:, 6]
        w = data.x[:, 7]
        velocity = torch.stack([u, v, w], dim=-1)  # [num_nodes, 3]

        # Predicted velocity change
        du_pred = pred[:, 2:5]  # Assuming indices 2,3,4 are delta_u, delta_v, delta_w

        # Compute viscous term: μ∇²u
        # ∇²u is Laplacian of each velocity component
        laplacian_u = torch.zeros_like(velocity)
        for i in range(3):
            grad_u_i = compute_gradient(velocity[:, i], data.pos, data.edge_index, data.edge_attr)
            laplacian_u[:, i] = compute_divergence(grad_u_i, data.pos, data.edge_index)

        # Momentum equation (simplified, ignoring pressure and convection):
        # ρ du/dt = μ∇²u + f
        viscous_accel = (self.mu / self.rho) * laplacian_u  # [num_nodes, 3]

        if external_force is not None:
            total_accel = viscous_accel + external_force / self.rho
        else:
            total_accel = viscous_accel

        # Predicted du should match physics-based acceleration * dt
        physics_du = total_accel * self.dt

        # Loss
        momentum_loss = torch.mean((du_pred - physics_du) ** 2)

        loss_dict = {
            'momentum_conservation_loss': momentum_loss.item()
        }

        total_loss = self.weight * momentum_loss

        return total_loss, loss_dict


class EnergyConservationLoss(nn.Module):
    """
    Energy conservation loss.
    Ensures total energy is conserved over time.
    """

    def __init__(
        self,
        specific_heat: float = 4186.0,  # J/(kg·K)
        density: float = 1000.0,  # kg/m^3
        weight: float = 1.0
    ):
        """
        Args:
            specific_heat: Specific heat capacity [J/(kg·K)]
            density: Density [kg/m^3]
            weight: Loss weight
        """
        super().__init__()

        self.cp = specific_heat
        self.rho = density
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data: Data,
        heat_source: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute energy conservation loss.

        Args:
            pred: Predicted state changes
            target: Target state changes
            data: PyG Data object
            heat_source: External heat sources [num_nodes]

        Returns:
            Loss and dictionary of loss components
        """
        # Extract temperature and velocity
        T = data.x[:, 3]
        velocity = data.x[:, 5:8]

        # Predicted changes
        dT_pred = pred[:, 0]
        du_pred = pred[:, 2:5]

        # Kinetic energy change
        KE_current = 0.5 * self.rho * torch.sum(velocity ** 2, dim=-1)
        velocity_next = velocity + du_pred
        KE_next = 0.5 * self.rho * torch.sum(velocity_next ** 2, dim=-1)
        dKE = KE_next - KE_current

        # Thermal energy change
        dE_thermal = self.rho * self.cp * dT_pred

        # Total energy change
        dE_total = dKE + dE_thermal

        # If heat source provided, energy change should match input
        if heat_source is not None:
            energy_loss = torch.mean((dE_total - heat_source) ** 2)
        else:
            # Otherwise, total energy should be conserved (sum ≈ 0)
            energy_loss = (torch.sum(dE_total) / data.num_nodes) ** 2

        loss_dict = {
            'energy_conservation_loss': energy_loss.item()
        }

        total_loss = self.weight * energy_loss

        return total_loss, loss_dict


class PhysicsInformedLoss(nn.Module):
    """
    Combined physics-informed loss.
    Integrates all conservation laws and physics constraints.
    """

    def __init__(
        self,
        config: dict
    ):
        """
        Args:
            config: Configuration dictionary with physics parameters
        """
        super().__init__()

        physics_config = config.get('physics', {})
        sim_config = config.get('simulation', {})

        # Initialize sub-losses
        self.mass_conservation = MassConservationLoss(
            density_liquid=sim_config.get('density_liquid', 1000.0),
            density_gas=sim_config.get('density_gas', 1.0),
            dt=config['data'].get('dt', 1e-5),
            weight=physics_config.get('mass_weight', 0.1)
        )

        self.momentum_conservation = MomentumConservationLoss(
            viscosity=sim_config.get('viscosity', 0.001),
            density=sim_config.get('density_liquid', 1000.0),
            dt=config['data'].get('dt', 1e-5),
            weight=0.05
        )

        self.energy_conservation = EnergyConservationLoss(
            specific_heat=sim_config.get('specific_heat', 4186.0),
            density=sim_config.get('density_liquid', 1000.0),
            weight=physics_config.get('energy_weight', 0.01)
        )

        self.use_physics = physics_config.get('use_physics_loss', True)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data: Data,
        external_force: Optional[torch.Tensor] = None,
        heat_source: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total physics-informed loss.

        Args:
            pred: Predicted state changes
            target: Target state changes
            data: PyG Data object
            external_force: External forces
            heat_source: Heat sources

        Returns:
            Total loss and dictionary of all loss components
        """
        if not self.use_physics:
            return torch.tensor(0.0, device=pred.device), {}

        # Compute individual losses
        mass_loss, mass_dict = self.mass_conservation(pred, target, data)
        momentum_loss, momentum_dict = self.momentum_conservation(
            pred, target, data, external_force
        )
        energy_loss, energy_dict = self.energy_conservation(
            pred, target, data, heat_source
        )

        # Total physics loss
        total_loss = mass_loss + momentum_loss + energy_loss

        # Combine all loss components
        loss_dict = {**mass_dict, **momentum_dict, **energy_dict}

        return total_loss, loss_dict


class CurvatureRegularization(nn.Module):
    """
    Regularization to maintain smooth interfaces.
    Penalizes high curvature to prevent numerical instabilities.
    """

    def __init__(self, weight: float = 0.01):
        """
        Args:
            weight: Regularization weight
        """
        super().__init__()
        self.weight = weight

    def forward(self, data: Data) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute curvature regularization loss.

        Args:
            data: PyG Data object

        Returns:
            Loss and dictionary
        """
        from .free_surface import compute_curvature

        phi = data.x[:, 8]  # Level-set field
        curvature = compute_curvature(phi, data.pos, data.edge_index)

        # Penalize high curvature
        curv_loss = torch.mean(curvature ** 2)

        loss_dict = {
            'curvature_regularization': curv_loss.item()
        }

        return self.weight * curv_loss, loss_dict
