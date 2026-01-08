"""
Thermal physics constraints
Implements heat transfer equations and phase change
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Optional, Tuple

from .free_surface import compute_gradient, compute_divergence


class HeatTransferLoss(nn.Module):
    """
    Heat transfer physics loss.
    Enforces heat diffusion equation: ρc_p ∂T/∂t = ∇·(k∇T) + Q
    """

    def __init__(
        self,
        thermal_conductivity: float = 0.6,  # W/(m·K)
        density: float = 1000.0,  # kg/m^3
        specific_heat: float = 4186.0,  # J/(kg·K)
        dt: float = 1e-5,  # time step
        weight: float = 1.0
    ):
        """
        Args:
            thermal_conductivity: Thermal conductivity [W/(m·K)]
            density: Material density [kg/m^3]
            specific_heat: Specific heat capacity [J/(kg·K)]
            dt: Time step [s]
            weight: Loss weight
        """
        super().__init__()

        self.k = thermal_conductivity
        self.rho = density
        self.cp = specific_heat
        self.dt = dt
        self.weight = weight
        self.alpha = self.k / (self.rho * self.cp)  # Thermal diffusivity

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data: Data
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute heat transfer loss.

        Args:
            pred: Predicted state changes [num_nodes, feature_dim]
            target: Target state changes [num_nodes, feature_dim]
            data: PyG Data object

        Returns:
            Loss and dictionary of loss components
        """
        # Extract temperature (assuming index 3 in node features)
        T_current = data.x[:, 3]  # [num_nodes]

        # Predicted temperature change
        dT_pred = pred[:, 0]  # Assuming pred[:, 0] is delta_T

        # Target temperature change
        dT_target = target[:, 0]

        # Compute Laplacian of temperature (∇²T)
        # ∇²T = ∇·(∇T)
        grad_T = compute_gradient(T_current, data.pos, data.edge_index, data.edge_attr)
        laplacian_T = compute_divergence(grad_T, data.pos, data.edge_index)

        # Heat equation: dT/dt = α∇²T
        # Predicted dT should match α∇²T * dt
        physics_dT = self.alpha * laplacian_T * self.dt

        # Loss: prediction should be consistent with physics
        heat_loss = torch.mean((dT_pred - physics_dT) ** 2)

        loss_dict = {
            'heat_transfer_loss': heat_loss.item()
        }

        total_loss = self.weight * heat_loss

        return total_loss, loss_dict


class PhaseChangeLoss(nn.Module):
    """
    Phase change physics loss.
    Enforces Stefan condition at solid-liquid interface.
    """

    def __init__(
        self,
        melting_temp: float = 273.15,  # K (water)
        latent_heat: float = 334000.0,  # J/kg (water)
        weight: float = 1.0
    ):
        """
        Args:
            melting_temp: Melting temperature [K]
            latent_heat: Latent heat of fusion [J/kg]
            weight: Loss weight
        """
        super().__init__()

        self.T_m = melting_temp
        self.L = latent_heat
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data: Data
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute phase change loss.

        Args:
            pred: Predicted state changes
            target: Target state changes
            data: PyG Data object

        Returns:
            Loss and dictionary of loss components
        """
        # Extract temperature and phase
        T_current = data.x[:, 3]  # Temperature
        phase_current = data.x[:, 4]  # Phase (0=solid, 1=liquid, 2=gas)

        # Predicted changes
        dT_pred = pred[:, 0]
        dphase_pred = pred[:, 1]

        # Phase transition constraint: if T ≈ T_m, phase should change
        # Solid-liquid transition
        melting_mask = (torch.abs(T_current - self.T_m) < 10.0) & (phase_current < 1.5)

        if melting_mask.any():
            # If near melting temp, phase should change proportionally to heat flux
            # Simplified: dphase ∝ dT
            phase_loss = torch.mean((dphase_pred[melting_mask] * self.L - dT_pred[melting_mask]) ** 2)
        else:
            phase_loss = torch.tensor(0.0, device=pred.device)

        # Phase bounds: 0 <= phase <= 2
        phase_next = phase_current + dphase_pred
        phase_bound_loss = torch.mean(
            torch.relu(-phase_next) ** 2 + torch.relu(phase_next - 2.0) ** 2
        )

        total_loss = self.weight * (phase_loss + 0.1 * phase_bound_loss)

        loss_dict = {
            'phase_change_loss': phase_loss.item() if isinstance(phase_loss, torch.Tensor) else phase_loss,
            'phase_bound_loss': phase_bound_loss.item()
        }

        return total_loss, loss_dict


class ThermalBoundaryCondition(nn.Module):
    """
    Apply thermal boundary conditions.
    """

    def __init__(
        self,
        ambient_temp: float = 293.15,  # K (room temperature)
        heat_transfer_coeff: float = 10.0,  # W/(m^2·K)
    ):
        """
        Args:
            ambient_temp: Ambient temperature [K]
            heat_transfer_coeff: Heat transfer coefficient [W/(m^2·K)]
        """
        super().__init__()

        self.T_amb = ambient_temp
        self.h = heat_transfer_coeff

    def forward(self, data: Data, node_type: torch.Tensor) -> torch.Tensor:
        """
        Compute heat flux at boundary nodes.

        Args:
            data: PyG Data object
            node_type: Node type indicators [num_nodes]
                      (0=interior, 1=boundary, 2=free_surface)

        Returns:
            Heat flux [num_nodes]
        """
        T = data.x[:, 3]  # Temperature

        # Convective heat flux at boundaries: q = h(T - T_amb)
        heat_flux = torch.zeros_like(T)

        # Apply to boundary and free surface nodes
        boundary_mask = (node_type == 1) | (node_type == 2)
        heat_flux[boundary_mask] = self.h * (T[boundary_mask] - self.T_amb)

        return heat_flux


class LaserHeatSource(nn.Module):
    """
    Gaussian laser heat source for additive manufacturing simulation.
    """

    def __init__(
        self,
        power: float = 100.0,  # W
        radius: float = 0.1,  # mm
        absorption: float = 0.3,  # Absorption coefficient
        efficiency: float = 0.5  # Laser efficiency
    ):
        """
        Args:
            power: Laser power [W]
            radius: Laser beam radius [mm]
            absorption: Material absorption coefficient
            efficiency: Laser-to-heat conversion efficiency
        """
        super().__init__()

        self.power = power
        self.radius = radius
        self.absorption = absorption
        self.efficiency = efficiency

    def forward(
        self,
        pos: torch.Tensor,
        laser_pos: torch.Tensor,
        time: float
    ) -> torch.Tensor:
        """
        Compute volumetric heat source from laser.

        Args:
            pos: Node positions [num_nodes, 3]
            laser_pos: Laser beam center position [3] (x, y, z)
            time: Current time [s]

        Returns:
            Volumetric heat source [num_nodes]
        """
        # Distance from laser center
        dist = torch.norm(pos - laser_pos.unsqueeze(0), dim=-1)  # [num_nodes]

        # Gaussian heat source: Q = Q0 * exp(-r^2 / r0^2)
        Q0 = (2 * self.absorption * self.efficiency * self.power) / (
            torch.pi * self.radius ** 2
        )

        heat_source = Q0 * torch.exp(-(dist ** 2) / (self.radius ** 2))

        return heat_source


class MarangoniForce(nn.Module):
    """
    Marangoni force due to surface tension gradient.
    Caused by temperature-dependent surface tension at free surface.
    """

    def __init__(
        self,
        surface_tension_ref: float = 0.07,  # N/m at reference temp
        temp_ref: float = 293.15,  # K
        dsigma_dT: float = -1.0e-4,  # dσ/dT [N/(m·K)]
    ):
        """
        Args:
            surface_tension_ref: Reference surface tension [N/m]
            temp_ref: Reference temperature [K]
            dsigma_dT: Temperature derivative of surface tension [N/(m·K)]
        """
        super().__init__()

        self.sigma_ref = surface_tension_ref
        self.T_ref = temp_ref
        self.dsigma_dT = dsigma_dT

    def forward(self, data: Data) -> torch.Tensor:
        """
        Compute Marangoni force at free surface.

        Args:
            data: PyG Data object

        Returns:
            Force field [num_nodes, 3]
        """
        T = data.x[:, 3]  # Temperature
        phi = data.x[:, 8]  # Level-set
        pos = data.pos
        edge_index = data.edge_index

        # Temperature gradient
        grad_T = compute_gradient(T, pos, edge_index)  # [num_nodes, 3]

        # Surface tension as function of temperature
        # σ(T) = σ_ref + dσ/dT * (T - T_ref)
        sigma = self.sigma_ref + self.dsigma_dT * (T - self.T_ref)

        # Marangoni stress: τ = ∇_s σ = dσ/dT * ∇_s T
        # Simplified: use full temperature gradient
        force = self.dsigma_dT * grad_T  # [num_nodes, 3]

        # Apply only at free surface
        interface_mask = torch.abs(phi) < 0.3
        force = force * interface_mask.unsqueeze(-1).float()

        return force
