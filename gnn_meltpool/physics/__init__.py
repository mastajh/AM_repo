"""
Physics module for GNN-based melt pool simulation.
"""

from .free_surface import (
    FreeSurfaceLoss,
    SurfaceTensionForce,
    compute_gradient,
    compute_divergence,
    compute_curvature,
    heaviside_smooth,
    delta_smooth
)

from .thermal import (
    HeatTransferLoss,
    PhaseChangeLoss,
    ThermalBoundaryCondition,
    LaserHeatSource,
    MarangoniForce
)

from .constraints import (
    MassConservationLoss,
    MomentumConservationLoss,
    EnergyConservationLoss,
    PhysicsInformedLoss,
    CurvatureRegularization
)

__all__ = [
    'FreeSurfaceLoss',
    'SurfaceTensionForce',
    'compute_gradient',
    'compute_divergence',
    'compute_curvature',
    'heaviside_smooth',
    'delta_smooth',
    'HeatTransferLoss',
    'PhaseChangeLoss',
    'ThermalBoundaryCondition',
    'LaserHeatSource',
    'MarangoniForce',
    'MassConservationLoss',
    'MomentumConservationLoss',
    'EnergyConservationLoss',
    'PhysicsInformedLoss',
    'CurvatureRegularization',
]
