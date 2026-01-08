"""
GNN-based Melt Pool and Free Surface Simulation

Physics-informed Graph Neural Network for predicting melt pool dynamics
and free surface behavior in metal additive manufacturing.
"""

__version__ = '1.0.0'
__author__ = 'GNN Meltpool Team'

from . import models
from . import physics
from . import training
from . import inference
from . import data

__all__ = [
    'models',
    'physics',
    'training',
    'inference',
    'data',
]
