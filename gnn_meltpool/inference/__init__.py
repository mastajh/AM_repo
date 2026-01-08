"""
Inference module for GNN-based melt pool simulation.
"""

from .rollout import (
    Rollout,
    RolloutEvaluator,
    load_checkpoint,
    run_rollout
)

from .visualize import (
    visualize_state_3d,
    visualize_trajectory_3d,
    plot_field_history,
    plot_average_fields,
    plot_interface_evolution,
    create_comparison_plot,
    visualize_slice_2d
)

__all__ = [
    'Rollout',
    'RolloutEvaluator',
    'load_checkpoint',
    'run_rollout',
    'visualize_state_3d',
    'visualize_trajectory_3d',
    'plot_field_history',
    'plot_average_fields',
    'plot_interface_evolution',
    'create_comparison_plot',
    'visualize_slice_2d',
]
