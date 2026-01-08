"""
Visualization utilities for simulation results
Uses PyVista for 3D visualization and matplotlib for plots
"""

import torch
import numpy as np
from torch_geometric.data import Data
from pathlib import Path
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def visualize_state_3d(
    data: Data,
    field: str = 'temperature',
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Visualize a single state in 3D using PyVista.

    Args:
        data: PyG Data object
        field: Field to visualize ('temperature', 'phase', 'velocity', 'level_set')
        save_path: Path to save image
        show: Whether to show interactive plot
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed. Install with: pip install pyvista")
        return

    # Extract positions and field
    pos = data.pos.cpu().numpy()

    if field == 'temperature':
        values = data.x[:, 3].cpu().numpy()
        title = 'Temperature (K)'
    elif field == 'phase':
        values = data.x[:, 4].cpu().numpy()
        title = 'Phase (0=solid, 1=liquid, 2=gas)'
    elif field == 'velocity':
        vel = data.x[:, 5:8].cpu().numpy()
        values = np.linalg.norm(vel, axis=-1)
        title = 'Velocity magnitude (m/s)'
    elif field == 'level_set':
        values = data.x[:, 8].cpu().numpy()
        title = 'Level-set field'
    else:
        raise ValueError(f"Unknown field: {field}")

    # Create point cloud
    point_cloud = pv.PolyData(pos)
    point_cloud[field] = values

    # Plot
    plotter = pv.Plotter(off_screen=not show)
    plotter.add_mesh(
        point_cloud,
        scalars=field,
        point_size=5,
        render_points_as_spheres=True,
        cmap='coolwarm'
    )
    plotter.add_title(title)
    plotter.add_axes()

    if save_path:
        plotter.screenshot(save_path)

    if show:
        plotter.show()
    else:
        plotter.close()


def visualize_trajectory_3d(
    trajectory: List[Data],
    field: str = 'temperature',
    output_dir: str = 'outputs/frames',
    fps: int = 30
):
    """
    Visualize trajectory as sequence of 3D images.

    Args:
        trajectory: List of states
        field: Field to visualize
        output_dir: Directory to save frames
        fps: Frames per second for video
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"Rendering {len(trajectory)} frames...")

    for i, data in enumerate(trajectory):
        save_path = output_path / f'frame_{i:04d}.png'
        visualize_state_3d(data, field=field, save_path=str(save_path), show=False)

        if i % 10 == 0:
            print(f"  Rendered frame {i}/{len(trajectory)}")

    print(f"Frames saved to {output_path}")
    print(f"Create video with: ffmpeg -framerate {fps} -i {output_path}/frame_%04d.png -c:v libx264 output.mp4")


def plot_field_history(
    trajectory: List[Data],
    field: str = 'temperature',
    node_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Plot time history of a field at a specific node.

    Args:
        trajectory: List of states
        field: Field to plot
        node_idx: Node index to track
        save_path: Path to save plot
    """
    # Extract field values over time
    if field == 'temperature':
        values = [data.x[node_idx, 3].item() for data in trajectory]
        ylabel = 'Temperature (K)'
    elif field == 'phase':
        values = [data.x[node_idx, 4].item() for data in trajectory]
        ylabel = 'Phase'
    elif field == 'velocity':
        values = [torch.norm(data.x[node_idx, 5:8]).item() for data in trajectory]
        ylabel = 'Velocity magnitude (m/s)'
    elif field == 'level_set':
        values = [data.x[node_idx, 8].item() for data in trajectory]
        ylabel = 'Level-set'
    else:
        raise ValueError(f"Unknown field: {field}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(values)
    plt.xlabel('Time step')
    plt.ylabel(ylabel)
    plt.title(f'{field.capitalize()} history at node {node_idx}')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_average_fields(
    trajectory: List[Data],
    save_path: Optional[str] = None
):
    """
    Plot average field values over time.

    Args:
        trajectory: List of states
        save_path: Path to save plot
    """
    # Extract average values
    avg_temp = [data.x[:, 3].mean().item() for data in trajectory]
    avg_phase = [data.x[:, 4].mean().item() for data in trajectory]
    avg_vel = [torch.norm(data.x[:, 5:8], dim=-1).mean().item() for data in trajectory]

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    axes[0].plot(avg_temp)
    axes[0].set_ylabel('Temperature (K)')
    axes[0].set_title('Average Temperature')
    axes[0].grid(True)

    axes[1].plot(avg_phase)
    axes[1].set_ylabel('Phase')
    axes[1].set_title('Average Phase')
    axes[1].grid(True)

    axes[2].plot(avg_vel)
    axes[2].set_ylabel('Velocity (m/s)')
    axes[2].set_xlabel('Time step')
    axes[2].set_title('Average Velocity Magnitude')
    axes[2].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_interface_evolution(
    trajectory: List[Data],
    save_path: Optional[str] = None
):
    """
    Plot evolution of free surface interface.

    Args:
        trajectory: List of states
        save_path: Path to save plot
    """
    # Track interface properties
    interface_volume = []
    interface_area = []

    for data in trajectory:
        phi = data.x[:, 8]

        # Volume (number of nodes with phi > 0)
        volume = (phi > 0).sum().item()
        interface_volume.append(volume)

        # Interface area (nodes near phi = 0)
        area = (torch.abs(phi) < 0.1).sum().item()
        interface_area.append(area)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(interface_volume)
    axes[0].set_ylabel('Volume (# nodes)')
    axes[0].set_title('Liquid Volume')
    axes[0].grid(True)

    axes[1].plot(interface_area)
    axes[1].set_ylabel('Interface area (# nodes)')
    axes[1].set_xlabel('Time step')
    axes[1].set_title('Interface Area')
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def create_comparison_plot(
    pred_trajectory: List[Data],
    gt_trajectory: List[Data],
    field: str = 'temperature',
    save_path: Optional[str] = None
):
    """
    Create comparison plot between prediction and ground truth.

    Args:
        pred_trajectory: Predicted trajectory
        gt_trajectory: Ground truth trajectory
        field: Field to compare
        save_path: Path to save plot
    """
    # Extract field values
    if field == 'temperature':
        idx = 3
    elif field == 'phase':
        idx = 4
    elif field == 'level_set':
        idx = 8
    else:
        raise ValueError(f"Unknown field: {field}")

    # Compute average values
    pred_avg = [data.x[:, idx].mean().item() for data in pred_trajectory]
    gt_avg = [data.x[:, idx].mean().item() for data in gt_trajectory[:len(pred_trajectory)]]

    # Compute error
    error = [abs(p - g) for p, g in zip(pred_avg, gt_avg)]

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(gt_avg, label='Ground Truth', linewidth=2)
    axes[0].plot(pred_avg, label='Prediction', linewidth=2, linestyle='--')
    axes[0].set_ylabel(f'Average {field}')
    axes[0].set_title(f'{field.capitalize()} Comparison')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(error, color='red')
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_xlabel('Time step')
    axes[1].set_title('Prediction Error')
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    plt.show()


def visualize_slice_2d(
    data: Data,
    field: str = 'temperature',
    slice_axis: str = 'z',
    slice_value: float = 0.25,
    save_path: Optional[str] = None
):
    """
    Visualize 2D slice of 3D data.

    Args:
        data: PyG Data object
        field: Field to visualize
        slice_axis: Axis to slice ('x', 'y', or 'z')
        slice_value: Position of slice
        save_path: Path to save plot
    """
    pos = data.pos.cpu().numpy()

    # Extract field
    if field == 'temperature':
        values = data.x[:, 3].cpu().numpy()
        title = 'Temperature (K)'
    elif field == 'phase':
        values = data.x[:, 4].cpu().numpy()
        title = 'Phase'
    elif field == 'level_set':
        values = data.x[:, 8].cpu().numpy()
        title = 'Level-set'
    else:
        raise ValueError(f"Unknown field: {field}")

    # Select slice
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[slice_axis]
    mask = np.abs(pos[:, axis_idx] - slice_value) < 0.05

    slice_pos = pos[mask]
    slice_values = values[mask]

    # Determine plot axes
    if slice_axis == 'z':
        x, y = slice_pos[:, 0], slice_pos[:, 1]
        xlabel, ylabel = 'X', 'Y'
    elif slice_axis == 'y':
        x, y = slice_pos[:, 0], slice_pos[:, 2]
        xlabel, ylabel = 'X', 'Z'
    else:  # x
        x, y = slice_pos[:, 1], slice_pos[:, 2]
        xlabel, ylabel = 'Y', 'Z'

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x, y, c=slice_values, cmap='coolwarm', s=20)
    plt.colorbar(scatter, label=title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{title} - {slice_axis}={slice_value}')
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved slice plot to {save_path}")

    plt.show()
