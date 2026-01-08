"""
Unsteady Flow Rollout

Autoregressive time stepping for transient flow simulation
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional


class UnsteadyRollout:
    """
    Autoregressive rollout for unsteady flow simulation

    Predicts flow evolution by iteratively applying the model:
    state(t+1) = state(t) + dt * model(state(t))

    Args:
        model_path: Path to trained model
        dt: Time step size
        device: Device for computation
    """

    def __init__(self, model_path, dt=0.001, device='cuda'):
        self.device = device
        self.dt = dt

        # Load model
        checkpoint = torch.load(model_path, map_location=device)

        from models.meshgraphnet import create_meshgraphnet

        model_config = checkpoint.get('config', {}).get('model', {})
        self.model = create_meshgraphnet(model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        # Check if model is designed for residual prediction
        self.use_residual = model_config.get('use_residual', False)

        print(f"Loaded model for unsteady rollout")
        print(f"Time step: {dt}")
        print(f"Residual mode: {self.use_residual}")

    @torch.no_grad()
    def rollout(
        self,
        initial_data,
        num_steps,
        save_interval=10,
        apply_bc=True
    ):
        """
        Perform autoregressive rollout

        Args:
            initial_data: PyG Data with initial conditions
            num_steps: Number of time steps to simulate
            save_interval: Save state every N steps
            apply_bc: Whether to reapply boundary conditions each step

        Returns:
            trajectory: List of states at saved intervals
        """
        # Initialize trajectory
        trajectory = [initial_data.clone().cpu()]

        current_data = initial_data.clone().to(self.device)

        print(f"Starting rollout for {num_steps} steps...")

        for step in tqdm(range(num_steps)):
            # Predict next state
            current_data = self.step(current_data, apply_bc=apply_bc)

            # Save at intervals
            if (step + 1) % save_interval == 0:
                trajectory.append(current_data.clone().cpu())

        print(f"Rollout complete. Saved {len(trajectory)} states.")

        return trajectory

    @torch.no_grad()
    def step(self, data, apply_bc=True):
        """
        Single time step update

        Args:
            data: Current state
            apply_bc: Whether to reapply boundary conditions

        Returns:
            updated_data: State at next time step
        """
        # Forward pass
        if self.use_residual:
            # Model predicts next state directly
            predictions = self.model(data)
        else:
            # Model predicts time derivative
            derivatives = self.model(data)

            # Euler integration
            current_state = data.x[:, 3:7]  # (u, v, w, p)
            predictions = current_state + derivatives * self.dt

        # Update state
        updated_data = data.clone()
        updated_data.x[:, 3:7] = predictions

        # Reapply boundary conditions
        if apply_bc:
            updated_data = self.apply_boundary_conditions(updated_data)

        return updated_data

    def apply_boundary_conditions(self, data):
        """
        Reapply boundary conditions

        Args:
            data: Current state

        Returns:
            data: State with enforced boundary conditions
        """
        # Extract node types from one-hot encoding
        node_type_onehot = data.x[:, -3:]
        node_types = torch.argmax(node_type_onehot, dim=1)

        # For wall nodes (type 1), enforce no-slip: u = 0
        wall_mask = (node_types == 1)
        data.x[wall_mask, 3:6] = 0.0

        # For boundary nodes (type 2), could enforce inlet/outlet conditions
        # This is simplified - actual BC depend on problem setup

        return data

    def compute_cfl_number(self, data):
        """
        Compute CFL number for stability analysis

        CFL = u * dt / dx

        Args:
            data: Current state

        Returns:
            max_cfl: Maximum CFL number
        """
        velocity = data.x[:, 3:6]
        vel_mag = torch.norm(velocity, dim=1)

        # Estimate mesh size from edge lengths
        row, col = data.edge_index
        edge_lengths = torch.norm(data.pos[col] - data.pos[row], dim=1)
        min_dx = edge_lengths.min()

        max_cfl = (vel_mag.max() * self.dt / min_dx).item()

        return max_cfl

    def save_trajectory(self, trajectory, output_dir):
        """
        Save trajectory to disk

        Args:
            trajectory: List of states
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, state in enumerate(trajectory):
            output_path = output_dir / f'state_{i:05d}.pt'
            torch.save(state, output_path)

        print(f"Saved {len(trajectory)} states to {output_dir}")


class AdaptiveTimeStepRollout(UnsteadyRollout):
    """
    Rollout with adaptive time stepping based on CFL condition

    Args:
        model_path: Path to model
        dt_initial: Initial time step
        cfl_target: Target CFL number (default 0.5)
        dt_min: Minimum time step
        dt_max: Maximum time step
        device: Device
    """

    def __init__(
        self,
        model_path,
        dt_initial=0.001,
        cfl_target=0.5,
        dt_min=1e-6,
        dt_max=0.01,
        device='cuda'
    ):
        super().__init__(model_path, dt_initial, device)

        self.cfl_target = cfl_target
        self.dt_min = dt_min
        self.dt_max = dt_max

    @torch.no_grad()
    def step(self, data, apply_bc=True):
        """
        Adaptive time step

        Args:
            data: Current state
            apply_bc: Apply boundary conditions

        Returns:
            updated_data: Next state
            actual_dt: Time step used
        """
        # Compute CFL number
        cfl = self.compute_cfl_number(data)

        # Adjust time step
        if cfl > 0:
            self.dt = min(
                max(self.cfl_target / cfl * self.dt, self.dt_min),
                self.dt_max
            )

        # Perform step
        updated_data = super().step(data, apply_bc)

        return updated_data


def compare_with_ground_truth(
    trajectory,
    ground_truth_trajectory,
    metric='mse'
):
    """
    Compare predicted trajectory with ground truth

    Args:
        trajectory: Predicted trajectory
        ground_truth_trajectory: Ground truth trajectory
        metric: Comparison metric ('mse', 'mae', 'correlation')

    Returns:
        errors: List of errors at each time step
    """
    errors = []

    for pred_state, gt_state in zip(trajectory, ground_truth_trajectory):
        pred_vel = pred_state.x[:, 3:6].numpy()
        gt_vel = gt_state.x[:, 3:6].numpy()

        if metric == 'mse':
            error = np.mean((pred_vel - gt_vel) ** 2)
        elif metric == 'mae':
            error = np.mean(np.abs(pred_vel - gt_vel))
        elif metric == 'correlation':
            # Flatten and compute correlation
            pred_flat = pred_vel.flatten()
            gt_flat = gt_vel.flatten()
            error = np.corrcoef(pred_flat, gt_flat)[0, 1]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        errors.append(error)

    return errors


if __name__ == '__main__':
    print("Unsteady rollout module ready")
    print("Usage:")
    print("  rollout = UnsteadyRollout('checkpoints/best_model.pt', dt=0.001)")
    print("  trajectory = rollout.rollout(initial_data, num_steps=1000)")
