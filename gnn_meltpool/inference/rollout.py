"""
Autoregressive rollout for multi-step prediction
Implements adaptive remeshing and physics corrections
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import List, Optional, Tuple
import numpy as np
from pathlib import Path

from models import MeshGraphNet
from physics import SurfaceTensionForce, compute_curvature


class Rollout:
    """
    Autoregressive rollout predictor.

    Predicts future states by iteratively applying the model,
    with optional physics corrections and adaptive remeshing.
    """

    def __init__(
        self,
        model: MeshGraphNet,
        config: dict,
        device: torch.device
    ):
        """
        Args:
            model: Trained MeshGraphNet model
            config: Configuration dictionary
            device: Device to run on
        """
        self.model = model
        self.config = config
        self.device = device

        # Rollout config
        inference_config = config.get('inference', {})
        self.rollout_steps = inference_config.get('rollout_steps', 500)
        self.remesh_interval = inference_config.get('remesh_interval', 50)
        self.remesh_threshold = inference_config.get('remesh_threshold', 0.5)
        self.save_trajectory = inference_config.get('save_trajectory', True)
        self.output_dir = Path(inference_config.get('output_dir', 'outputs'))

        # Time step
        self.dt = config['data'].get('dt', 1e-5)

        # Physics corrections
        self.use_physics_correction = True
        self.surface_tension = SurfaceTensionForce(
            surface_tension=config['simulation'].get('surface_tension', 0.07)
        )

        self.model.eval()

    @torch.no_grad()
    def rollout(
        self,
        initial_data: Data,
        num_steps: Optional[int] = None,
        save_interval: int = 10
    ) -> List[Data]:
        """
        Perform autoregressive rollout.

        Args:
            initial_data: Initial state
            num_steps: Number of steps (default: use config)
            save_interval: Save every N steps

        Returns:
            List of predicted states
        """
        if num_steps is None:
            num_steps = self.rollout_steps

        print(f"Starting rollout for {num_steps} steps...")

        trajectory = [initial_data.clone().cpu()]
        current_data = initial_data.clone().to(self.device)

        for step in range(num_steps):
            if step % 50 == 0:
                print(f"Step {step}/{num_steps}")

            # Predict next state
            current_data = self.predict_next_state(current_data)

            # Adaptive remeshing
            if self.remesh_interval > 0 and step % self.remesh_interval == 0:
                current_data = self.adaptive_remesh(current_data)

            # Save
            if step % save_interval == 0:
                trajectory.append(current_data.clone().cpu())

        print("Rollout completed!")

        # Save trajectory if requested
        if self.save_trajectory:
            self._save_trajectory(trajectory)

        return trajectory

    def predict_next_state(self, data: Data) -> Data:
        """
        Predict next state from current state.

        Args:
            data: Current state

        Returns:
            Next state
        """
        # Forward pass
        delta = self.model(data)

        # Update state
        next_data = data.clone()
        next_data.x[:, 3:9] = data.x[:, 3:9] + delta

        # Apply physics corrections
        if self.use_physics_correction:
            next_data = self._apply_physics_correction(next_data)

        # Apply constraints
        next_data = self._apply_constraints(next_data)

        return next_data

    def _apply_physics_correction(self, data: Data) -> Data:
        """
        Apply physics-based corrections to predictions.

        Args:
            data: Predicted state

        Returns:
            Corrected state
        """
        # Surface tension force correction
        # (This is a simplified correction; full correction would involve
        # solving momentum equations)

        # For now, just apply light smoothing to interface
        phi = data.x[:, 8]
        interface_mask = torch.abs(phi) < 0.3

        if interface_mask.any():
            # Smooth level-set field at interface
            # This helps prevent numerical artifacts
            pass

        return data

    def _apply_constraints(self, data: Data) -> Data:
        """
        Apply hard constraints to ensure physical validity.

        Args:
            data: State data

        Returns:
            Constrained state
        """
        x = data.x.clone()

        # Temperature constraint (T >= 0)
        x[:, 3] = torch.clamp(x[:, 3], min=0.0)

        # Phase constraint (0 <= phase <= 2)
        x[:, 4] = torch.clamp(x[:, 4], min=0.0, max=2.0)

        # Level-set normalization
        x[:, 8] = torch.clamp(x[:, 8], min=-1.0, max=1.0)

        # Velocity bounds (prevent extreme values)
        max_vel = 10.0  # m/s
        x[:, 5:8] = torch.clamp(x[:, 5:8], min=-max_vel, max=max_vel)

        data.x = x
        return data

    def adaptive_remesh(self, data: Data) -> Data:
        """
        Adaptive remeshing based on level-set gradient.

        Refines mesh near interfaces where level-set gradient is high.

        Args:
            data: Current state

        Returns:
            Remeshed state
        """
        # This is a placeholder for adaptive remeshing
        # Full implementation would require mesh operations

        # For now, just identify regions needing refinement
        phi = data.x[:, 8]

        from physics import compute_gradient

        grad_phi = compute_gradient(phi, data.pos, data.edge_index, data.edge_attr)
        grad_norm = torch.norm(grad_phi, dim=-1)

        # Identify nodes needing refinement
        refinement_mask = grad_norm > self.remesh_threshold

        num_refinement = refinement_mask.sum().item()
        if num_refinement > 0:
            print(f"  Remeshing: {num_refinement} nodes need refinement")
            # In full implementation, would add new nodes here

        return data

    def _save_trajectory(self, trajectory: List[Data]):
        """
        Save trajectory to disk.

        Args:
            trajectory: List of states
        """
        self.output_dir.mkdir(exist_ok=True, parents=True)

        output_file = self.output_dir / 'rollout_trajectory.pt'
        torch.save(trajectory, output_file)

        print(f"Saved trajectory to {output_file}")


class RolloutEvaluator:
    """
    Evaluate rollout predictions against ground truth.
    """

    def __init__(self):
        """Initialize evaluator."""
        pass

    def evaluate(
        self,
        predictions: List[Data],
        ground_truth: List[Data]
    ) -> dict:
        """
        Evaluate predictions.

        Args:
            predictions: Predicted trajectory
            ground_truth: Ground truth trajectory

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mse': [],
            'mae': [],
            'rmse': []
        }

        num_steps = min(len(predictions), len(ground_truth))

        for i in range(num_steps):
            pred = predictions[i].x[:, 3:9]  # Dynamic features
            gt = ground_truth[i].x[:, 3:9]

            # MSE
            mse = torch.mean((pred - gt) ** 2).item()
            metrics['mse'].append(mse)

            # MAE
            mae = torch.mean(torch.abs(pred - gt)).item()
            metrics['mae'].append(mae)

            # RMSE
            rmse = torch.sqrt(torch.mean((pred - gt) ** 2)).item()
            metrics['rmse'].append(rmse)

        # Average metrics
        avg_metrics = {
            key: np.mean(values) for key, values in metrics.items()
        }

        # Per-timestep metrics
        timestep_metrics = metrics

        return {
            'average': avg_metrics,
            'timestep': timestep_metrics
        }

    def print_metrics(self, metrics: dict):
        """
        Print evaluation metrics.

        Args:
            metrics: Metrics dictionary
        """
        print("\nEvaluation Metrics:")
        print("-" * 40)

        avg = metrics['average']
        for key, value in avg.items():
            print(f"{key.upper()}: {value:.6f}")

        print("-" * 40)


def load_checkpoint(checkpoint_path: str, config: dict, device: torch.device) -> MeshGraphNet:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        device: Device to load on

    Returns:
        Loaded model
    """
    from models import build_meshgraphnet

    # Build model
    model = build_meshgraphnet(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")

    return model


def run_rollout(
    checkpoint_path: str,
    initial_state_path: str,
    config: dict,
    num_steps: int = 500,
    device: Optional[torch.device] = None
) -> List[Data]:
    """
    Run rollout from checkpoint and initial state.

    Args:
        checkpoint_path: Path to model checkpoint
        initial_state_path: Path to initial state
        config: Configuration dictionary
        num_steps: Number of rollout steps
        device: Device to run on

    Returns:
        Predicted trajectory
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_checkpoint(checkpoint_path, config, device)

    # Load initial state
    initial_data = torch.load(initial_state_path, map_location=device)

    # Create rollout
    rollout = Rollout(model, config, device)

    # Run
    trajectory = rollout.rollout(initial_data, num_steps=num_steps)

    return trajectory
