"""
Steady-State Flow Prediction

Predicts equilibrium flow field directly from geometry and boundary conditions
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class SteadyStatePredictor:
    """
    Predict steady-state flow field

    Args:
        model_path: Path to trained model checkpoint
        device: Device for inference ('cuda' or 'cpu')
    """

    def __init__(self, model_path, device='cuda'):
        self.device = device

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Import model class
        from models.meshgraphnet import create_meshgraphnet

        # Create model from config
        model_config = checkpoint.get('config', {}).get('model', {})
        self.model = create_meshgraphnet(model_config)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        print(f"Loaded model from {model_path}")
        print(f"Model has {self.model.count_parameters():,} parameters")

    @torch.no_grad()
    def predict(self, data):
        """
        Predict flow field for given geometry

        Args:
            data: PyG Data object with geometry and boundary conditions

        Returns:
            velocity: [N, 3] velocity field (u, v, w)
            pressure: [N, 1] pressure field (p)
        """
        # Move data to device
        data = data.to(self.device)

        # Forward pass
        predictions = self.model(data)

        # Split into velocity and pressure
        velocity = predictions[:, :3].cpu().numpy()
        pressure = predictions[:, 3].cpu().numpy()

        return velocity, pressure

    @torch.no_grad()
    def predict_batch(self, data_list):
        """
        Predict for multiple geometries

        Args:
            data_list: List of PyG Data objects

        Returns:
            velocities: List of velocity fields
            pressures: List of pressure fields
        """
        from torch_geometric.loader import DataLoader

        # Create dataloader
        loader = DataLoader(data_list, batch_size=1, shuffle=False)

        velocities = []
        pressures = []

        for batch in loader:
            vel, pres = self.predict(batch)
            velocities.append(vel)
            pressures.append(pres)

        return velocities, pressures

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        data,
        n_samples=10,
        dropout_rate=0.1
    ):
        """
        Predict with uncertainty estimation using MC Dropout

        Args:
            data: PyG Data object
            n_samples: Number of Monte Carlo samples
            dropout_rate: Dropout rate for uncertainty

        Returns:
            mean: [N, 4] mean predictions
            std: [N, 4] prediction uncertainty
        """
        # Enable dropout for MC sampling
        self.model.train()

        # Add dropout to model if not present
        # This is a simplified version - better to have dropout built into model
        data = data.to(self.device)

        predictions = []
        for _ in range(n_samples):
            pred = self.model(data)
            predictions.append(pred.cpu())

        # Stack predictions
        predictions = torch.stack(predictions)

        # Compute statistics
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        # Set model back to eval mode
        self.model.eval()

        return mean.numpy(), std.numpy()

    def compute_drag_coefficient(
        self,
        pressure,
        positions,
        normals,
        wall_mask,
        reference_area,
        dynamic_pressure
    ):
        """
        Compute drag coefficient from pressure field

        C_D = F_D / (q * A_ref)
        where q = 0.5 * rho * U^2

        Args:
            pressure: [N] pressure field
            positions: [N, 3] node positions
            normals: [N, 3] surface normals
            wall_mask: [N] boolean mask for wall nodes
            reference_area: Reference area (typically frontal area)
            dynamic_pressure: Dynamic pressure (0.5 * rho * U^2)

        Returns:
            cd: Drag coefficient
            cl: Lift coefficient
        """
        # Extract wall nodes
        wall_pressure = pressure[wall_mask]
        wall_normals = normals[wall_mask]
        wall_positions = positions[wall_mask]

        # Compute pressure force on each surface element
        # This is simplified - actual implementation needs proper surface integration
        force = wall_pressure[:, None] * wall_normals

        # Sum forces
        total_force = force.sum(axis=0)

        # Drag (x-direction) and lift (z-direction)
        drag_force = total_force[0]
        lift_force = total_force[2]

        # Coefficients
        cd = drag_force / (dynamic_pressure * reference_area)
        cl = lift_force / (dynamic_pressure * reference_area)

        return cd, cl

    def save_results(
        self,
        velocity,
        pressure,
        output_path,
        mesh=None
    ):
        """
        Save prediction results

        Args:
            velocity: Velocity field
            pressure: Pressure field
            output_path: Output file path (.npz or .vtk)
            mesh: Optional mesh for VTK output
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.npz':
            # Save as NumPy archive
            np.savez(
                output_path,
                velocity=velocity,
                pressure=pressure
            )
            print(f"Saved results to {output_path}")

        elif output_path.suffix == '.vtk' and mesh is not None:
            # Save as VTK with mesh
            try:
                import pyvista as pv
                mesh['velocity'] = velocity
                mesh['pressure'] = pressure
                mesh.save(output_path)
                print(f"Saved VTK results to {output_path}")
            except ImportError:
                print("PyVista not available. Install with: pip install pyvista")

        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")


class EnsemblePredictor:
    """
    Ensemble of multiple models for improved predictions

    Args:
        model_paths: List of paths to model checkpoints
        device: Device for inference
    """

    def __init__(self, model_paths, device='cuda'):
        self.predictors = [
            SteadyStatePredictor(path, device)
            for path in model_paths
        ]
        self.device = device

    @torch.no_grad()
    def predict(self, data):
        """
        Predict using ensemble average

        Args:
            data: PyG Data object

        Returns:
            velocity: Ensemble mean velocity
            pressure: Ensemble mean pressure
            velocity_std: Ensemble std for velocity
            pressure_std: Ensemble std for pressure
        """
        velocities = []
        pressures = []

        for predictor in self.predictors:
            vel, pres = predictor.predict(data)
            velocities.append(vel)
            pressures.append(pres)

        # Stack and compute statistics
        velocities = np.stack(velocities)
        pressures = np.stack(pressures)

        velocity_mean = velocities.mean(axis=0)
        velocity_std = velocities.std(axis=0)

        pressure_mean = pressures.mean(axis=0)
        pressure_std = pressures.std(axis=0)

        return velocity_mean, pressure_mean, velocity_std, pressure_std


def load_predictor(model_path, device='cuda', ensemble=False):
    """
    Factory function to load predictor

    Args:
        model_path: Path to model or list of paths for ensemble
        device: Device for inference
        ensemble: Whether to use ensemble

    Returns:
        predictor: SteadyStatePredictor or EnsemblePredictor
    """
    if ensemble:
        if isinstance(model_path, (list, tuple)):
            return EnsemblePredictor(model_path, device)
        else:
            raise ValueError("Ensemble mode requires list of model paths")
    else:
        return SteadyStatePredictor(model_path, device)


if __name__ == '__main__':
    print("Steady-state predictor ready")
    print("Usage:")
    print("  predictor = SteadyStatePredictor('checkpoints/best_model.pt')")
    print("  velocity, pressure = predictor.predict(data)")
