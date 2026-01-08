"""
Synthetic data generation for testing
Generates simple physics-based simulations without full LBM/CFD solver
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from pathlib import Path
import h5py
from typing import Tuple, List
from tqdm import tqdm


class DropletImpactSimulator:
    """
    Simple droplet impact simulation.

    Simulates a liquid droplet falling under gravity and impacting a surface,
    using simplified physics (no full Navier-Stokes).
    """

    def __init__(
        self,
        domain_size: Tuple[float, float, float] = (1.0, 1.0, 0.5),
        num_particles: int = 1000,
        droplet_radius: float = 0.1,
        gravity: float = -9.81,
        dt: float = 1e-5
    ):
        """
        Args:
            domain_size: Domain dimensions (x, y, z) in mm
            num_particles: Number of particles/nodes
            droplet_radius: Initial droplet radius in mm
            gravity: Gravity acceleration in m/s^2
            dt: Time step in seconds
        """
        self.domain_size = np.array(domain_size)
        self.num_particles = num_particles
        self.droplet_radius = droplet_radius
        self.gravity = gravity
        self.dt = dt

    def generate_sequence(
        self,
        num_steps: int = 200,
        initial_velocity: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    ) -> List[Data]:
        """
        Generate a simulation sequence.

        Args:
            num_steps: Number of time steps
            initial_velocity: Initial droplet velocity (m/s)

        Returns:
            List of PyG Data objects
        """
        # Initialize particles
        pos, vel, temp, phase, phi = self._initialize_droplet(initial_velocity)

        sequence = []

        for step in range(num_steps):
            # Update physics
            pos, vel, temp, phase, phi = self._update_state(pos, vel, temp, phase, phi)

            # Create graph
            data = self._create_graph(pos, vel, temp, phase, phi)

            sequence.append(data)

        return sequence

    def _initialize_droplet(
        self,
        initial_velocity: Tuple[float, float, float]
    ) -> Tuple[np.ndarray, ...]:
        """
        Initialize droplet particles.

        Returns:
            pos, vel, temp, phase, phi
        """
        # Create uniform grid
        n_per_dim = int(np.ceil(self.num_particles ** (1/3)))
        x = np.linspace(0, self.domain_size[0], n_per_dim)
        y = np.linspace(0, self.domain_size[1], n_per_dim)
        z = np.linspace(0, self.domain_size[2], n_per_dim)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        pos = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)
        pos = pos[:self.num_particles]

        # Droplet center (mid-air)
        center = self.domain_size / 2.0
        center[2] = self.domain_size[2] * 0.7  # Start above ground

        # Level-set: distance to droplet center
        dist = np.linalg.norm(pos - center, axis=-1)
        phi = dist - self.droplet_radius

        # Phase: liquid inside droplet, gas outside
        phase = np.where(phi < 0, 1.0, 2.0)  # 1=liquid, 2=gas

        # Temperature: warmer inside droplet
        temp = np.where(phi < 0, 320.0, 293.15)  # K

        # Velocity: initial velocity for droplet, zero for gas
        vel = np.zeros((self.num_particles, 3))
        droplet_mask = phi < 0
        vel[droplet_mask] = initial_velocity

        return pos, vel, temp, phase, phi

    def _update_state(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        temp: np.ndarray,
        phase: np.ndarray,
        phi: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """
        Update state with simple physics.

        Args:
            pos: Positions [N, 3]
            vel: Velocities [N, 3]
            temp: Temperatures [N]
            phase: Phases [N]
            phi: Level-set [N]

        Returns:
            Updated pos, vel, temp, phase, phi
        """
        # Gravity (only on liquid)
        liquid_mask = phase < 1.5
        gravity_accel = np.array([0.0, 0.0, self.gravity])
        vel[liquid_mask] += gravity_accel * self.dt

        # Update positions (Lagrangian)
        # For liquid particles, move with velocity
        pos[liquid_mask] += vel[liquid_mask] * self.dt

        # Boundary conditions (bounce off ground)
        ground_collision = (pos[:, 2] < 0.01) & liquid_mask
        vel[ground_collision, 2] = -0.5 * vel[ground_collision, 2]  # Inelastic bounce
        pos[ground_collision, 2] = 0.01

        # Update level-set (simple advection)
        # In full simulation, would solve level-set equation
        # Here, just recompute based on liquid particle positions
        liquid_pos = pos[liquid_mask]
        if len(liquid_pos) > 0:
            # Approximate droplet center as mean of liquid particles
            new_center = np.mean(liquid_pos, axis=0)

            # Update phi based on distance to new center
            dist = np.linalg.norm(pos - new_center, axis=-1)
            phi = dist - self.droplet_radius

            # Update phase
            phase = np.where(phi < 0, 1.0, 2.0)

        # Temperature decay (simple cooling)
        temp = temp - (temp - 293.15) * 0.001 * self.dt

        return pos, vel, temp, phase, phi

    def _create_graph(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        temp: np.ndarray,
        phase: np.ndarray,
        phi: np.ndarray
    ) -> Data:
        """
        Create PyG Data object from state.

        Args:
            pos: Positions [N, 3]
            vel: Velocities [N, 3]
            temp: Temperatures [N]
            phase: Phases [N]
            phi: Level-set [N]

        Returns:
            PyG Data object
        """
        # Convert to tensors
        pos_t = torch.from_numpy(pos).float()
        vel_t = torch.from_numpy(vel).float()
        temp_t = torch.from_numpy(temp).float()
        phase_t = torch.from_numpy(phase).float()
        phi_t = torch.from_numpy(phi).float()

        # Node type (0=interior, 1=boundary, 2=free_surface)
        node_type = torch.zeros(self.num_particles)
        interface_mask = torch.abs(phi_t) < 0.1
        node_type[interface_mask] = 2

        # Node features: [x, y, z, T, phase, u, v, w, phi, type]
        node_features = torch.cat([
            pos_t,
            temp_t.unsqueeze(-1),
            phase_t.unsqueeze(-1),
            vel_t,
            phi_t.unsqueeze(-1),
            node_type.unsqueeze(-1)
        ], dim=-1)

        # Build edges (k-NN)
        edge_index = knn_graph(pos_t, k=8, loop=False)

        # Edge features: [dx, dy, dz, distance]
        row, col = edge_index
        rel_pos = pos_t[col] - pos_t[row]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        edge_attr = torch.cat([rel_pos, dist], dim=-1)

        # Create Data
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos_t
        )

        return data


class MeltPoolSimulator:
    """
    Simple melt pool simulation with laser heating.

    Simulates laser melting of powder bed with simplified physics.
    """

    def __init__(
        self,
        domain_size: Tuple[float, float, float] = (1.0, 1.0, 0.5),
        num_particles: int = 1000,
        laser_power: float = 100.0,
        laser_radius: float = 0.1,
        dt: float = 1e-5
    ):
        """
        Args:
            domain_size: Domain dimensions (mm)
            num_particles: Number of particles
            laser_power: Laser power (W)
            laser_radius: Laser beam radius (mm)
            dt: Time step (s)
        """
        self.domain_size = np.array(domain_size)
        self.num_particles = num_particles
        self.laser_power = laser_power
        self.laser_radius = laser_radius
        self.dt = dt

        # Material properties
        self.T_melt = 1800.0  # Melting temp (K) - typical for metals
        self.T_initial = 293.15  # Room temp
        self.k = 50.0  # Thermal conductivity
        self.cp = 500.0  # Specific heat
        self.rho = 7000.0  # Density (kg/m^3)

    def generate_sequence(
        self,
        num_steps: int = 200,
        laser_path: str = 'linear'
    ) -> List[Data]:
        """
        Generate melt pool sequence.

        Args:
            num_steps: Number of time steps
            laser_path: Laser scan path ('linear', 'circular')

        Returns:
            List of PyG Data objects
        """
        # Initialize
        pos, temp, phase, vel = self._initialize_powder_bed()

        sequence = []

        for step in range(num_steps):
            # Laser position
            laser_pos = self._get_laser_position(step, num_steps, laser_path)

            # Update physics
            pos, temp, phase, vel = self._update_meltpool(
                pos, temp, phase, vel, laser_pos
            )

            # Create graph
            phi = self._compute_level_set(pos, phase)
            data = self._create_graph(pos, vel, temp, phase, phi)

            sequence.append(data)

        return sequence

    def _initialize_powder_bed(self) -> Tuple[np.ndarray, ...]:
        """Initialize powder bed."""
        # Uniform grid
        n_per_dim = int(np.ceil(self.num_particles ** (1/3)))
        x = np.linspace(0, self.domain_size[0], n_per_dim)
        y = np.linspace(0, self.domain_size[1], n_per_dim)
        z = np.linspace(0, self.domain_size[2], n_per_dim // 2)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        pos = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)
        pos = pos[:self.num_particles]

        # Initial state: solid at room temperature
        temp = np.full(self.num_particles, self.T_initial)
        phase = np.zeros(self.num_particles)  # 0 = solid
        vel = np.zeros((self.num_particles, 3))

        return pos, temp, phase, vel

    def _get_laser_position(
        self,
        step: int,
        total_steps: int,
        path_type: str
    ) -> np.ndarray:
        """Get laser beam position at current step."""
        if path_type == 'linear':
            # Linear scan along x-axis
            t = step / total_steps
            x = t * self.domain_size[0]
            y = self.domain_size[1] / 2
            z = self.domain_size[2]
        elif path_type == 'circular':
            # Circular scan
            t = step / total_steps * 2 * np.pi
            radius = self.domain_size[0] / 4
            x = self.domain_size[0] / 2 + radius * np.cos(t)
            y = self.domain_size[1] / 2 + radius * np.sin(t)
            z = self.domain_size[2]
        else:
            raise ValueError(f"Unknown laser path: {path_type}")

        return np.array([x, y, z])

    def _update_meltpool(
        self,
        pos: np.ndarray,
        temp: np.ndarray,
        phase: np.ndarray,
        vel: np.ndarray,
        laser_pos: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        """Update melt pool state."""
        # Laser heating (Gaussian)
        dist_to_laser = np.linalg.norm(pos - laser_pos, axis=-1)
        Q = self.laser_power * np.exp(-(dist_to_laser ** 2) / (self.laser_radius ** 2))

        # Temperature increase
        dT = Q * self.dt / (self.rho * self.cp)
        temp = temp + dT

        # Phase change (melting)
        melting_mask = (temp > self.T_melt) & (phase < 0.5)
        phase[melting_mask] = 1.0  # Solid -> Liquid

        # Cooling (simple heat loss)
        temp = temp - (temp - self.T_initial) * 0.01 * self.dt

        # Solidification
        solidifying_mask = (temp < self.T_melt) & (phase > 0.5)
        phase[solidifying_mask] = 0.0  # Liquid -> Solid

        # Marangoni flow (simplified)
        # Liquid flows away from hot regions
        liquid_mask = phase > 0.5
        if liquid_mask.any():
            temp_grad_approx = -(pos - laser_pos) / (dist_to_laser[:, None] + 1e-6)
            vel[liquid_mask] = 0.1 * temp_grad_approx[liquid_mask]

        return pos, temp, phase, vel

    def _compute_level_set(self, pos: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """Compute level-set from phase."""
        # Simple: phi = 0 at phase = 0.5, negative for liquid, positive for solid
        phi = 0.5 - phase
        return phi

    def _create_graph(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        temp: np.ndarray,
        phase: np.ndarray,
        phi: np.ndarray
    ) -> Data:
        """Create PyG Data from state."""
        pos_t = torch.from_numpy(pos).float()
        vel_t = torch.from_numpy(vel).float()
        temp_t = torch.from_numpy(temp).float()
        phase_t = torch.from_numpy(phase).float()
        phi_t = torch.from_numpy(phi).float()

        node_type = torch.zeros(self.num_particles)

        node_features = torch.cat([
            pos_t,
            temp_t.unsqueeze(-1),
            phase_t.unsqueeze(-1),
            vel_t,
            phi_t.unsqueeze(-1),
            node_type.unsqueeze(-1)
        ], dim=-1)

        edge_index = knn_graph(pos_t, k=8, loop=False)

        row, col = edge_index
        rel_pos = pos_t[col] - pos_t[row]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        edge_attr = torch.cat([rel_pos, dist], dim=-1)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos_t
        )

        return data


def generate_dataset(
    output_dir: str,
    num_sequences: int = 100,
    sequence_length: int = 200,
    simulator_type: str = 'droplet'
):
    """
    Generate synthetic dataset and save to HDF5.

    Args:
        output_dir: Output directory
        num_sequences: Number of sequences to generate
        sequence_length: Length of each sequence
        simulator_type: 'droplet' or 'meltpool'
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Create simulator
    if simulator_type == 'droplet':
        simulator = DropletImpactSimulator()
    elif simulator_type == 'meltpool':
        simulator = MeltPoolSimulator()
    else:
        raise ValueError(f"Unknown simulator: {simulator_type}")

    print(f"Generating {num_sequences} sequences...")

    # Generate and save sequences
    for seq_idx in tqdm(range(num_sequences)):
        # Generate sequence
        if simulator_type == 'droplet':
            initial_vel = (np.random.rand(3) - 0.5) * 2.0
            sequence = simulator.generate_sequence(sequence_length, initial_vel)
        else:
            laser_path = np.random.choice(['linear', 'circular'])
            sequence = simulator.generate_sequence(sequence_length, laser_path)

        # Save to HDF5
        file_path = output_path / f'sequence_{seq_idx:04d}.h5'
        save_sequence_to_hdf5(sequence, file_path, seq_idx)

    print(f"Dataset saved to {output_path}")


def save_sequence_to_hdf5(sequence: List[Data], file_path: Path, seq_idx: int):
    """Save sequence to HDF5 file."""
    with h5py.File(file_path, 'w') as f:
        seq_key = f'sequence_{seq_idx}'
        seq_group = f.create_group(seq_key)

        for frame_idx, data in enumerate(sequence):
            frame_key = f'frame_{frame_idx}'
            frame_group = seq_group.create_group(frame_key)

            # Save data
            frame_group.create_dataset('node_features', data=data.x.numpy())
            frame_group.create_dataset('edge_index', data=data.edge_index.numpy())
            frame_group.create_dataset('edge_attr', data=data.edge_attr.numpy())


if __name__ == '__main__':
    # Generate training data
    generate_dataset(
        output_dir='data/processed/train',
        num_sequences=100,
        sequence_length=200,
        simulator_type='droplet'
    )

    # Generate validation data
    generate_dataset(
        output_dir='data/processed/val',
        num_sequences=20,
        sequence_length=200,
        simulator_type='droplet'
    )

    # Generate test data
    generate_dataset(
        output_dir='data/processed/test',
        num_sequences=10,
        sequence_length=200,
        simulator_type='droplet'
    )
