"""
Dataset classes for melt pool simulation
Loads and preprocesses graph sequences from simulation data
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from pathlib import Path
import h5py
from typing import List, Tuple, Optional
import glob


class MeltPoolDataset(Dataset):
    """
    Dataset for melt pool and free surface dynamics.

    Each sample is a pair (current_graph, next_state_delta) extracted
    from simulation sequences.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        noise_std: float = 0.003,
        normalize: bool = True,
        sequence_length: int = 200,
        dt: float = 1e-5
    ):
        """
        Args:
            data_dir: Directory containing processed graph sequences
            split: Dataset split ('train', 'val', 'test')
            noise_std: Standard deviation of noise to add to inputs
            normalize: Whether to normalize features
            sequence_length: Length of each sequence
            dt: Time step between frames
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.split = split
        self.noise_std = noise_std
        self.normalize = normalize
        self.sequence_length = sequence_length
        self.dt = dt

        # Load sequences
        self.sequences = self._load_sequences()

        # Compute normalization statistics
        if self.normalize:
            self.stats = self._compute_stats()
        else:
            self.stats = None

        print(f"Loaded {len(self.sequences)} sequences for {split} split")

    def _load_sequences(self) -> List[Tuple[Path, int]]:
        """
        Load list of available sequences.

        Returns:
            List of (file_path, sequence_index) tuples
        """
        sequences = []

        # Find all HDF5 files in split directory
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist. Creating empty dataset.")
            return sequences

        for file_path in sorted(split_dir.glob('*.h5')):
            # Check number of sequences in file
            with h5py.File(file_path, 'r') as f:
                num_sequences = len([k for k in f.keys() if k.startswith('sequence_')])

            # Add all sequences from this file
            for seq_idx in range(num_sequences):
                sequences.append((file_path, seq_idx))

        return sequences

    def _compute_stats(self) -> dict:
        """
        Compute normalization statistics from training data.

        Returns:
            Dictionary with mean and std for each feature
        """
        # For now, return default stats
        # In production, compute from actual data
        stats = {
            'pos_mean': torch.zeros(3),
            'pos_std': torch.ones(3),
            'temp_mean': 293.15,
            'temp_std': 100.0,
            'vel_mean': 0.0,
            'vel_std': 1.0,
            'phi_mean': 0.0,
            'phi_std': 1.0
        }
        return stats

    def __len__(self) -> int:
        """Number of samples in dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (current_graph, target_delta)
            - current_graph: PyG Data object with current state
            - target_delta: Target state changes [num_nodes, feature_dim]
        """
        file_path, seq_idx = self.sequences[idx]

        # Load sequence from file
        with h5py.File(file_path, 'r') as f:
            seq_key = f'sequence_{seq_idx}'

            # Load two consecutive frames
            frame_idx = np.random.randint(0, self.sequence_length - 1)

            # Current frame
            graph_t = self._load_frame(f, seq_key, frame_idx)

            # Next frame
            graph_t1 = self._load_frame(f, seq_key, frame_idx + 1)

        # Add noise to current state (for robustness)
        if self.noise_std > 0 and self.split == 'train':
            noise = torch.randn_like(graph_t.x) * self.noise_std
            graph_t.x = graph_t.x + noise

        # Compute target (state change)
        target_delta = graph_t1.x[:, 3:9] - graph_t.x[:, 3:9]  # [T, phase, u, v, w, phi]

        return graph_t, target_delta

    def _load_frame(self, file: h5py.File, seq_key: str, frame_idx: int) -> Data:
        """
        Load a single frame as PyG Data object.

        Args:
            file: Open HDF5 file
            seq_key: Sequence key in file
            frame_idx: Frame index in sequence

        Returns:
            PyG Data object
        """
        frame_key = f'{seq_key}/frame_{frame_idx}'

        # Load data
        x = torch.from_numpy(file[f'{frame_key}/node_features'][:]).float()
        edge_index = torch.from_numpy(file[f'{frame_key}/edge_index'][:]).long()
        edge_attr = torch.from_numpy(file[f'{frame_key}/edge_attr'][:]).float()
        pos = x[:, :3]  # Position is first 3 features

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos
        )

        return data


class SyntheticMeltPoolDataset(Dataset):
    """
    Synthetic dataset for testing without real simulation data.
    Generates simple physics-based trajectories on-the-fly.
    """

    def __init__(
        self,
        num_sequences: int = 100,
        sequence_length: int = 200,
        num_nodes: int = 1000,
        noise_std: float = 0.003,
        dt: float = 1e-5
    ):
        """
        Args:
            num_sequences: Number of sequences to generate
            sequence_length: Length of each sequence
            num_nodes: Number of nodes per graph
            noise_std: Noise standard deviation
            dt: Time step
        """
        super().__init__()

        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.num_nodes = num_nodes
        self.noise_std = noise_std
        self.dt = dt

        # Pre-generate sequence metadata
        self.sequence_params = self._generate_sequence_params()

    def _generate_sequence_params(self) -> List[dict]:
        """Generate random parameters for each sequence."""
        params = []
        for _ in range(self.num_sequences):
            params.append({
                'droplet_center': np.random.rand(3) * 0.5,
                'droplet_radius': 0.1 + np.random.rand() * 0.05,
                'initial_velocity': (np.random.rand(3) - 0.5) * 2.0,
                'temperature': 300 + np.random.rand() * 100
            })
        return params

    def __len__(self) -> int:
        return self.num_sequences * (self.sequence_length - 1)

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor]:
        """
        Generate a sample on-the-fly.

        Args:
            idx: Sample index

        Returns:
            Tuple of (current_graph, target_delta)
        """
        # Determine which sequence and frame
        seq_idx = idx // (self.sequence_length - 1)
        frame_idx = idx % (self.sequence_length - 1)

        params = self.sequence_params[seq_idx]

        # Generate current and next frames
        graph_t = self._generate_frame(params, frame_idx)
        graph_t1 = self._generate_frame(params, frame_idx + 1)

        # Add noise
        if self.noise_std > 0:
            noise = torch.randn_like(graph_t.x) * self.noise_std
            graph_t.x = graph_t.x + noise

        # Target delta
        target_delta = graph_t1.x[:, 3:9] - graph_t.x[:, 3:9]

        return graph_t, target_delta

    def _generate_frame(self, params: dict, frame_idx: int) -> Data:
        """
        Generate a single frame.

        Args:
            params: Sequence parameters
            frame_idx: Frame index

        Returns:
            PyG Data object
        """
        # Simple cubic grid
        nodes_per_dim = int(np.ceil(self.num_nodes ** (1/3)))
        x = np.linspace(0, 1, nodes_per_dim)
        y = np.linspace(0, 1, nodes_per_dim)
        z = np.linspace(0, 0.5, nodes_per_dim // 2)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        pos = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)
        pos = pos[:self.num_nodes]  # Trim to exact number

        pos = torch.from_numpy(pos).float()

        # Simulate droplet: distance from center
        center = torch.tensor(params['droplet_center']).float()
        radius = params['droplet_radius']

        dist_from_center = torch.norm(pos - center, dim=-1)

        # Level-set: negative inside droplet, positive outside
        phi = dist_from_center - radius

        # Phase: 1 (liquid) inside, 2 (gas) outside
        phase = torch.where(phi < 0, torch.ones_like(phi), torch.ones_like(phi) * 2.0)

        # Temperature: higher inside droplet
        temp = torch.where(
            phi < 0,
            torch.full_like(phi, params['temperature']),
            torch.full_like(phi, 293.15)
        )

        # Velocity: initial velocity + gravity effect
        time = frame_idx * self.dt
        gravity = torch.tensor([0.0, 0.0, -9.81])
        vel_init = torch.tensor(params['initial_velocity']).float()
        velocity = vel_init + gravity * time
        velocity = velocity.unsqueeze(0).repeat(self.num_nodes, 1)

        # Node type: all interior for simplicity
        node_type = torch.zeros(self.num_nodes)

        # Assemble node features: [x, y, z, T, phase, u, v, w, phi, type]
        node_features = torch.cat([
            pos,  # x, y, z
            temp.unsqueeze(-1),  # T
            phase.unsqueeze(-1),  # phase
            velocity,  # u, v, w
            phi.unsqueeze(-1),  # phi
            node_type.unsqueeze(-1)  # type
        ], dim=-1)

        # Build edges (k-nearest neighbors)
        edge_index, edge_attr = self._build_edges(pos)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos
        )

        return data

    def _build_edges(self, pos: torch.Tensor, k: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build k-nearest neighbor edges.

        Args:
            pos: Node positions [num_nodes, 3]
            k: Number of neighbors

        Returns:
            edge_index [2, num_edges]
            edge_attr [num_edges, 4]
        """
        from torch_geometric.nn import knn_graph
        from models.encoder import build_edge_features

        # Build k-NN graph
        edge_index = knn_graph(pos, k=k, loop=False)

        # Build edge features
        edge_attr = build_edge_features(pos, edge_index)

        return edge_index, edge_attr


def collate_fn(batch: List[Tuple[Data, torch.Tensor]]) -> Tuple[Data, torch.Tensor]:
    """
    Custom collate function for DataLoader.

    Args:
        batch: List of (graph, target) tuples

    Returns:
        Batched graph and targets
    """
    from torch_geometric.data import Batch

    graphs, targets = zip(*batch)

    # Batch graphs
    batched_graph = Batch.from_data_list(graphs)

    # Stack targets
    batched_targets = torch.cat(targets, dim=0)

    return batched_graph, batched_targets
