"""
PyTorch Geometric Dataset for Car Aerodynamics

Supports both steady-state and unsteady simulations
"""

import os
import torch
from torch_geometric.data import Dataset, InMemoryDataset, Data
from pathlib import Path
import json
from typing import Optional, Callable, List


class CarAeroDataset(InMemoryDataset):
    """
    In-memory dataset for car aerodynamics

    For steady-state: Each sample = (geometry + BC, flow field)
    For unsteady: Each sample = (state at t, state at t+1)

    Args:
        root: Root directory
        mode: 'steady' or 'unsteady'
        split: 'train', 'val', or 'test'
        transform: Optional transform to apply
        pre_transform: Optional pre-transform
    """

    def __init__(
        self,
        root,
        mode='steady',
        split='train',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None
    ):
        self.mode = mode
        self.split = split
        assert split in ['train', 'val', 'test'], "Split must be train/val/test"
        assert mode in ['steady', 'unsteady'], "Mode must be steady/unsteady"

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Raw file names (not used in this implementation)"""
        return []

    @property
    def processed_file_names(self) -> List[str]:
        """Processed file names"""
        return [f'{self.split}_{self.mode}.pt']

    def process(self):
        """
        Process raw data into graphs

        This method loads preprocessed graph files and creates
        train/val/test splits
        """
        # Path to preprocessed graphs
        processed_dir = Path(self.root) / 'processed_graphs'

        if not processed_dir.exists():
            print(f"Warning: {processed_dir} does not exist.")
            print("Run data/preprocess.py first to generate graph data.")
            # Create empty dataset
            data_list = []
        else:
            # Load all graph files
            graph_files = sorted(list(processed_dir.glob('graph_*.pt')))
            print(f"Found {len(graph_files)} graph files")

            # Load graphs
            data_list = []
            for graph_file in graph_files:
                data = torch.load(graph_file)
                data_list.append(data)

            # Create train/val/test split
            num_samples = len(data_list)
            train_size = int(0.8 * num_samples)
            val_size = int(0.1 * num_samples)

            if self.split == 'train':
                data_list = data_list[:train_size]
            elif self.split == 'val':
                data_list = data_list[train_size:train_size + val_size]
            else:  # test
                data_list = data_list[train_size + val_size:]

            print(f"Split '{self.split}': {len(data_list)} samples")

        # Apply pre-transform if provided
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Collate and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_statistics(self):
        """Compute dataset statistics"""
        if len(self) == 0:
            return {}

        # Sample a few graphs to compute stats
        num_samples = min(100, len(self))
        indices = torch.randperm(len(self))[:num_samples]

        num_nodes_list = []
        num_edges_list = []
        velocity_list = []
        pressure_list = []

        for idx in indices:
            data = self[idx.item()]
            num_nodes_list.append(data.num_nodes)
            num_edges_list.append(data.num_edges)

            if hasattr(data, 'y'):
                velocity_list.append(data.y[:, :3])
                pressure_list.append(data.y[:, 3:4])

        stats = {
            'num_samples': len(self),
            'avg_num_nodes': sum(num_nodes_list) / len(num_nodes_list),
            'avg_num_edges': sum(num_edges_list) / len(num_edges_list),
        }

        if velocity_list:
            all_vel = torch.cat(velocity_list, dim=0)
            all_pres = torch.cat(pressure_list, dim=0)

            stats['velocity_mean'] = all_vel.mean(dim=0).tolist()
            stats['velocity_std'] = all_vel.std(dim=0).tolist()
            stats['pressure_mean'] = all_pres.mean().item()
            stats['pressure_std'] = all_pres.std().item()

        return stats


class OnDiskCarAeroDataset(Dataset):
    """
    On-disk dataset for very large datasets that don't fit in memory

    Args:
        root: Root directory with processed graphs
        mode: 'steady' or 'unsteady'
        split: 'train', 'val', or 'test'
        transform: Optional transform
    """

    def __init__(
        self,
        root,
        mode='steady',
        split='train',
        transform: Optional[Callable] = None
    ):
        self.root = Path(root)
        self.mode = mode
        self.split = split
        self._transform = transform

        # Load file list
        processed_dir = self.root / 'processed_graphs'
        self.graph_files = sorted(list(processed_dir.glob('graph_*.pt')))

        # Create split
        num_samples = len(self.graph_files)
        train_size = int(0.8 * num_samples)
        val_size = int(0.1 * num_samples)

        if split == 'train':
            self.graph_files = self.graph_files[:train_size]
        elif split == 'val':
            self.graph_files = self.graph_files[train_size:train_size + val_size]
        else:  # test
            self.graph_files = self.graph_files[train_size + val_size:]

        super().__init__(root, transform)

    def len(self) -> int:
        return len(self.graph_files)

    def get(self, idx: int) -> Data:
        """Load graph from disk"""
        data = torch.load(self.graph_files[idx])

        if self._transform is not None:
            data = self._transform(data)

        return data


class NoiseTransform:
    """
    Add noise to node features for data augmentation

    Args:
        noise_std: Standard deviation of Gaussian noise
        velocity_only: Only add noise to velocity features
    """

    def __init__(self, noise_std=0.003, velocity_only=True):
        self.noise_std = noise_std
        self.velocity_only = velocity_only

    def __call__(self, data):
        data = data.clone()

        if self.velocity_only:
            # Add noise only to velocity features (indices 3:6)
            noise = torch.randn_like(data.x[:, 3:6]) * self.noise_std
            data.x[:, 3:6] = data.x[:, 3:6] + noise
        else:
            # Add noise to all features
            noise = torch.randn_like(data.x) * self.noise_std
            data.x = data.x + noise

        return data


class NormalizeTransform:
    """
    Normalize features using pre-computed statistics

    Args:
        stats: Dictionary with mean and std for each feature
    """

    def __init__(self, stats):
        self.stats = stats

    def __call__(self, data):
        data = data.clone()

        # Normalize velocity
        if 'velocity_mean' in self.stats:
            vel_mean = torch.tensor(self.stats['velocity_mean'])
            vel_std = torch.tensor(self.stats['velocity_std'])
            data.x[:, 3:6] = (data.x[:, 3:6] - vel_mean) / (vel_std + 1e-8)

        # Normalize pressure
        if 'pressure_mean' in self.stats:
            pres_mean = self.stats['pressure_mean']
            pres_std = self.stats['pressure_std']
            data.x[:, 6:7] = (data.x[:, 6:7] - pres_mean) / (pres_std + 1e-8)

        return data


def create_datasets(root, mode='steady', use_transforms=True, noise_std=0.003):
    """
    Factory function to create train/val/test datasets

    Args:
        root: Root directory
        mode: 'steady' or 'unsteady'
        use_transforms: Whether to use data augmentation
        noise_std: Noise level for augmentation

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Create datasets
    train_dataset = CarAeroDataset(root, mode=mode, split='train')
    val_dataset = CarAeroDataset(root, mode=mode, split='val')
    test_dataset = CarAeroDataset(root, mode=mode, split='test')

    # Compute statistics from training set
    if use_transforms:
        stats = train_dataset.get_statistics()
        print("Dataset statistics:", stats)

        # Add transforms to training set
        train_dataset.transform = NoiseTransform(noise_std=noise_std)

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    # Test dataset creation
    print("Testing CarAeroDataset...")

    root = 'data'
    try:
        train_ds, val_ds, test_ds = create_datasets(root, mode='steady')

        print(f"\nTrain samples: {len(train_ds)}")
        print(f"Val samples: {len(val_ds)}")
        print(f"Test samples: {len(test_ds)}")

        if len(train_ds) > 0:
            sample = train_ds[0]
            print(f"\nSample data:")
            print(f"  Nodes: {sample.num_nodes}")
            print(f"  Edges: {sample.num_edges}")
            print(f"  Node features shape: {sample.x.shape}")
            print(f"  Edge features shape: {sample.edge_attr.shape}")
            if hasattr(sample, 'y'):
                print(f"  Target shape: {sample.y.shape}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run data/preprocess.py first")
