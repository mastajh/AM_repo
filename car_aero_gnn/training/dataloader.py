"""
DataLoaders for distributed training and efficient batching
"""

import torch
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler


def create_dataloader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    distributed=False,
    rank=0,
    world_size=1
):
    """
    Create DataLoader with optional distributed training support

    Args:
        dataset: PyG Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        distributed: Use distributed sampler
        rank: Process rank for distributed training
        world_size: Total number of processes

    Returns:
        dataloader: PyG DataLoader
    """
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        follow_batch=['x', 'edge_attr']  # Track batch assignments
    )

    return loader


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=4,
    num_workers=4,
    distributed=False,
    rank=0,
    world_size=1
):
    """
    Create train/val/test dataloaders

    Args:
        train_dataset, val_dataset, test_dataset: Datasets
        batch_size: Batch size
        num_workers: Number of workers
        distributed: Use distributed training
        rank: Process rank
        world_size: Total processes

    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        distributed=False  # No need for distributed validation
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        distributed=False
    )

    return train_loader, val_loader, test_loader


class InfiniteDataLoader:
    """
    Wrapper for infinite iteration over dataloader
    Useful for training with step-based schedules instead of epoch-based
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch


if __name__ == '__main__':
    # Test dataloader creation
    from .dataset import create_datasets

    print("Testing DataLoader creation...")

    try:
        train_ds, val_ds, test_ds = create_datasets('data', mode='steady')

        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds,
            batch_size=2,
            num_workers=0  # Use 0 for testing
        )

        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")

        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            print(f"\nBatch info:")
            print(f"  Total nodes: {batch.num_nodes}")
            print(f"  Total edges: {batch.num_edges}")
            print(f"  Batch size: {batch.num_graphs}")

        print("\nDataLoader test passed!")

    except Exception as e:
        print(f"Error: {e}")
