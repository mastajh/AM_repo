"""
Main entry point for GNN-based melt pool simulation.

Usage:
    Training:
        python main.py --mode train --config configs/config.yaml

    Inference:
        python main.py --mode inference --checkpoint checkpoints/best_model.pt --input initial_state.pt --steps 500

    Data generation:
        python main.py --mode generate_data --num_sequences 100
"""

import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from models import build_meshgraphnet
from training import (
    MeltPoolDataset,
    SyntheticMeltPoolDataset,
    collate_fn,
    build_loss,
    Trainer,
    build_optimizer,
    build_scheduler
)
from inference import run_rollout, visualize_trajectory_3d, plot_average_fields
from data.generate_data import generate_dataset


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train(config: dict, resume_from: str = None):
    """
    Train the model.

    Args:
        config: Configuration dictionary
        resume_from: Optional checkpoint to resume from
    """
    print("=" * 60)
    print("Training MeshGraphNet for Melt Pool Simulation")
    print("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model
    print("\nBuilding model...")
    model = build_meshgraphnet(config)
    model = model.to(device)

    model_info = model.get_model_size()
    print(f"Model parameters: {model_info['total']:,}")
    print(f"Model size: {model_info['total_mb']:.2f} MB")

    # Build datasets
    print("\nLoading datasets...")
    data_config = config.get('data', {})

    # Check if processed data exists
    data_dir = Path(data_config.get('data_dir', 'data/processed'))
    use_synthetic = not (data_dir / 'train').exists()

    if use_synthetic:
        print("Using synthetic dataset (no processed data found)")
        train_dataset = SyntheticMeltPoolDataset(
            num_sequences=data_config.get('train_sequences', 100),
            sequence_length=data_config.get('sequence_length', 200),
            num_nodes=1000,
            noise_std=config['training'].get('noise_std', 0.003),
            dt=data_config.get('dt', 1e-5)
        )
        val_dataset = SyntheticMeltPoolDataset(
            num_sequences=data_config.get('val_sequences', 20),
            sequence_length=data_config.get('sequence_length', 200),
            num_nodes=1000,
            noise_std=0.0,  # No noise for validation
            dt=data_config.get('dt', 1e-5)
        )
    else:
        print("Using processed dataset")
        train_dataset = MeltPoolDataset(
            data_dir=str(data_dir),
            split='train',
            noise_std=config['training'].get('noise_std', 0.003),
            normalize=True,
            sequence_length=data_config.get('sequence_length', 200),
            dt=data_config.get('dt', 1e-5)
        )
        val_dataset = MeltPoolDataset(
            data_dir=str(data_dir),
            split='val',
            noise_std=0.0,
            normalize=True,
            sequence_length=data_config.get('sequence_length', 200),
            dt=data_config.get('dt', 1e-5)
        )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training'].get('batch_size', 4),
        shuffle=True,
        num_workers=0,  # Use 0 for debugging, increase for production
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training'].get('batch_size', 4),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # Build loss
    print("\nBuilding loss function...")
    loss_fn = build_loss(config)

    # Build optimizer and scheduler
    print("Building optimizer...")
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # Build trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )

    # Resume from checkpoint if provided
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)

    # Train
    print("\nStarting training...")
    print("=" * 60)
    trainer.train()

    print("\nTraining completed!")


def inference(
    config: dict,
    checkpoint_path: str,
    initial_state_path: str,
    num_steps: int = 500
):
    """
    Run inference (rollout prediction).

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
        initial_state_path: Path to initial state
        num_steps: Number of rollout steps
    """
    print("=" * 60)
    print("Running Inference")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Run rollout
    print(f"\nRunning rollout for {num_steps} steps...")
    trajectory = run_rollout(
        checkpoint_path=checkpoint_path,
        initial_state_path=initial_state_path,
        config=config,
        num_steps=num_steps,
        device=device
    )

    print(f"\nGenerated {len(trajectory)} frames")

    # Visualize
    print("\nGenerating visualizations...")
    output_dir = config['inference'].get('output_dir', 'outputs')

    # Plot average fields
    plot_average_fields(
        trajectory,
        save_path=f'{output_dir}/average_fields.png'
    )

    # Optional: Generate 3D visualization frames
    # visualize_trajectory_3d(
    #     trajectory,
    #     field='temperature',
    #     output_dir=f'{output_dir}/frames',
    #     fps=30
    # )

    print("\nInference completed!")


def generate_data_main(
    num_sequences: int = 100,
    sequence_length: int = 200,
    simulator_type: str = 'droplet'
):
    """
    Generate synthetic training data.

    Args:
        num_sequences: Number of sequences to generate
        sequence_length: Length of each sequence
        simulator_type: Type of simulator ('droplet' or 'meltpool')
    """
    print("=" * 60)
    print("Generating Synthetic Data")
    print("=" * 60)

    # Generate training data
    print("\nGenerating training data...")
    generate_dataset(
        output_dir='data/processed/train',
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        simulator_type=simulator_type
    )

    # Generate validation data
    print("\nGenerating validation data...")
    generate_dataset(
        output_dir='data/processed/val',
        num_sequences=num_sequences // 5,
        sequence_length=sequence_length,
        simulator_type=simulator_type
    )

    # Generate test data
    print("\nGenerating test data...")
    generate_dataset(
        output_dir='data/processed/test',
        num_sequences=num_sequences // 10,
        sequence_length=sequence_length,
        simulator_type=simulator_type
    )

    print("\nData generation completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='GNN-based Melt Pool Simulation'
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'inference', 'generate_data'],
        help='Mode to run'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (for inference or resuming training)'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Path to initial state (for inference)'
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=500,
        help='Number of rollout steps (for inference)'
    )

    parser.add_argument(
        '--num_sequences',
        type=int,
        default=100,
        help='Number of sequences to generate (for data generation)'
    )

    parser.add_argument(
        '--sequence_length',
        type=int,
        default=200,
        help='Length of each sequence (for data generation)'
    )

    parser.add_argument(
        '--simulator',
        type=str,
        default='droplet',
        choices=['droplet', 'meltpool'],
        help='Simulator type (for data generation)'
    )

    args = parser.parse_args()

    # Load config
    if args.mode != 'generate_data':
        config = load_config(args.config)
    else:
        config = None

    # Run mode
    if args.mode == 'train':
        train(config, resume_from=args.checkpoint)

    elif args.mode == 'inference':
        if not args.checkpoint:
            raise ValueError("--checkpoint required for inference mode")
        if not args.input:
            raise ValueError("--input required for inference mode")

        inference(
            config=config,
            checkpoint_path=args.checkpoint,
            initial_state_path=args.input,
            num_steps=args.steps
        )

    elif args.mode == 'generate_data':
        generate_data_main(
            num_sequences=args.num_sequences,
            sequence_length=args.sequence_length,
            simulator_type=args.simulator
        )


if __name__ == '__main__':
    main()
