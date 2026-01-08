"""
Main CLI Entry Point for Car Aerodynamics GNN

Commands:
- train: Train a model
- inference: Run inference on test data
- visualize: Visualize results
- benchmark: Benchmark against CFD
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys


def train_command(args):
    """Train model"""
    print("=" * 70)
    print("Training Mode")
    print("=" * 70)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from {args.config}")

    # Import modules
    from models import create_meshgraphnet
    from training import create_datasets, create_dataloaders
    from training.trainer import create_trainer

    # Create datasets
    print("\nLoading datasets...")
    train_ds, val_ds, test_ds = create_datasets(
        root=config['data']['root_dir'],
        mode=config['inference']['mode'],
        use_transforms=True,
        noise_std=config['training']['noise_std']
    )

    print(f"Train: {len(train_ds)} samples")
    print(f"Val: {len(val_ds)} samples")
    print(f"Test: {len(test_ds)} samples")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=config['training']['batch_size'],
        num_workers=args.num_workers
    )

    # Create model
    print("\nCreating model...")
    model = create_meshgraphnet(config['model'])
    model.get_model_summary()

    # Create trainer
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"\nUsing device: {device}")

    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(num_epochs=config['training']['epochs'])


def inference_command(args):
    """Run inference"""
    print("=" * 70)
    print("Inference Mode")
    print("=" * 70)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Import modules
    from inference import load_predictor
    import torch

    # Load predictor
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    predictor = load_predictor(args.checkpoint, device=device)

    # Load input data
    print(f"\nLoading input from {args.input}")
    data = torch.load(args.input)

    # Run prediction
    print("Running inference...")
    velocity, pressure = predictor.predict(data)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    np.savez(
        output_dir / 'predictions.npz',
        velocity=velocity,
        pressure=pressure
    )

    print(f"\nResults saved to {output_dir / 'predictions.npz'}")


def visualize_command(args):
    """Visualize results"""
    print("=" * 70)
    print("Visualization Mode")
    print("=" * 70)

    try:
        import pyvista as pv
    except ImportError:
        print("Error: PyVista not installed. Install with: pip install pyvista")
        return

    from visualization import AeroVisualizer
    import numpy as np

    # Load mesh
    print(f"Loading mesh from {args.mesh}")
    mesh = pv.read(args.mesh)

    # Load results
    print(f"Loading results from {args.results}")
    results = np.load(args.results)
    velocity = results['velocity']
    pressure = results['pressure']

    # Create visualizer
    viz = AeroVisualizer(mesh)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")

    if 'velocity' in args.plots or 'all' in args.plots:
        print("  - Velocity magnitude")
        viz.visualize_velocity_magnitude(
            velocity,
            save_path=output_dir / 'velocity_magnitude.png'
        )

    if 'pressure' in args.plots or 'all' in args.plots:
        print("  - Pressure distribution")
        viz.visualize_pressure(
            pressure,
            save_path=output_dir / 'pressure.png'
        )

    if 'streamlines' in args.plots or 'all' in args.plots:
        print("  - Streamlines")
        from visualization import StreamlineVisualizer
        stream_viz = StreamlineVisualizer(mesh)
        stream_viz.visualize_streamlines(
            velocity,
            save_path=output_dir / 'streamlines.png'
        )

    print(f"\nVisualization saved to {output_dir}")


def benchmark_command(args):
    """Benchmark model"""
    print("=" * 70)
    print("Benchmark Mode")
    print("=" * 70)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    from inference import load_predictor, AeroBenchmark
    from training import create_datasets
    from torch_geometric.loader import DataLoader
    import torch

    # Load predictor
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    predictor = load_predictor(args.checkpoint, device=device)

    # Load test dataset
    print("\nLoading test dataset...")
    _, _, test_ds = create_datasets(
        root=config['data']['root_dir'],
        mode=config['inference']['mode']
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False
    )

    # Create benchmark
    benchmark = AeroBenchmark(predictor, test_loader)

    # Run benchmarks
    report = benchmark.generate_report(args.output)

    print("\nBenchmark complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Car Aerodynamics GNN - End-to-End Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--config', type=str, required=True, help='Config file')
    train_parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    train_parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    train_parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--config', type=str, required=True, help='Config file')
    inference_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    inference_parser.add_argument('--input', type=str, required=True, help='Input graph data')
    inference_parser.add_argument('--output', type=str, default='outputs/', help='Output directory')
    inference_parser.add_argument('--cpu', action='store_true', help='Use CPU')

    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize results')
    viz_parser.add_argument('--mesh', type=str, required=True, help='Mesh file (VTK)')
    viz_parser.add_argument('--results', type=str, required=True, help='Results file (.npz)')
    viz_parser.add_argument('--output', type=str, default='outputs/', help='Output directory')
    viz_parser.add_argument(
        '--plots',
        nargs='+',
        choices=['velocity', 'pressure', 'streamlines', 'all'],
        default=['all'],
        help='Plots to generate'
    )

    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark model')
    bench_parser.add_argument('--config', type=str, required=True, help='Config file')
    bench_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    bench_parser.add_argument('--output', type=str, default='benchmark_report.json', help='Output report')
    bench_parser.add_argument('--cpu', action='store_true', help='Use CPU')

    args = parser.parse_args()

    if args.command == 'train':
        train_command(args)
    elif args.command == 'inference':
        inference_command(args)
    elif args.command == 'visualize':
        visualize_command(args)
    elif args.command == 'benchmark':
        benchmark_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
