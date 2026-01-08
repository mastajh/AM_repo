"""
Quick start example for GNN melt pool simulation.

This script demonstrates:
1. Generating synthetic data
2. Training a small model
3. Running inference
4. Visualizing results
"""

import torch
import yaml
from pathlib import Path

from models import build_meshgraphnet
from training import SyntheticMeltPoolDataset, collate_fn, build_loss, build_optimizer
from torch.utils.data import DataLoader


def quick_start_example():
    """Run a minimal training example."""

    print("=" * 60)
    print("GNN Melt Pool Simulation - Quick Start Example")
    print("=" * 60)

    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Override for quick demo
    config['training']['epochs'] = 10
    config['training']['batch_size'] = 2
    config['data']['train_sequences'] = 10
    config['data']['val_sequences'] = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 1. Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    train_dataset = SyntheticMeltPoolDataset(
        num_sequences=10,
        sequence_length=50,
        num_nodes=500,
        noise_std=0.003
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    print(f"   Created {len(train_dataset)} training samples")

    # 2. Build model
    print("\n2. Building MeshGraphNet model...")
    model = build_meshgraphnet(config)
    model = model.to(device)

    model_info = model.get_model_size()
    print(f"   Model parameters: {model_info['total']:,}")
    print(f"   Model size: {model_info['total_mb']:.2f} MB")

    # 3. Setup training
    print("\n3. Setting up training...")
    loss_fn = build_loss(config)
    optimizer = build_optimizer(model, config)

    # 4. Train for a few epochs
    print("\n4. Training (demo - 10 epochs)...")
    model.train()

    for epoch in range(10):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(data)
            loss, loss_dict = loss_fn(pred, target, data)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"   Epoch {epoch+1}/10: Loss = {avg_loss:.6f}")

    # 5. Save model
    print("\n5. Saving model...")
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config
    }
    torch.save(checkpoint, 'checkpoints/demo_model.pt')
    print("   Saved to checkpoints/demo_model.pt")

    # 6. Quick inference test
    print("\n6. Testing inference...")
    model.eval()

    # Get a sample
    sample_data, _ = train_dataset[0]
    sample_data = sample_data.to(device)

    with torch.no_grad():
        # Predict 10 steps
        trajectory = [sample_data.clone().cpu()]
        current_data = sample_data

        for step in range(10):
            delta = model(current_data)
            current_data.x[:, 3:9] = current_data.x[:, 3:9] + delta
            trajectory.append(current_data.clone().cpu())

    print(f"   Generated {len(trajectory)} trajectory frames")

    # 7. Plot results
    print("\n7. Plotting results...")
    try:
        import matplotlib.pyplot as plt

        # Extract average temperature over time
        avg_temps = [data.x[:, 3].mean().item() for data in trajectory]

        plt.figure(figsize=(8, 5))
        plt.plot(avg_temps, marker='o')
        plt.xlabel('Time Step')
        plt.ylabel('Average Temperature (K)')
        plt.title('Temperature Evolution (Demo)')
        plt.grid(True)
        plt.savefig('outputs/demo_temperature.png', dpi=150, bbox_inches='tight')
        print("   Saved plot to outputs/demo_temperature.png")

    except Exception as e:
        print(f"   Plotting failed: {e}")

    print("\n" + "=" * 60)
    print("Quick start example completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train full model: python main.py --mode train --config configs/config.yaml")
    print("2. Generate more data: python main.py --mode generate_data --num_sequences 100")
    print("3. Run inference: python main.py --mode inference --checkpoint checkpoints/best_model.pt")
    print()


if __name__ == '__main__':
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)

    # Run example
    quick_start_example()
