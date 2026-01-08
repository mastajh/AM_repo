# GNN-Based Melt Pool & Free Surface Simulation

Physics-informed Graph Neural Network surrogate model for predicting melt pool dynamics and free surface droplet behavior in metal additive manufacturing.

## Overview

This project implements a **MeshGraphNet**-based model for simulating:
- Metal melt pool dynamics during laser-based additive manufacturing
- Free surface droplet impact and spreading
- Thermal evolution with phase change (solid/liquid/gas)
- Level-set interface tracking

The model combines **data-driven learning** with **physics-informed constraints** including:
- Mass conservation (continuity equation)
- Momentum conservation (simplified Navier-Stokes)
- Energy conservation
- Free surface tracking (VOF/Level-set)
- Surface tension effects
- Heat transfer and phase change

## Features

- **MeshGraphNet Architecture**: Encode-Process-Decode with message passing
- **Physics-Informed Losses**: Conservation laws and physical constraints
- **Adaptive Remeshing**: Dynamic refinement near interfaces
- **Autoregressive Rollout**: Multi-step prediction
- **Synthetic Data Generation**: Test without full CFD solver
- **3D Visualization**: PyVista-based rendering

## Project Structure

```
gnn_meltpool/
├── data/
│   ├── raw/              # LBM/CFD simulation raw data
│   ├── processed/        # Preprocessed graph data
│   └── generate_data.py  # Synthetic data generator
├── models/
│   ├── encoder.py        # Node/edge encoders
│   ├── processor.py      # Message passing GNN
│   ├── decoder.py        # Output decoder
│   └── meshgraphnet.py   # Complete model
├── physics/
│   ├── free_surface.py   # VOF/Level-set physics
│   ├── thermal.py        # Heat transfer
│   └── constraints.py    # Conservation laws
├── training/
│   ├── dataset.py        # PyG dataset
│   ├── trainer.py        # Training loop
│   └── losses.py         # Physics-informed losses
├── inference/
│   ├── rollout.py        # Autoregressive prediction
│   └── visualize.py      # Result visualization
├── configs/
│   └── config.yaml       # Hyperparameters
├── main.py               # Entry point
└── requirements.txt
```

## Installation

### Requirements
- Python >= 3.8
- PyTorch >= 2.0
- PyTorch Geometric
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
cd gnn_meltpool

# Install dependencies
pip install -r requirements.txt

# For PyTorch Geometric, follow installation guide:
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

## Quick Start

### 1. Generate Synthetic Data

```bash
python main.py --mode generate_data --num_sequences 100 --simulator droplet
```

Options:
- `--simulator`: `droplet` (droplet impact) or `meltpool` (laser melting)
- `--num_sequences`: Number of training sequences
- `--sequence_length`: Timesteps per sequence

### 2. Train Model

```bash
python main.py --mode train --config configs/config.yaml
```

The model will:
- Load data from `data/processed/`
- Train with physics-informed losses
- Save checkpoints to `checkpoints/`
- Log to TensorBoard (default) or Weights & Biases

Monitor training:
```bash
tensorboard --logdir logs
```

### 3. Run Inference

```bash
python main.py --mode inference \
    --checkpoint checkpoints/best_model.pt \
    --input data/initial_state.pt \
    --steps 500
```

Output:
- Trajectory saved to `outputs/rollout_trajectory.pt`
- Visualization plots in `outputs/`

### 4. Visualize Results

```python
from inference import visualize_trajectory_3d, plot_average_fields

# Load trajectory
trajectory = torch.load('outputs/rollout_trajectory.pt')

# 3D visualization
visualize_trajectory_3d(trajectory, field='temperature', output_dir='outputs/frames')

# Plot statistics
plot_average_fields(trajectory, save_path='outputs/avg_fields.png')
```

## Configuration

Edit `configs/config.yaml` to customize:

### Model Architecture
```yaml
model:
  latent_dim: 128           # Latent feature dimension
  num_message_passing: 15   # Number of GNN layers
  hidden_dim: 128           # MLP hidden dimension
```

### Training
```yaml
training:
  batch_size: 4
  learning_rate: 1.0e-4
  epochs: 500
  noise_std: 0.003          # Input noise for robustness
```

### Physics Weights
```yaml
physics:
  use_physics_loss: true
  mass_weight: 0.1          # Mass conservation
  surface_weight: 0.1       # Free surface
  energy_weight: 0.01       # Energy conservation
  curvature_weight: 0.01    # Interface smoothness
```

### Data
```yaml
data:
  sequence_length: 200
  dt: 1.0e-5               # Time step (seconds)
  world_edge_radius: 0.1   # Edge connectivity radius
```

## Model Architecture

### Graph Representation

**Nodes** (particles/cells):
- Position: `(x, y, z)`
- Temperature: `T`
- Phase: `0=solid, 1=liquid, 2=gas`
- Velocity: `(u, v, w)`
- Level-set: `φ` (interface tracking)
- Type: `interior/boundary/free_surface`

**Edges**:
- Relative position: `(dx, dy, dz)`
- Distance: `||dx||`
- Connectivity: mesh-space + world-space

### MeshGraphNet

```
Input Graph
    ↓
[Encoder] → Latent features
    ↓
[Processor] → 15 message passing layers
    ↓
[Decoder] → State changes (Δx)
    ↓
Output: Next state
```

### Physics-Informed Losses

**Total Loss**:
```
L = L_data + α·L_mass + β·L_momentum + γ·L_energy + δ·L_surface + ε·L_curv
```

- **L_data**: MSE between prediction and target
- **L_mass**: Mass conservation (∇·(ρu) = 0)
- **L_momentum**: Simplified Navier-Stokes
- **L_energy**: Energy conservation
- **L_surface**: Interface tracking (|∇φ| ≈ 1)
- **L_curv**: Curvature regularization

## Advanced Usage

### Custom Data Pipeline

```python
from training import MeltPoolDataset
from torch.utils.data import DataLoader

# Load your own data
dataset = MeltPoolDataset(
    data_dir='path/to/data',
    split='train',
    normalize=True
)

loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
```

### Custom Physics Constraints

```python
from physics import PhysicsInformedLoss

# Add custom loss
class CustomLoss(nn.Module):
    def forward(self, pred, target, data):
        # Your physics constraint
        return loss, loss_dict

# Use in training
loss_fn = CustomLoss(config)
```

### Adaptive Remeshing

```python
from inference import Rollout

rollout = Rollout(model, config, device)
rollout.remesh_interval = 50      # Remesh every 50 steps
rollout.remesh_threshold = 0.5    # Level-set gradient threshold
```

## Performance

Typical performance on synthetic data:
- **Training time**: ~12 hours (500 epochs, 100 sequences, GPU)
- **Inference speed**: ~0.1s per step (1000 nodes, GPU)
- **Model size**: ~5M parameters (~20 MB)

## Citation

This implementation is based on:

```bibtex
@inproceedings{pfaff2021learning,
  title={Learning Mesh-Based Simulation with Graph Networks},
  author={Pfaff, Tobias and Fortunato, Meire and Sanchez-Gonzalez, Alvaro and Battaglia, Peter},
  booktitle={ICML},
  year={2021}
}
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Reduce `num_nodes` in synthetic data
- Use gradient checkpointing (advanced)

### Poor Convergence
- Check physics loss weights
- Increase `noise_std` for regularization
- Reduce learning rate
- Use more training data

### Unstable Rollout
- Enable physics corrections: `use_physics_correction=True`
- Reduce rollout timestep `dt`
- Apply stronger constraints

## Contributing

Contributions welcome! Areas for improvement:
- Full adaptive remeshing implementation
- Multi-GPU training
- Real CFD data integration
- Advanced physics models (recoil pressure, evaporation)

## License

MIT License - see LICENSE file

## Contact

For questions and feedback, please open an issue on GitHub.
