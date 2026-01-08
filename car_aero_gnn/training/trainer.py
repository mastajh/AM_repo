"""
Training Loop for MeshGraphNet

Includes:
- Training and validation loops
- Checkpointing
- Logging (WandB support)
- Early stopping
"""

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from pathlib import Path
import json
from tqdm import tqdm
import time

from .losses import create_loss_function
from .scheduler import create_scheduler


try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Logging disabled.")


class Trainer:
    """
    Trainer for MeshGraphNet

    Args:
        model: MeshGraphNet model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Training configuration dict
        device: Device to train on
        use_wandb: Whether to use Weights & Biases logging
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        device='cuda',
        use_wandb=False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb and HAS_WANDB

        # Optimizer
        optimizer_type = config.get('optimizer', 'adam')
        lr = config.get('learning_rate', 1e-4)
        weight_decay = config.get('weight_decay', 0.0)

        if optimizer_type == 'adam':
            self.optimizer = Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            self.optimizer = AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Scheduler
        self.scheduler = create_scheduler(self.optimizer, config)

        # Loss function
        self.loss_fn = create_loss_function(config)

        # Gradient clipping
        self.grad_clip = config.get('grad_clip', 1.0)

        # Noise augmentation
        self.noise_std = config.get('noise_std', 0.003)

        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Best model tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.early_stop_patience = config.get('early_stop_patience', 50)

        # Initialize wandb
        if self.use_wandb:
            wandb.init(
                project=config.get('wandb_project', 'car_aero_gnn'),
                config=config,
                name=config.get('experiment_name', None)
            )
            wandb.watch(model, log='all', log_freq=100)

        # Training state
        self.current_epoch = 0
        self.global_step = 0

    def add_noise(self, data):
        """
        Add noise to velocity features for training stability

        Args:
            data: Batch data

        Returns:
            Noisy data
        """
        if self.noise_std > 0:
            # Add noise to velocity features (indices 3:6)
            noise = torch.randn_like(data.x[:, 3:6]) * self.noise_std
            data.x[:, 3:6] = data.x[:, 3:6] + noise

        return data

    def train_epoch(self):
        """
        Train for one epoch

        Returns:
            avg_loss: Average training loss
            loss_dict: Dictionary with loss components
        """
        self.model.train()

        total_loss = 0
        total_loss_dict = {}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')

        for batch in pbar:
            batch = batch.to(self.device)

            # Add noise augmentation
            batch = self.add_noise(batch)

            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(batch)

            # Compute loss
            loss, loss_dict = self.loss_fn(pred, batch.y, batch)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip
                )

            self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in total_loss_dict:
                    total_loss_dict[key] = 0
                total_loss_dict[key] += value

            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.6f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # Log to wandb
            if self.use_wandb and self.global_step % 10 == 0:
                wandb.log({
                    'train/step_loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/step': self.global_step
                })

        # Compute averages
        avg_loss = total_loss / num_batches
        for key in total_loss_dict:
            total_loss_dict[key] /= num_batches

        return avg_loss, total_loss_dict

    @torch.no_grad()
    def validate(self):
        """
        Validate on validation set

        Returns:
            avg_loss: Average validation loss
            loss_dict: Dictionary with loss components
        """
        self.model.eval()

        total_loss = 0
        total_loss_dict = {}
        num_batches = 0

        for batch in tqdm(self.val_loader, desc='Validation'):
            batch = batch.to(self.device)

            # Forward pass
            pred = self.model(batch)

            # Compute loss
            loss, loss_dict = self.loss_fn(pred, batch.y, batch)

            # Accumulate losses
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in total_loss_dict:
                    total_loss_dict[key] = 0
                total_loss_dict[key] += value

            num_batches += 1

        # Compute averages
        avg_loss = total_loss / num_batches
        for key in total_loss_dict:
            total_loss_dict[key] /= num_batches

        return avg_loss, total_loss_dict

    def save_checkpoint(self, filename='checkpoint.pt', is_best=False):
        """
        Save training checkpoint

        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint to resume training

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Resumed from epoch {self.current_epoch}")

    def train(self, num_epochs):
        """
        Main training loop

        Args:
            num_epochs: Number of epochs to train
        """
        print("=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print("=" * 70)

        start_time = time.time()

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_loss_dict = self.train_epoch()

            # Validate
            val_loss, val_loss_dict = self.validate()

            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Print progress
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.2e}")

            # Log to wandb
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                }
                for key, value in train_loss_dict.items():
                    log_dict[f'train/{key}'] = value
                for key, value in val_loss_dict.items():
                    log_dict[f'val/{key}'] = value

                wandb.log(log_dict)

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(is_best=True)
                print(f"  New best validation loss: {val_loss:.6f}")
            else:
                self.epochs_without_improvement += 1

            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

            # Early stopping
            if self.epochs_without_improvement >= self.early_stop_patience:
                print(f"\nEarly stopping after {self.early_stop_patience} epochs without improvement")
                break

        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"Total time: {elapsed_time / 3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Final checkpoint saved to {self.checkpoint_dir}")

        if self.use_wandb:
            wandb.finish()


def create_trainer(model, train_loader, val_loader, config, device='cuda'):
    """
    Factory function to create trainer

    Args:
        model: MeshGraphNet model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Training configuration
        device: Device to train on

    Returns:
        trainer: Trainer instance
    """
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        use_wandb=config.get('use_wandb', False)
    )

    return trainer


if __name__ == '__main__':
    print("Trainer module ready")
    print("Use create_trainer() to instantiate a trainer")
