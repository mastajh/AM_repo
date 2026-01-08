"""
Training loop and utilities
Handles model training, validation, and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict
import time

from models import MeshGraphNet
from .losses import MeltPoolLoss


class Trainer:
    """
    Trainer class for MeshGraphNet.

    Handles:
    - Training loop
    - Validation
    - Checkpointing
    - Logging
    - Early stopping
    """

    def __init__(
        self,
        model: MeshGraphNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        config: dict,
        device: torch.device
    ):
        """
        Args:
            model: MeshGraphNet model
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # Training config
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 500)
        self.gradient_clip = train_config.get('gradient_clip', 1.0)
        self.validation_interval = train_config.get('validation_interval', 10)
        self.checkpoint_interval = train_config.get('checkpoint_interval', 20)
        self.early_stopping_patience = train_config.get('early_stopping_patience', 50)

        # Logging config
        log_config = config.get('logging', {})
        self.use_tensorboard = log_config.get('use_tensorboard', True)
        self.use_wandb = log_config.get('use_wandb', False)
        self.log_dir = Path(log_config.get('log_dir', 'logs'))

        # Initialize loggers
        self._init_loggers()

        # Checkpointing
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

    def _init_loggers(self):
        """Initialize logging."""
        if self.use_tensorboard:
            self.log_dir.mkdir(exist_ok=True, parents=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None

        if self.use_wandb:
            try:
                import wandb
                log_config = self.config.get('logging', {})
                wandb.init(
                    project=log_config.get('project_name', 'gnn_meltpool'),
                    name=log_config.get('experiment_name', 'baseline'),
                    config=self.config
                )
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed, disabling wandb logging")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of average losses
        """
        self.model.train()

        epoch_losses = {}
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')

        for batch_idx, (data, target) in enumerate(pbar):
            # Move to device
            data = data.to(self.device)
            target = target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(data)

            # Compute loss
            loss, loss_dict = self.loss_fn(pred, target, data)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            # Optimizer step
            self.optimizer.step()

            # Accumulate losses
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Dictionary of average validation losses
        """
        self.model.eval()

        epoch_losses = {}
        num_batches = len(self.val_loader)

        for data, target in tqdm(self.val_loader, desc='Validation'):
            # Move to device
            data = data.to(self.device)
            target = target.to(self.device)

            # Forward pass
            pred = self.model(data)

            # Compute loss
            loss, loss_dict = self.loss_fn(pred, target, data)

            # Accumulate losses
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.epochs} epochs")
        print(f"Model parameters: {self.model.count_parameters():,}")

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Train
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)

            # Log training losses
            self._log_losses(train_losses, 'train')

            # Validation
            if epoch % self.validation_interval == 0:
                val_losses = self.validate()
                self.val_losses.append(val_losses)

                # Log validation losses
                self._log_losses(val_losses, 'val')

                # Check for improvement
                val_loss = val_losses['total_loss']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint('best_model.pt')
                    print(f"New best validation loss: {val_loss:.6f}")
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Checkpoint
            if epoch % self.checkpoint_interval == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total_loss'])
                else:
                    self.scheduler.step()

                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.writer:
                    self.writer.add_scalar('learning_rate', current_lr, epoch)
                if self.wandb:
                    self.wandb.log({'learning_rate': current_lr}, step=epoch)

        print("Training completed!")

        # Save final model
        self._save_checkpoint('final_model.pt')

        # Close loggers
        if self.writer:
            self.writer.close()
        if self.wandb:
            self.wandb.finish()

    def _log_losses(self, losses: Dict[str, float], split: str):
        """
        Log losses to tensorboard and wandb.

        Args:
            losses: Dictionary of losses
            split: 'train' or 'val'
        """
        # Tensorboard
        if self.writer:
            for key, value in losses.items():
                self.writer.add_scalar(f'{split}/{key}', value, self.current_epoch)

        # Wandb
        if self.wandb:
            wandb_dict = {f'{split}/{key}': value for key, value in losses.items()}
            self.wandb.log(wandb_dict, step=self.current_epoch)

        # Print
        loss_str = ', '.join([f'{key}: {value:.6f}' for key, value in losses.items()])
        print(f"[{split.upper()}] Epoch {self.current_epoch}: {loss_str}")

    def _save_checkpoint(self, filename: str):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint: {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from epoch {self.current_epoch}")


def build_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """
    Build optimizer from configuration.

    Args:
        model: Model to optimize
        config: Configuration dictionary

    Returns:
        Optimizer
    """
    train_config = config.get('training', {})
    lr = train_config.get('learning_rate', 1e-4)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    return optimizer


def build_scheduler(
    optimizer: optim.Optimizer,
    config: dict
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Build learning rate scheduler from configuration.

    Args:
        optimizer: Optimizer
        config: Configuration dictionary

    Returns:
        Scheduler or None
    """
    train_config = config.get('training', {})
    scheduler_type = train_config.get('scheduler', 'cosine')

    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config.get('epochs', 500)
        )
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=20
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100,
            gamma=0.5
        )
    else:
        scheduler = None

    return scheduler
