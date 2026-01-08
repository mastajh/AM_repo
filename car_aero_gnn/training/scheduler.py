"""
Learning Rate Schedulers
"""

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    ExponentialLR
)


def create_scheduler(optimizer, config):
    """
    Create learning rate scheduler from config

    Args:
        optimizer: PyTorch optimizer
        config: Configuration dict

    Returns:
        scheduler: Learning rate scheduler
    """
    scheduler_type = config.get('scheduler', 'cosine')

    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.get('epochs', 500),
            eta_min=config.get('min_lr', 1e-6)
        )

    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('lr_decay_factor', 0.5),
            patience=config.get('lr_patience', 10),
            min_lr=config.get('min_lr', 1e-6)
        )

    elif scheduler_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=config.get('lr_decay_steps', 50),
            gamma=config.get('lr_decay_factor', 0.5)
        )

    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(
            optimizer,
            gamma=config.get('lr_decay_factor', 0.95)
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


class WarmupScheduler:
    """
    Learning rate warmup scheduler

    Linearly increases learning rate from 0 to initial_lr over warmup_steps

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        initial_lr: Target learning rate after warmup
    """

    def __init__(self, optimizer, warmup_steps, initial_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.current_step = 0

    def step(self):
        """Update learning rate"""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            lr = self.initial_lr * self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr


class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing scheduler

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        initial_lr: Peak learning rate
        min_lr: Minimum learning rate
    """

    def __init__(
        self,
        optimizer,
        warmup_steps,
        total_steps,
        initial_lr,
        min_lr=1e-6
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        """Update learning rate"""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Warmup phase
            lr = self.initial_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing phase
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (
                1 + torch.cos(torch.tensor(progress * 3.14159))
            )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = float(lr)

    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


if __name__ == '__main__':
    # Test schedulers
    import matplotlib.pyplot as plt

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Test warmup + cosine scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=100,
        total_steps=1000,
        initial_lr=1e-3,
        min_lr=1e-6
    )

    lrs = []
    for step in range(1000):
        scheduler.step()
        lrs.append(scheduler.get_lr())

    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Warmup + Cosine Annealing')
    plt.grid(True)
    plt.savefig('scheduler_test.png')
    print("Scheduler test plot saved to scheduler_test.png")
