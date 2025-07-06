"""
from __future__ import annotations

Grid-NeRF Training Utilities

This module provides training utilities for Grid-NeRF:
- Logging setup
- Configuration management
- Learning rate scheduling
"""

import logging
import yaml
import json
from pathlib import Path
from collections.abc import Mapping
import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from typing import Any, Optional, Union
import math


def setup_logging(log_dir: str | Path, rank: int = 0) -> None:
    """Set up logging configuration.

    Args:
        log_dir: Directory to save log files
        rank: Process rank for distributed training
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Only setup logging for rank 0 process
    if rank == 0:
        log_file = log_dir / "train.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )


def load_config(config_path: str | Path) -> Mapping[str, Any]:
    """Load configuration from file.

    Args:
        config_path: Path to config file (yaml or json)

    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        if config_path.suffix == ".yaml":
            config = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    return config


def save_config(config: Mapping[str, Any], save_path: str | Path) -> None:
    """Save configuration to file.

    Args:
        config: Configuration dictionary
        save_path: Path to save config file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        if save_path.suffix == ".yaml":
            yaml.dump(config, f, default_flow_style=False)
        elif save_path.suffix == ".json":
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {save_path.suffix}")


class CosineAnnealingWarmRestarts(_LRScheduler):
    """Cosine Annealing with Warm Restarts scheduler."""

    def __init__(
        self, optimizer, T_0: int, T_mult: int = 1, eta_min: float = 0, last_epoch: int = -1
    ):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + torch.cos(torch.tensor(np.pi * self.T_cur / self.T_i)))
            / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(np.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


def get_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    max_steps: int = 50000,
    warmup_steps: int = 500,
    **kwargs,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Get learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ("cosine", "step", "exponential")
        max_steps: Maximum number of training steps
        warmup_steps: Number of warmup steps
        **kwargs: Additional arguments for specific schedulers

    Returns:
        Optional[torch.optim.lr_scheduler._LRScheduler]: Learning rate scheduler
    """
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_steps - warmup_steps,
            eta_min=kwargs.get("eta_min", 0.0),
        )
    elif scheduler_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 10000),
            gamma=kwargs.get("gamma", 0.1),
        )
    elif scheduler_type == "exponential":
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get("gamma", 0.95),
        )
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}")
        return None
