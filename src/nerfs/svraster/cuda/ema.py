"""
from __future__ import annotations

Exponential Moving Average (EMA) model implementation.

This module provides EMA functionality for model parameter averaging,
which can help stabilize training and improve model performance.
"""

import copy
import torch
from typing import Dict, Optional

__all__ = ["EMAModel"]


class EMAModel:
    """
    Exponential Moving Average model wrapper.

    Maintains moving averages of model parameters during training.
    This can help produce more stable and better performing models.
    """

    def __init__(
        self, model: torch.nn.Module, decay: float = 0.999, device: Optional[torch.device] = None
    ):
        """
        Initialize EMA model.

        Args:
            model: The model whose parameters we want to maintain averages for
            decay: The decay rate for the moving average (default: 0.999)
            device: The device to store the EMA parameters on
        """
        self.decay = decay
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Deep copy of the model parameters
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach().to(self.device)
                self.original[name] = param.data.clone().detach().to(self.device)

    def update(self, model: torch.nn.Module) -> None:
        """
        Update moving averages of model parameters.

        Args:
            model: The model whose parameters to update the moving averages with
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                    self.shadow[name].copy_(new_average)

    def apply_shadow(self, model: torch.nn.Module) -> None:
        """
        Apply the moving averages to the model parameters.

        Args:
            model: The model whose parameters to update with the moving averages
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name].copy_(param.data)
                param.data.copy_(self.shadow[name])

    def restore_original(self, model: torch.nn.Module) -> None:
        """
        Restore the original model parameters.

        Args:
            model: The model whose parameters to restore
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.original
                param.data.copy_(self.original[name])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get the state dictionary of the EMA model."""
        return {
            "decay": self.decay,
            "shadow": copy.deepcopy(self.shadow),
            "original": copy.deepcopy(self.original),
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Load a state dictionary into the EMA model.

        Args:
            state_dict: The state dictionary to load
        """
        self.decay = state_dict["decay"]
        self.shadow = copy.deepcopy(state_dict["shadow"])
        self.original = copy.deepcopy(state_dict["original"])
