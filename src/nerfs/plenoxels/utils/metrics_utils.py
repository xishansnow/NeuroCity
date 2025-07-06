"""
from __future__ import annotations

Metrics utilities for evaluating Plenoxels and other NeRF models.

This module provides standard metrics used in NeRF evaluation including:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
"""

from typing import Dict, Optional, Union
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity
import lpips
import logging

logger = logging.getLogger(__name__)


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute Peak Signal-to-Noise Ratio between predicted and target images.

    Args:
        pred: Predicted image tensor of shape [H, W, 3] or [B, H, W, 3]
        target: Ground truth image tensor of same shape as pred
        mask: Optional mask tensor of shape [H, W] or [B, H, W]

    Returns:
        PSNR value as float
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred shape {pred.shape} != target shape {target.shape}")

    # Handle batch dimension
    if pred.dim() == 4:
        return (
            torch.stack(
                [
                    compute_psnr(p, t, None if mask is None else m)
                    for p, t, m in zip(
                        pred, target, mask if mask is not None else [None] * len(pred)
                    )
                ]
            )
            .mean()
            .item()
        )

    # Move to CPU for numpy operations
    pred = pred.detach().cpu()
    target = target.detach().cpu()

    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return (-10 * torch.log10(mse)).item()


def compute_ssim(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> float:
    """Compute Structural Similarity Index between predicted and target images.

    Args:
        pred: Predicted image tensor/array of shape [H, W, 3] or [B, H, W, 3]
        target: Ground truth image tensor/array of same shape as pred
        mask: Optional mask tensor/array of shape [H, W] or [B, H, W]

    Returns:
        SSIM value as float
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred shape {pred.shape} != target shape {target.shape}")

    # Handle batch dimension
    if pred.ndim == 4:
        return np.mean(
            [
                compute_ssim(p, t, None if mask is None else m)
                for p, t, m in zip(pred, target, mask if mask is not None else [None] * len(pred))
            ]
        )

    # Apply mask if provided
    if mask is not None:
        pred = pred * mask[..., None]
        target = target * mask[..., None]

    return structural_similarity(pred, target, channel_axis=-1, data_range=1.0)


class LPIPSMetric:
    """LPIPS metric computer using pretrained networks."""

    def __init__(self, net_type: str = "alex", device: Optional[torch.device] = None):
        """Initialize LPIPS metric computer.

        Args:
            net_type: Network type ('alex', 'vgg', or 'squeeze')
            device: Torch device to use
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net_type).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute LPIPS distance between predicted and target images.

        Args:
            pred: Predicted image tensor of shape [H, W, 3] or [B, H, W, 3]
            target: Ground truth image tensor of same shape as pred
            mask: Optional mask tensor of shape [H, W] or [B, H, W]

        Returns:
            LPIPS distance as float
        """
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred shape {pred.shape} != target shape {target.shape}"
            )

        # Handle batch dimension
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        # Move to device
        pred = pred.to(self.device)
        target = target.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # LPIPS expects [B, C, H, W] format
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

        # Apply mask if provided
        if mask is not None:
            pred = pred * mask.unsqueeze(1)
            target = target * mask.unsqueeze(1)

        # Compute distance
        dist = self.model(pred, target)
        return dist.mean().item()


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    net_type: str = "alex",
    device: Optional[torch.device] = None,
) -> float:
    """Compute LPIPS distance between predicted and target images.

    Args:
        pred: Predicted image tensor of shape [H, W, 3] or [B, H, W, 3]
        target: Ground truth image tensor of same shape as pred
        mask: Optional mask tensor of shape [H, W] or [B, H, W]
        net_type: Network type ('alex', 'vgg', or 'squeeze')
        device: Torch device to use

    Returns:
        LPIPS distance as float
    """
    lpips_fn = LPIPSMetric(net_type=net_type, device=device)
    return lpips_fn(pred, target, mask)


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    lpips_net_type: str = "alex",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Compute all available metrics between predicted and target images.

    Args:
        pred: Predicted image tensor of shape [H, W, 3] or [B, H, W, 3]
        target: Ground truth image tensor of same shape as pred
        mask: Optional mask tensor of shape [H, W] or [B, H, W]
        lpips_net_type: Network type for LPIPS ('alex', 'vgg', or 'squeeze')
        device: Torch device to use

    Returns:
        Dictionary containing computed metrics
    """
    metrics = {
        "psnr": compute_psnr(pred, target, mask),
        "ssim": compute_ssim(pred, target, mask),
        "lpips": compute_lpips(pred, target, mask, lpips_net_type, device),
    }
    return metrics
