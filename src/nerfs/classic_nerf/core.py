from __future__ import annotations
from typing import Any, Callable, Optional, TypeVar, Union

"""
Classic NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.

This module implements the original NeRF model as described in:
"NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
by Ben Mildenhall et al. (ECCV 2020)

Key components:
- Positional encoding for high-frequency details
- Multi-layer perceptron (MLP) for scene representation  
- Hierarchical volume sampling
- Volume rendering with neural radiance fields
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass

T = TypeVar("T", bound=torch.Tensor)


@dataclass
class NeRFConfig:
    """Configuration for classic NeRF model."""

    # Network architecture
    netdepth: int = 8  # Depth of the MLP
    netwidth: int = 256  # Width of the MLP
    netdepth_fine: int = 8  # Depth of the fine MLP
    netwidth_fine: int = 256  # Width of the fine MLP

    # Positional encoding
    multires: int = 10  # Positional encoding for coordinates
    multires_views: int = 4  # Positional encoding for viewing directions

    # Sampling
    N_samples: int = 64  # Number of coarse samples
    N_importance: int = 128  # Number of fine samples
    perturb: bool = True  # Whether to perturb sampling
    use_viewdirs: bool = True  # Whether to use viewing directions

    # Rendering
    raw_noise_std: float = 0.0  # Noise added to raw density predictions
    white_bkgd: bool = False  # Whether to use white background

    # Training
    learning_rate: float = 5e-4  # Learning rate
    lrate_decay: int = 250  # Exponential learning rate decay (in 1000s)

    # Scene bounds
    near: float = 2.0  # Near bound for sampling
    far: float = 6.0  # Far bound for sampling

    # Loss weights
    rgb_loss_weight: float = 1.0  # Weight for RGB loss

    # Optimizer
    beta1: float = 0.9  # Adam beta1
    beta2: float = 0.999  # Adam beta2
    epsilon: float = 1e-7  # Adam epsilon


class Embedder(nn.Module):
    """Positional encoding embedder for coordinates and directions."""

    def __init__(
        self,
        input_dims: int,
        max_freq_log2: int,
        num_freqs: int,
        log_sampling: bool = True,
        include_input: bool = True,
        periodic_fns: List[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.include_input = include_input
        self.periodic_fns = [torch.sin, torch.cos] if periodic_fns is None else periodic_fns
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling

        # Create frequency bands
        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, steps=num_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq_log2, steps=num_freqs)

        # Store frequency bands as parameters
        self.freq_bands = nn.ParameterList(
            [nn.Parameter(freq.unsqueeze(0), requires_grad=False) for freq in freq_bands]
        )

        # Calculate output dimension
        self.out_dim = 0
        if self.include_input:
            self.out_dim += input_dims
        self.out_dim += input_dims * num_freqs * len(self.periodic_fns)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to inputs."""
        # Create output tensor
        embeds = []

        # Add raw inputs if requested
        if self.include_input:
            embeds.append(inputs)

        # Add encoded inputs
        for freq_param in self.freq_bands:
            freq = freq_param[0]  # Get scalar value from parameter
            for p_fn in self.periodic_fns:
                embeds.append(p_fn(inputs * freq))

        # Concatenate all embeddings
        return torch.cat(embeds, dim=-1)


def get_embedder(multires: int, input_dims: int = 3) -> Tuple[Embedder | nn.Identity, int]:
    """Get positional encoding embedder."""
    if multires == -1:
        return nn.Identity(), input_dims

    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


class NeRF(nn.Module):
    """Classic NeRF model."""

    def __init__(self, config: NeRFConfig):
        super().__init__()
        self.config = config

        # Get positional encoders
        self.embed_fn, input_ch = get_embedder(config.multires, 3)

        input_ch_views = 0
        self.embeddirs_fn = None
        if config.use_viewdirs:
            self.embeddirs_fn, input_ch_views = get_embedder(config.multires_views, 3)

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        # Create MLP layers
        self.skips = [4]  # Skip connections at layer 4
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, config.netwidth)]
            + [
                (
                    nn.Linear(config.netwidth, config.netwidth)
                    if i not in self.skips
                    else nn.Linear(config.netwidth + input_ch, config.netwidth)
                )
                for i in range(config.netdepth - 1)
            ]
        )

        # View-dependent branch
        if config.use_viewdirs:
            self.feature_linear = nn.Linear(config.netwidth, config.netwidth)
            self.alpha_linear = nn.Linear(config.netwidth, 1)
            self.rgb_linear = nn.Linear(config.netwidth // 2, 3)
            self.views_linears = nn.ModuleList(
                [nn.Linear(input_ch_views + config.netwidth, config.netwidth // 2)]
            )
        else:
            self.output_linear = nn.Linear(config.netwidth, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through NeRF.

        Args:
            x: Input tensor [N, input_ch + input_ch_views] if use_viewdirs, else [N, input_ch]

        Returns:
            outputs: [N, 4] (RGB + density)
        """
        if self.config.use_viewdirs:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            input_pts = x

        # Forward through MLP
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.config.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


def raw2outputs(
    raw: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    raw_noise_std: float = 0.0,
    white_bkgd: bool = False,
    pytest: bool = False,
) -> Dict[str, Any]:
    """
    Transforms model's predictions to semantically meaningful values.

    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
        raw_noise_std: Std dev of noise added to regularize sigma_a output.
        white_bkgd: If True, assume a white background.
        pytest: If True, return also intermediate calculations for testing.

    Returns:
        dict: Dictionary containing rendered outputs
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw) * dists)

    # Compute distances between adjacent samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists.device)], dim=-1
    )

    # Multiply each distance by the norm of its corresponding direction ray
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Extract RGB and density from raw predictions
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape, device=raw.device) * raw_noise_std

    # Alpha compositing
    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    weights = (
        alpha
        * torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], -1), -1)[
            ..., :-1
        ]
    )

    # Weighted sum for final RGB
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    # Estimated depth map is weighted sum of z_vals
    depth_map = torch.sum(weights * z_vals, -1)

    # Disparity map is inverse depth
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map)

    # Sum of weights along each ray
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    ret = {"rgb_map": rgb_map, "disp_map": disp_map, "acc_map": acc_map}
    if pytest:
        ret["weights"] = weights
        ret["alpha"] = alpha
        ret["z_vals"] = z_vals
        ret["depth_map"] = depth_map

    return ret


def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    N_samples: int,
    det: bool = False,
    pytest: bool = False,
) -> torch.Tensor:
    """
    Sample @N_samples samples from @bins with distribution defined by @weights.

    Args:
        bins: [N_rays, M-1]. where M is the number of bins edges.
        weights: [N_rays, M-2]. where M is the number of bins edges.
        N_samples: Number of samples.
        det: If True, will perform deterministic sampling.
        pytest: If True, use fixed random numbers for testing.

    Returns:
        samples: [N_rays, N_samples] Sampled points.
    """
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # [N_rays, M]

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=weights.device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        u = torch.from_numpy(np.random.rand(*new_shape)).float().to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # [N_rays, N_samples, 2]

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeRFRenderer:
    """Volume renderer for NeRF."""

    def __init__(self, config: NeRFConfig):
        self.config = config

    def run_network(
        self,
        inputs: torch.Tensor,
        viewdirs: torch.Tensor | None,
        fn: NeRF,
    ) -> torch.Tensor:
        """
        Prepare inputs and apply network.

        Args:
            inputs: [N_rays, N_samples, 3] sample coordinates
            viewdirs: [N_rays, 3] viewing directions
            fn: Neural network

        Returns:
            outputs: [N_rays, N_samples, 4] raw predictions
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

        # Get embedded coordinates
        embedded = fn.embed_fn(inputs_flat)

        if viewdirs is not None and fn.embeddirs_fn is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

            # Get embedded directions
            embedded_dirs = fn.embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        outputs_flat = fn(embedded)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs

    def render_rays(
        self,
        ray_batch: Dict[str, torch.Tensor],
        network_fn: NeRF,
        network_fine: NeRF | None = None,
        retraw: bool = False,
        lindisp: bool = False,
        perturb: bool = True,
        N_importance: int = 0,
        raw_noise_std: float = 0.0,
        verbose: bool = False,
        pytest: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Volumetric rendering.

        Args:
            ray_batch: dict with ray information including:
                rays_o: [N_rays, 3] ray origins
                rays_d: [N_rays, 3] ray directions
                viewdirs: [N_rays, 3] ray viewing directions
                near: [N_rays] near bounds
                far: [N_rays] far bounds
            network_fn: Model for coarse network
            network_fine: Model for fine network
            retraw: If True, include raw output
            lindisp: If True, sample linearly in inverse depth
            perturb: If True, perturb sampling points
            N_importance: Number of importance samples
            raw_noise_std: Standard deviation of noise added to raw density
            verbose: If True, print warnings about numerical errors
            pytest: If True, use deterministic random numbers

        Returns:
            dict: Dictionary containing rendered outputs
        """
        N_rays = ray_batch["rays_o"].shape[0]
        rays_o, rays_d = ray_batch["rays_o"], ray_batch["rays_d"]
        viewdirs = ray_batch.get("viewdirs", None)
        bounds = torch.reshape(ray_batch["bounds"], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [N_rays, 1]

        # Sample points along rays
        t_vals = torch.linspace(0.0, 1.0, steps=self.config.N_samples, device=rays_o.device)
        if not lindisp:
            z_vals = near * (1.0 - t_vals) + far * (t_vals)
        else:
            z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

        z_vals = z_vals.expand([N_rays, self.config.N_samples])

        if perturb:
            # Get intervals between samples
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # Stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=z_vals.device)
            z_vals = lower + (upper - lower) * t_rand

        pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        )  # [N_rays, N_samples, 3]

        # Run network
        raw = self.run_network(pts, viewdirs, network_fn)

        ret = raw2outputs(raw, z_vals, rays_d, raw_noise_std, self.config.white_bkgd, pytest=pytest)

        # Hierarchical sampling
        if N_importance > 0:
            ret0 = ret

            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(
                z_vals_mid,
                ret["weights"][..., 1:-1],
                N_importance,
                det=False,
            )
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

            # Run fine network
            run_fn = network_fn if network_fine is None else network_fine
            raw = self.run_network(pts, viewdirs, run_fn)

            ret = raw2outputs(
                raw,
                z_vals,
                rays_d,
                raw_noise_std,
                self.config.white_bkgd,
                pytest=pytest,
            )

            # Store coarse results
            for k in ret0:
                ret[k + "0"] = ret0[k]

        if retraw:
            ret["raw"] = raw

        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and verbose:
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret


def create_nerf(config: NeRFConfig) -> Tuple[NeRF, NeRF | None, Dict[str, Any]]:
    """
    Create NeRF models.

    Args:
        config: Configuration for NeRF models

    Returns:
        tuple: (coarse model, fine model, render kwargs)
    """
    # Create coarse network
    model = NeRF(config)
    grad_vars = list(model.parameters())

    # Create fine network
    model_fine = None
    if config.N_importance > 0:
        model_fine = NeRF(config)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: network_fn(inputs)

    # Create optimizer
    optimizer = torch.optim.Adam(
        params=grad_vars,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
    )

    start = 0
    basedir = "./logs"
    expname = "nerf"

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "perturb": config.perturb,
        "N_importance": config.N_importance,
        "network_fine": model_fine,
        "N_samples": config.N_samples,
        "network_fn": model,
        "use_viewdirs": config.use_viewdirs,
        "white_bkgd": config.white_bkgd,
        "raw_noise_std": config.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0

    return model, model_fine, render_kwargs_train


class NeRFLoss(nn.Module):
    """Loss function for NeRF."""

    def __init__(self, config: NeRFConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss between prediction and target.

        Args:
            pred: Dictionary containing predicted values
            target: Ground truth RGB values

        Returns:
            dict: Dictionary containing loss values
        """
        # RGB loss
        img_loss = torch.mean((pred["rgb_map"] - target) ** 2)
        loss = self.config.rgb_loss_weight * img_loss

        # Add coarse loss if available
        if "rgb_map0" in pred:
            img_loss0 = torch.mean((pred["rgb_map0"] - target) ** 2)
            loss = loss + self.config.rgb_loss_weight * img_loss0

        return {"loss": loss, "img_loss": img_loss}


def create_bungee_nerf(
    num_views: int = 50,
    max_resolution: int = 128,
):
    """Create Bungee NeRF model."""
    # ... existing code ...


def batch_process_coordinates(
    coords_list: List[np.ndarray],
    batch_size: int = 1000,
):
    """Process coordinates in batches."""
    # ... existing code ...


def compute_voxel_bounds(
    voxel_positions: torch.Tensor,
    voxel_sizes: torch.Tensor,
):
    """Compute voxel bounds."""
    # ... existing code ...
