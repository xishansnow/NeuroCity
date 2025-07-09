"""
from __future__ import annotations

Utility functions for Classic NeRF.
"""

import numpy as np
import torch
import torch.nn.functional as F
import imageio
import os
from typing import Any


def to8b(x: np.ndarray) -> np.ndarray:
    """Convert to 8-bit image."""
    return (255*np.clip(x, 0, 1)).astype(np.uint8)


def img2mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute mean squared error between images."""
    return torch.mean((x - y) ** 2)


def mse2psnr(mse: torch.Tensor) -> torch.Tensor:
    """Convert MSE to PSNR."""
    return -10. * torch.log10(mse)


def get_rays(
    H: int,
    W: int,
    K: torch.Tensor,
    c2w: torch.Tensor
):
    """Get ray origins and directions from camera parameters."""
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij'
    )
    i = i.t()
    j = j.t()
    
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def render_path(
    render_poses: torch.Tensor,
    hwf: tuple[int, int,float],
    K: torch.Tensor,
    chunk: int,
    render_kwargs: dict[str,Any],
    savedir: str | None = None
):
    """Render images along a path."""
    H, W, focal = hwf
    
    rgbs = []
    disps = []
    
    for i, c2w in enumerate(render_poses):
        rays_o, rays_d = get_rays(H, W, K, c2w)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        
        rgb, disp, acc, extras = render(
            H,
            W,
            K,
            chunk=chunk,
            rays=torch.cat,
        )
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        
        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, f'{i:03d}.png')
            imageio.imwrite(filename, rgb8)
    
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    
    return rgbs, disps


def render(
    H: int,
    W: int,
    K: torch.Tensor,
    chunk: int = 1024*32,
    rays: torch.Tensor | None = None,
    c2w: torch.Tensor | None = None,
    ndc: bool = True,
    near: float = 0.,
    far: float = 1.,
    use_viewdirs: bool = False,
    c2w_staticcam: torch.Tensor | None = None,
    **kwargs
):
    """Render rays."""
    if c2w is not None:
        # Special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # Use provided ray batch
        rays_o, rays_d = rays[:, :3], rays[:, 3:6]
    
    if use_viewdirs:
        # Provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # Special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # For forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def batchify_rays(
    rays_flat: torch.Tensor,
    chunk: int = 1024*32,
    **kwargs
):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def ndc_rays(
    H: int,
    W: int,
    K: torch.Tensor,
    near: float,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor
):
    """Normalized device coordinate rays."""
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*K[0][0])) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*K[1][1])) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*K[0][0])) * (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*K[1][1])) * (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]
    
    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)
    
    return rays_o, rays_d


def create_spherical_poses(radius: float = 4.0, n_poses: int = 120) -> torch.Tensor:
    """Create spherical camera poses for rendering."""
    def trans_t(t: float) -> torch.Tensor:
        return torch.tensor([
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]
        ], dtype=torch.float32)

    def rot_phi(phi: float) -> torch.Tensor:
        return torch.tensor([
            [1, 0, 0, 0], [0, np.cos(phi), -np.sin(phi), 0], [0, np.sin(phi), np.cos(phi), 0], [0, 0, 0, 1]], dtype=torch.float32)

    def rot_theta(th: float) -> torch.Tensor:
        return torch.tensor([
            [np.cos(th), 0, -np.sin(th), 0], [0, 1, 0, 0], [np.sin(th), 0, np.cos(th), 0], [0, 0, 0, 1]], dtype=torch.float32)

    def pose_spherical(theta: float, phi: float, radius: float) -> torch.Tensor:
        c2w = trans_t(radius)
        c2w = rot_phi(phi / 180. * np.pi) @ c2w
        c2w = rot_theta(theta / 180. * np.pi) @ c2w
        c2w = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w

    render_poses = []
    for theta in np.linspace(-180, 180, n_poses+1)[:-1]:
        render_poses.append(pose_spherical(theta, -30.0, radius))
    
    return torch.stack(render_poses, 0)


def visualize_depth(depth: np.ndarray, near: float = 2.0, far: float = 6.0) -> np.ndarray:
    """Visualize depth map."""
    depth_vis = (depth - near) / (far - near)
    depth_vis = np.clip(depth_vis, 0, 1)
    return depth_vis


def compute_ssim(img1: torch.Tensor | np.ndarray, img2: torch.Tensor | np.ndarray) -> float:
    """Compute SSIM between two images."""
    # Convert to tensors if needed
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()
    
    # Add batch and channel dimensions if needed
    if len(img1.shape) == 3:
        img1 = img1.permute(2, 0, 1).unsqueeze(0)
    if len(img2.shape) == 3:
        img2 = img2.permute(2, 0, 1).unsqueeze(0)
    
    # SSIM computation
    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def save_image_grid(images: torch.Tensor | np.ndarray, save_path: str, nrow: int = 4) -> None:
    """Save a grid of images."""
    from torchvision.utils import make_grid
    
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    
    # Rearrange to [B, C, H, W]
    if len(images.shape) == 4 and images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)
    
    grid = make_grid(images, nrow=nrow, normalize=True)
    grid_np = grid.permute(1, 2, 0).numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    
    imageio.imwrite(save_path, grid_np)


def load_config_from_args(args: Any) -> Any:
    """Load NeRF config from command line arguments."""
    from .core import NeRFConfig
    
    config = NeRFConfig()
    
    # Update config from args
    for key, value in vars(args).items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config
