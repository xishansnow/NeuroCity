"""
Mesh AutoEncoder for DNMP.

This module implements the mesh autoencoder that encodes mesh shapes into
low-dimensional latent codes and decodes them back to vertex positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class LatentCode(nn.Module):
    """Learnable latent code for mesh shape representation."""

    def __init__(self, latent_dim: int, init_std: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.code = nn.Parameter(torch.randn(latent_dim) * init_std)

    def forward(self) -> torch.Tensor:
        return self.code

    def reset(self, init_std: float = 0.1):
        """Reset latent code to random initialization."""
        with torch.no_grad():
            self.code.data = torch.randn_like(self.code) * init_std


class MeshEncoder(nn.Module):
    """
    Encoder that maps mesh vertices to latent codes.
    Used for pre-training the autoencoder.
    """

    def __init__(
        self,
        input_dim: int = 3,
        latent_dim: int = 128,
        hidden_dims: tuple[int, ...] = (256, 128, 64, 32),
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Build encoder network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                ]
            )
            prev_dim = hidden_dim

        # Final layer to latent space
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)

        # For variational encoding (optional)
        self.use_variational = False
        if self.use_variational:
            self.mu_layer = nn.Linear(prev_dim, latent_dim)
            self.logvar_layer = nn.Linear(prev_dim, latent_dim)

    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Encode mesh vertices to latent code.

        Args:
            vertices: Mesh vertices [B, N, 3] or [N, 3]

        Returns:
            latent_code: Latent representation [B, latent_dim] or [latent_dim]
        """
        if vertices.dim() == 2:
            # Single mesh case
            vertices = vertices.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, num_vertices, _ = vertices.shape

        # Flatten vertices for processing
        vertices_flat = vertices.view(batch_size, -1)

        if self.use_variational:
            # Variational encoding
            hidden = self.encoder[:-1](vertices_flat)
            mu = self.mu_layer(hidden)
            logvar = self.logvar_layer(hidden)

            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latent_code = mu + eps * std

            # Return mu, logvar for VAE loss
            if squeeze_output:
                return latent_code.squeeze(0), mu.squeeze(0), logvar.squeeze(0)
            else:
                return latent_code, mu, logvar
        else:
            # Standard encoding
            latent_code = self.encoder(vertices_flat)

            if squeeze_output:
                return latent_code.squeeze(0)
            else:
                return latent_code


class MeshDecoder(nn.Module):
    """
    Decoder that maps latent codes to mesh vertices.
    This is the core component used in DNMP primitives.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        output_dim: int = 3,
        num_vertices: int = None,
        hidden_dims: tuple[int, ...] = (256, 128, 64, 32),
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_vertices = num_vertices
        self.use_residual = use_residual

        # Build decoder network
        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(
                        prev_dim,
                        hidden_dim,
                    )
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        if num_vertices is not None:
            output_size = num_vertices * output_dim
        else:
            # Will be determined dynamically
            output_size = 1024 * output_dim  # Default size

        layers.append(nn.Linear(prev_dim, output_size))

        self.decoder = nn.Sequential(*layers)

        # Base mesh for residual connection
        if self.use_residual and num_vertices is not None:
            self.register_buffer(
                "base_vertices",
                self._generate_base_mesh,
            )

    def _generate_base_mesh(self, num_vertices: int, output_dim: int) -> torch.Tensor:
        """Generate base mesh vertices (e.g., regular grid)."""
        # Create a regular 3D grid as base mesh
        grid_size = int(np.ceil(num_vertices ** (1 / 3)))

        x = torch.linspace(-1, 1, grid_size)
        y = torch.linspace(-1, 1, grid_size)
        z = torch.linspace(-1, 1, grid_size)

        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        base_vertices = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)

        # Truncate or pad to exact number of vertices
        if base_vertices.shape[0] > num_vertices:
            base_vertices = base_vertices[:num_vertices]
        elif base_vertices.shape[0] < num_vertices:
            padding = torch.zeros(num_vertices - base_vertices.shape[0], 3)
            base_vertices = torch.cat([base_vertices, padding], dim=0)

        return base_vertices

    def forward(self, latent_code: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to mesh vertices.

        Args:
            latent_code: Latent representation [B, latent_dim] or [latent_dim]

        Returns:
            vertices: Decoded mesh vertices [B, N, 3] or [N, 3]
        """
        if latent_code.dim() == 1:
            latent_code = latent_code.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = latent_code.shape[0]

        # Decode to vertex positions
        decoded = self.decoder(latent_code)

        if self.num_vertices is not None:
            # Reshape to vertex format
            vertices = decoded.view(batch_size, self.num_vertices, self.output_dim)

            # Add residual connection with base mesh
            if self.use_residual:
                vertices = vertices + self.base_vertices.unsqueeze(0)
        else:
            # Dynamic reshaping (for variable mesh sizes)
            vertices = decoded.view(batch_size, -1, self.output_dim)

        if squeeze_output:
            return vertices.squeeze(0)
        else:
            return vertices


class MeshAutoEncoder(nn.Module):
    """
    Complete mesh autoencoder combining encoder and decoder.
    Used for pre-training on mesh datasets.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_vertices: int = None,
        hidden_dims: tuple[int, ...] = (256, 128, 64, 32),
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_vertices = num_vertices
        self.use_variational = use_variational

        # Encoder
        input_dim = num_vertices * 3 if num_vertices else 3072  # Default
        self.encoder = MeshEncoder(
            input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims
        )
        self.encoder.use_variational = use_variational

        # Decoder
        self.decoder = MeshDecoder(
            latent_dim=latent_dim,
            output_dim=3,
            num_vertices=num_vertices,
            hidden_dims=hidden_dims[::-1],  # Reverse for decoder
        )

    def encode(self, vertices: torch.Tensor) -> torch.Tensor:
        """Encode mesh vertices to latent code."""
        return self.encoder(vertices)

    def decode(self, latent_code: torch.Tensor) -> torch.Tensor:
        """Decode latent code to mesh vertices."""
        return self.decoder(latent_code)

    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass."""
        if self.use_variational:
            latent_code, mu, logvar = self.encoder(vertices)
            reconstructed = self.decoder(latent_code)
            return reconstructed, mu, logvar
        else:
            latent_code = self.encoder(vertices)
            reconstructed = self.decoder(latent_code)
            return reconstructed

    def compute_loss(
        self,
        vertices: torch.Tensor,
        reconstructed: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
    ):
        """
        Compute autoencoder loss.

        Args:
            vertices: Original mesh vertices
            reconstructed: Reconstructed mesh vertices
            mu: Mean of latent distribution (for VAE)
            logvar: Log variance of latent distribution (for VAE)

        Returns:
            Total loss
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, vertices)

        if self.use_variational and mu is not None and logvar is not None:
            # KL divergence loss for VAE
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / vertices.numel()  # Normalize by number of elements

            total_loss = recon_loss + 0.1 * kl_loss  # Beta = 0.1
            return total_loss
        else:
            return recon_loss

    def sample_latent(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample random latent codes."""
        return torch.randn(num_samples, self.latent_dim, device=device)

    def interpolate(
        self, vertices1: torch.Tensor, vertices2: torch.Tensor, alpha: float
    ):
        """
        Interpolate between two meshes in latent space.

        Args:
            vertices1: First mesh vertices
            vertices2: Second mesh vertices
            alpha: Interpolation factor (0 to 1)

        Returns:
            Interpolated mesh vertices
        """
        latent1 = self.encode(vertices1)
        latent2 = self.encode(vertices2)

        # Linear interpolation in latent space
        latent_interp = (1 - alpha) * latent1 + alpha * latent2

        return self.decode(latent_interp)


def create_mesh_autoencoder(config_dict: dict) -> MeshAutoEncoder:
    """
    Factory function to create mesh autoencoder from config.

    Args:
        config_dict: Configuration dictionary

    Returns:
        MeshAutoEncoder instance
    """
    return MeshAutoEncoder(latent_dim=config_dict.get("latent_dim", 128))


def pretrain_mesh_autoencoder(
    autoencoder: MeshAutoEncoder,
    mesh_dataset: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: torch.device = None,
):
    """
    Pre-train mesh autoencoder on mesh dataset.

    Args:
        autoencoder: MeshAutoEncoder to train
        mesh_dataset: DataLoader with mesh data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Training device

    Returns:
        Trained autoencoder
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = autoencoder.to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    autoencoder.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, vertices in enumerate(mesh_dataset):
            vertices = vertices.to(device)

            optimizer.zero_grad()

            if autoencoder.use_variational:
                reconstructed, mu, logvar = autoencoder(vertices)
                loss = autoencoder.compute_loss(vertices, reconstructed, mu, logvar)
            else:
                reconstructed = autoencoder(vertices)
                loss = autoencoder.compute_loss(vertices, reconstructed)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch} completed, Average Loss: {avg_loss:.6f}")

    return autoencoder
