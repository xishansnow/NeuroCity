from typing import Optional, Union
"""
Appearance Embedding Module for Block-NeRF

This module implements appearance embeddings to handle environmental variations
such as lighting changes, weather conditions, and temporal differences in 
large-scale urban scenes.

Based on Block-NeRF: Scalable Large Scene Neural View Synthesis (CVPR 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class AppearanceEmbedding(nn.Module):
    """
    Appearance embedding module for handling environmental variations.
    
    This module learns a latent appearance code for each image or block, allowing the model to handle different lighting conditions, weather, and temporal changes in large-scale scenes.
    """
    
    def __init__(
        self,
        num_embeddings: int = 1000,
        embedding_dim: int = 32,
        learnable: bool = True,
        initialization_std: float = 0.1,
        regularization_weight: float = 0.0
    ):
        """
        Initialize appearance embedding module.
        
        Args:
            num_embeddings: Maximum number of appearance codes
            embedding_dim: Dimension of each appearance embedding
            learnable: Whether embeddings are learnable parameters
            initialization_std: Standard deviation for initialization
            regularization_weight: L2 regularization weight for embeddings
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.learnable = learnable
        self.regularization_weight = regularization_weight
        
        # Create embedding table
        self.embeddings = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        
        # Initialize embeddings
        if learnable:
            nn.init.normal_(self.embeddings.weight, mean=0.0, std=initialization_std)
        else:
            # Fixed random embeddings
            with torch.no_grad():
                self.embeddings.weight.normal_(mean=0.0, std=initialization_std)
            self.embeddings.weight.requires_grad = False
        
        # Optional learnable global appearance transform
        self.use_global_transform = True
        if self.use_global_transform:
            self.global_transform = nn.Sequential(
                nn.Linear(
                    embedding_dim,
                    embedding_dim,
                )
            )
    
    def forward(self, appearance_ids: torch.Tensor) -> torch.Tensor:
        """
        Get appearance embeddings for given IDs.
        
        Args:
            appearance_ids: Tensor of appearance IDs (..., )
            
        Returns:
            Appearance embeddings (..., embedding_dim)
        """
        # Clamp IDs to valid range
        appearance_ids = torch.clamp(appearance_ids, 0, self.num_embeddings - 1)
        
        # Get embeddings
        embeddings = self.embeddings(appearance_ids)
        
        # Apply global transform if enabled
        if self.use_global_transform:
            embeddings = self.global_transform(embeddings)
        
        return embeddings
    
    def get_embedding_regularization(self) -> torch.Tensor:
        """Get L2 regularization loss for embeddings."""
        if self.regularization_weight <= 0:
            return torch.tensor(0.0, device=self.embeddings.weight.device)
        
        # L2 norm of all embeddings
        embedding_norm = torch.norm(self.embeddings.weight, dim=1)
        return self.regularization_weight * torch.mean(embedding_norm ** 2)
    
    def add_new_embedding(self, initialization: Optional[torch.Tensor] = None) -> int:
        """
        Add a new embedding to the table.
        
        Args:
            initialization: Optional initialization tensor (embedding_dim, )
            
        Returns:
            ID of the new embedding
        """
        if not self.learnable:
            raise ValueError("Cannot add embeddings to non-learnable embedding table")
        
        current_size = self.embeddings.num_embeddings
        
        # Create new embedding table with one more entry
        new_embeddings = nn.Embedding(
            num_embeddings=current_size + 1, embedding_dim=self.embedding_dim
        ).to(self.embeddings.weight.device)
        
        # Copy existing embeddings
        with torch.no_grad():
            new_embeddings.weight[:-1] = self.embeddings.weight
            
            # Initialize new embedding
            if initialization is not None:
                new_embeddings.weight[-1] = initialization
            else:
                nn.init.normal_(new_embeddings.weight[-1:], mean=0.0, std=0.1)
        
        # Replace embedding table
        self.embeddings = new_embeddings
        self.num_embeddings = current_size + 1
        
        return current_size  # Return ID of new embedding
    
    def interpolate_embeddings(self, id1: int, id2: int, weight: float) -> torch.Tensor:
        """
        Interpolate between two appearance embeddings.
        
        Args:
            id1: First embedding ID
            id2: Second embedding ID
            weight: Interpolation weight (0.0 = id1, 1.0 = id2)
            
        Returns:
            Interpolated embedding
        """
        emb1 = self.embeddings(torch.tensor(id1))
        emb2 = self.embeddings(torch.tensor(id2))
        
        interpolated = (1 - weight) * emb1 + weight * emb2
        
        if self.use_global_transform:
            interpolated = self.global_transform(interpolated)
        
        return interpolated
    
    def get_all_embeddings(self) -> torch.Tensor:
        """Get all embeddings in the table."""
        ids = torch.arange(self.num_embeddings, device=self.embeddings.weight.device)
        return self.forward(ids)
    
    def find_closest_embedding(self, target_embedding: torch.Tensor) -> int:
        """
        Find the closest embedding to a target embedding.
        
        Args:
            target_embedding: Target embedding (embedding_dim, )
            
        Returns:
            ID of closest embedding
        """
        all_embeddings = self.get_all_embeddings()  # (num_embeddings, embedding_dim)
        
        # Compute L2 distances
        distances = torch.norm(all_embeddings - target_embedding, dim=1)
        
        return torch.argmin(distances).item()

class ExposureEmbedding(nn.Module):
    """
    Exposure embedding module for handling camera exposure variations.
    """
    
    def __init__(self, exposure_dim: int = 8, use_learnable_transform: bool = True):
        """
        Initialize exposure embedding.
        
        Args:
            exposure_dim: Dimension of exposure embedding
            use_learnable_transform: Whether to use learnable transform
        """
        super().__init__()
        
        self.exposure_dim = exposure_dim
        self.use_learnable_transform = use_learnable_transform
        
        if use_learnable_transform:
            self.exposure_transform = nn.Sequential(
                nn.Linear(
                    1,
                    exposure_dim // 2,
                )
            )
        else:
            # Use simple positional encoding for exposure
            self.register_buffer(
                'freq_bands',
                torch.pow,
            )
    
    def forward(self, exposure_values: torch.Tensor) -> torch.Tensor:
        """
        Convert exposure values to embeddings.
        
        Args:
            exposure_values: Exposure values (..., 1)
            
        Returns:
            Exposure embeddings (..., exposure_dim)
        """
        if self.use_learnable_transform:
            return self.exposure_transform(exposure_values)
        else:
            # Positional encoding
            scaled = exposure_values * self.freq_bands
            return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)

class EnvironmentalEmbedding(nn.Module):
    """
    Environmental embedding module for additional environmental factors.
    """
    
    def __init__(self, weather_classes: int = 10, # sunny, cloudy, rainy, etc.
                 time_of_day_dim: int = 8, season_classes: int = 4, # spring, summer, fall, winter
                 embedding_dim: int = 32):
        """
        Initialize environmental embedding.
        
        Args:
            weather_classes: Number of weather condition classes
            time_of_day_dim: Dimension for time of day encoding
            season_classes: Number of season classes
            embedding_dim: Output embedding dimension
        """
        super().__init__()
        
        self.weather_embedding = nn.Embedding(weather_classes, embedding_dim // 4)
        self.season_embedding = nn.Embedding(season_classes, embedding_dim // 4)
        
        # Time of day as continuous value (0-24 hours)
        self.time_transform = nn.Sequential(
            nn.Linear(1, time_of_day_dim), nn.ReLU(), nn.Linear(time_of_day_dim, embedding_dim // 2)
        )
        
        # Combine all environmental factors
        self.combiner = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim,
            )
        )
    
    def forward(
        self,
        weather_ids: torch.Tensor,
        time_of_day: torch.Tensor,
        season_ids: torch.Tensor    
    ):
        """
        Combine environmental factors into embedding.
        
        Args:
            weather_ids: Weather condition IDs (..., )
            time_of_day: Time of day in hours (..., 1)
            season_ids: Season IDs (..., )
            
        Returns:
            Environmental embedding (..., embedding_dim)
        """
        weather_emb = self.weather_embedding(weather_ids)
        season_emb = self.season_embedding(season_ids)
        time_emb = self.time_transform(time_of_day)
        
        # Concatenate all factors
        combined = torch.cat([weather_emb, season_emb, time_emb], dim=-1)
        
        return self.combiner(combined)

class AdaptiveAppearanceEmbedding(nn.Module):
    """
    Adaptive appearance embedding that can handle various types of environmental variations.
    """
    
    def __init__(
        self,
        appearance_config: dict,
        use_exposure: bool = True,
        use_environmental: bool = False
    ):
        """
        Initialize adaptive appearance embedding.
        
        Args:
            appearance_config: Configuration for appearance embedding
            use_exposure: Whether to include exposure embedding
            use_environmental: Whether to include environmental embedding
        """
        super().__init__()
        
        self.use_exposure = use_exposure
        self.use_environmental = use_environmental
        
        # Main appearance embedding
        self.appearance_embedding = AppearanceEmbedding(**appearance_config)
        
        total_dim = appearance_config['embedding_dim']
        
        # Optional exposure embedding
        if use_exposure:
            self.exposure_embedding = ExposureEmbedding()
            total_dim += self.exposure_embedding.exposure_dim
        
        # Optional environmental embedding
        if use_environmental:
            self.environmental_embedding = EnvironmentalEmbedding()
            total_dim += 32  # Default environmental embedding dim
        
        # Final projection to desired output dimension
        self.output_projection = nn.Linear(total_dim, appearance_config['embedding_dim'])
    
    def forward(
        self,
        appearance_ids: torch.Tensor,
        exposure_values: Optional[torch.Tensor] = None,
        weather_ids: Optional[torch.Tensor] = None,
        time_of_day: Optional[torch.Tensor] = None,
        season_ids: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through adaptive appearance embedding.
        """
        embeddings = [self.appearance_embedding(appearance_ids)]
        
        if self.use_exposure and exposure_values is not None:
            exp_emb = self.exposure_embedding(exposure_values)
            embeddings.append(exp_emb)
        
        if self.use_environmental and all(
            x is not None for x in [
                weather_ids,
                time_of_day,
                season_ids
            ]
        ):
            env_emb = self.environmental_embedding(weather_ids, time_of_day, season_ids)
            embeddings.append(env_emb)
        
        # Concatenate all embeddings
        combined = torch.cat(embeddings, dim=-1)
        
        return self.output_projection(combined) 