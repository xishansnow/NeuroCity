"""
Rasterization module for DNMP-NeRF.

This module implements mesh rasterization for deformable neural mesh primitives,
including vertex interpolation and depth testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class RasterizationConfig:
    """Configuration for mesh rasterization."""
    
    # Image resolution
    image_width: int = 512
    image_height: int = 512
    
    # Depth settings
    near_plane: float = 0.1
    far_plane: float = 100.0
    depth_resolution: float = 0.01
    
    # Anti-aliasing
    enable_antialiasing: bool = True
    samples_per_pixel: int = 4
    
    # Culling
    enable_backface_culling: bool = True
    enable_frustum_culling: bool = True
    
    # Performance
    use_sparse_gradients: bool = True
    max_faces_per_bin: int = 256


class VertexInterpolator(nn.Module):
    """Interpolates vertex features using barycentric coordinates."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, 
                vertex_features: torch.Tensor,
                faces: torch.Tensor,
                barycentric_coords: torch.Tensor,
                face_ids: torch.Tensor) -> torch.Tensor:
        """
        Interpolate vertex features using barycentric coordinates.
        
        Args:
            vertex_features: Vertex features [V, F]
            faces: Face indices [N_faces, 3]
            barycentric_coords: Barycentric coordinates [N_pixels, 3]
            face_ids: Face IDs for each pixel [N_pixels]
            
        Returns:
            interpolated_features: Interpolated features [N_pixels, F]
        """
        # Get vertex indices for each face
        face_vertices = faces[face_ids]  # [N_pixels, 3]
        
        # Get vertex features for each face
        v0_features = vertex_features[face_vertices[:, 0]]  # [N_pixels, F]
        v1_features = vertex_features[face_vertices[:, 1]]  # [N_pixels, F]
        v2_features = vertex_features[face_vertices[:, 2]]  # [N_pixels, F]
        
        # Interpolate using barycentric coordinates
        interpolated = (barycentric_coords[:, 0:1] * v0_features +
                       barycentric_coords[:, 1:2] * v1_features +
                       barycentric_coords[:, 2:3] * v2_features)
        
        return interpolated


class MeshRasterizer(nn.Module):
    """Core mesh rasterization functionality."""
    
    def __init__(self, config: RasterizationConfig):
        super().__init__()
        self.config = config
        self.vertex_interpolator = VertexInterpolator()
        
    def project_vertices(self, 
                        vertices: torch.Tensor,
                        camera_matrix: torch.Tensor,
                        view_matrix: torch.Tensor) -> torch.Tensor:
        """
        Project 3D vertices to 2D screen coordinates.
        
        Args:
            vertices: 3D vertices [V, 3]
            camera_matrix: Camera projection matrix [4, 4]
            view_matrix: View transformation matrix [4, 4]
            
        Returns:
            projected_vertices: 2D screen coordinates [V, 4] (x, y, z, w)
        """
        # Convert to homogeneous coordinates
        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[:, 0:1])], dim=1)
        
        # Apply view and projection transformations
        mvp_matrix = camera_matrix @ view_matrix
        projected = vertices_homo @ mvp_matrix.T
        
        # Perspective divide
        projected[:, :3] /= projected[:, 3:4]
        
        # Convert to screen coordinates
        screen_coords = projected.clone()
        screen_coords[:, 0] = (projected[:, 0] + 1.0) * 0.5 * self.config.image_width
        screen_coords[:, 1] = (projected[:, 1] + 1.0) * 0.5 * self.config.image_height
        
        return screen_coords
    
    def compute_barycentric_coords(self,
                                  pixel_coords: torch.Tensor,
                                  triangle_vertices: torch.Tensor) -> torch.Tensor:
        """
        Compute barycentric coordinates for pixels inside triangles.
        
        Args:
            pixel_coords: Pixel coordinates [N, 2]
            triangle_vertices: Triangle vertices in screen space [N, 3, 2]
            
        Returns:
            barycentric_coords: Barycentric coordinates [N, 3]
        """
        v0 = triangle_vertices[:, 0]  # [N, 2]
        v1 = triangle_vertices[:, 1]  # [N, 2]
        v2 = triangle_vertices[:, 2]  # [N, 2]
        
        # Compute vectors
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = pixel_coords - v0
        
        # Compute dot products
        dot00 = torch.sum(v0v2 * v0v2, dim=1)
        dot01 = torch.sum(v0v2 * v0v1, dim=1)
        dot02 = torch.sum(v0v2 * v0p, dim=1)
        dot11 = torch.sum(v0v1 * v0v1, dim=1)
        dot12 = torch.sum(v0v1 * v0p, dim=1)
        
        # Compute barycentric coordinates
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-8)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        w = 1.0 - u - v
        
        return torch.stack([w, u, v], dim=1)
    
    def depth_test(self,
                   fragments: Dict[str, torch.Tensor],
                   new_depths: torch.Tensor,
                   new_face_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform depth testing and update fragment buffer.
        
        Args:
            fragments: Current fragment buffer
            new_depths: New fragment depths [N]
            new_face_ids: New fragment face IDs [N]
            
        Returns:
            updated_fragments: Updated fragment buffer
        """
        if 'depth' not in fragments:
            # Initialize fragment buffer
            fragments['depth'] = new_depths
            fragments['face_ids'] = new_face_ids
            fragments['mask'] = torch.ones_like(new_depths, dtype=torch.bool)
        else:
            # Depth test
            closer_mask = new_depths < fragments['depth']
            
            # Update closer fragments
            fragments['depth'] = torch.where(closer_mask, new_depths, fragments['depth'])
            fragments['face_ids'] = torch.where(closer_mask, new_face_ids, fragments['face_ids'])
            fragments['mask'] = fragments['mask'] | closer_mask
        
        return fragments
    
    def rasterize_mesh(self,
                      vertices: torch.Tensor,
                      faces: torch.Tensor,
                      vertex_features: torch.Tensor,
                      camera_matrix: torch.Tensor,
                      view_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Rasterize mesh to produce fragments.
        
        Args:
            vertices: Mesh vertices [V, 3]
            faces: Mesh faces [F, 3]
            vertex_features: Vertex features [V, D]
            camera_matrix: Camera projection matrix [4, 4]
            view_matrix: View transformation matrix [4, 4]
            
        Returns:
            fragments: Rasterization fragments including depths, face IDs, etc.
        """
        device = vertices.device
        
        # Project vertices to screen space
        projected_vertices = self.project_vertices(vertices, camera_matrix, view_matrix)
        
        # Initialize fragment buffer
        fragments = {}
        
        # Create pixel grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.config.image_height, device=device, dtype=torch.float32),
            torch.arange(self.config.image_width, device=device, dtype=torch.float32),
            indexing='ij'
        )
        pixel_coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1)
        
        # Process each face
        for face_idx, face in enumerate(faces):
            # Get triangle vertices in screen space
            triangle_vertices = projected_vertices[face][:, :2]  # [3, 2]
            
            # Check if triangle is front-facing (if backface culling enabled)
            if self.config.enable_backface_culling:
                edge1 = triangle_vertices[1] - triangle_vertices[0]
                edge2 = triangle_vertices[2] - triangle_vertices[0]
                cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
                if cross < 0:  # Back-facing
                    continue
            
            # Compute bounding box
            min_coords = triangle_vertices.min(dim=0)[0]
            max_coords = triangle_vertices.max(dim=0)[0]
            
            # Clip to screen bounds
            min_coords = torch.clamp(min_coords, 0, 
                                   torch.tensor([self.config.image_width - 1, 
                                               self.config.image_height - 1], 
                                               device=device, dtype=torch.float32))
            max_coords = torch.clamp(max_coords, 0,
                                   torch.tensor([self.config.image_width - 1,
                                               self.config.image_height - 1],
                                               device=device, dtype=torch.float32))
            
            # Skip if triangle is outside screen bounds
            if (max_coords[0] < min_coords[0]) or (max_coords[1] < min_coords[1]):
                continue
            
            # Get pixels in bounding box
            x_min, x_max = int(min_coords[0]), int(max_coords[0]) + 1
            y_min, y_max = int(min_coords[1]), int(max_coords[1]) + 1
            
            if x_max <= x_min or y_max <= y_min:
                continue
            
            # Create pixel coordinates for this triangle
            yy, xx = torch.meshgrid(
                torch.arange(y_min, y_max, device=device, dtype=torch.float32),
                torch.arange(x_min, x_max, device=device, dtype=torch.float32),
                indexing='ij'
            )
            local_pixel_coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
            
            # Compute barycentric coordinates
            triangle_vertices_expanded = triangle_vertices.unsqueeze(0).expand(
                local_pixel_coords.shape[0], -1, -1)
            barycentric_coords = self.compute_barycentric_coords(
                local_pixel_coords, triangle_vertices_expanded)
            
            # Check which pixels are inside the triangle
            inside_mask = (barycentric_coords >= 0).all(dim=1) & (barycentric_coords <= 1).all(dim=1)
            
            if not inside_mask.any():
                continue
            
            # Filter to pixels inside triangle
            inside_pixels = local_pixel_coords[inside_mask]
            inside_barycentric = barycentric_coords[inside_mask]
            
            # Compute depths for inside pixels
            triangle_depths = projected_vertices[face][:, 2]  # [3]
            pixel_depths = torch.sum(inside_barycentric * triangle_depths.unsqueeze(0), dim=1)
            
            # Create face IDs
            face_ids = torch.full((inside_pixels.shape[0],), face_idx, device=device, dtype=torch.long)
            
            # Convert pixel coordinates to linear indices
            pixel_indices = (inside_pixels[:, 1].long() * self.config.image_width + 
                           inside_pixels[:, 0].long())
            
            # Perform depth test and update fragments
            if 'pixel_indices' not in fragments:
                fragments['pixel_indices'] = pixel_indices
                fragments['depths'] = pixel_depths
                fragments['face_ids'] = face_ids
                fragments['barycentric_coords'] = inside_barycentric
            else:
                # Combine with existing fragments
                all_indices = torch.cat([fragments['pixel_indices'], pixel_indices])
                all_depths = torch.cat([fragments['depths'], pixel_depths])
                all_face_ids = torch.cat([fragments['face_ids'], face_ids])
                all_barycentric = torch.cat([fragments['barycentric_coords'], inside_barycentric])
                
                # Sort by pixel index then by depth
                sorted_indices = torch.argsort(all_indices * 1000000 + all_depths)
                
                # Keep only closest fragment per pixel
                unique_indices, first_occurrence = torch.unique(all_indices[sorted_indices], 
                                                              return_inverse=True)
                
                fragments['pixel_indices'] = unique_indices
                fragments['depths'] = all_depths[sorted_indices][first_occurrence]
                fragments['face_ids'] = all_face_ids[sorted_indices][first_occurrence]
                fragments['barycentric_coords'] = all_barycentric[sorted_indices][first_occurrence]
        
        # Interpolate vertex features
        if 'face_ids' in fragments and vertex_features is not None:
            fragments['interpolated_features'] = self.vertex_interpolator(
                vertex_features, faces, fragments['barycentric_coords'], fragments['face_ids'])
        
        return fragments


class DNMPRasterizer(nn.Module):
    """Main rasterizer for DNMP primitives."""
    
    def __init__(self, config: RasterizationConfig):
        super().__init__()
        self.config = config
        self.mesh_rasterizer = MeshRasterizer(config)
    
    def forward(self,
                primitives: List,
                camera_matrix: torch.Tensor,
                view_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Rasterize all DNMP primitives.
        
        Args:
            primitives: List of DeformableNeuralMeshPrimitive objects
            camera_matrix: Camera projection matrix [4, 4]
            view_matrix: View transformation matrix [4, 4]
            
        Returns:
            combined_fragments: Combined rasterization results
        """
        device = camera_matrix.device
        all_fragments = []
        
        # Rasterize each primitive
        for primitive in primitives:
            vertices, faces, vertex_features = primitive()
            
            fragments = self.mesh_rasterizer.rasterize_mesh(
                vertices, faces, vertex_features, camera_matrix, view_matrix)
            
            if 'face_ids' in fragments:
                all_fragments.append(fragments)
        
        # Combine fragments from all primitives
        if not all_fragments:
            # Return empty fragments
            return {
                'pixel_indices': torch.empty(0, device=device, dtype=torch.long),
                'depths': torch.empty(0, device=device),
                'face_ids': torch.empty(0, device=device, dtype=torch.long),
                'primitive_ids': torch.empty(0, device=device, dtype=torch.long),
                'barycentric_coords': torch.empty(0, 3, device=device),
                'interpolated_features': torch.empty(0, 0, device=device)
            }
        
        # Combine all fragments
        combined_fragments = {}
        
        all_pixel_indices = []
        all_depths = []
        all_face_ids = []
        all_primitive_ids = []
        all_barycentric = []
        all_features = []
        
        for prim_idx, fragments in enumerate(all_fragments):
            all_pixel_indices.append(fragments['pixel_indices'])
            all_depths.append(fragments['depths'])
            all_face_ids.append(fragments['face_ids'])
            all_barycentric.append(fragments['barycentric_coords'])
            
            # Add primitive IDs
            primitive_ids = torch.full_like(fragments['face_ids'], prim_idx)
            all_primitive_ids.append(primitive_ids)
            
            if 'interpolated_features' in fragments:
                all_features.append(fragments['interpolated_features'])
        
        # Concatenate all fragments
        combined_fragments['pixel_indices'] = torch.cat(all_pixel_indices)
        combined_fragments['depths'] = torch.cat(all_depths)
        combined_fragments['face_ids'] = torch.cat(all_face_ids)
        combined_fragments['primitive_ids'] = torch.cat(all_primitive_ids)
        combined_fragments['barycentric_coords'] = torch.cat(all_barycentric)
        
        if all_features:
            combined_fragments['interpolated_features'] = torch.cat(all_features)
        
        # Perform global depth test
        if len(combined_fragments['pixel_indices']) > 0:
            # Sort by pixel index then by depth
            sort_keys = (combined_fragments['pixel_indices'].float() * 1000000 + 
                        combined_fragments['depths'])
            sorted_indices = torch.argsort(sort_keys)
            
            # Keep only closest fragment per pixel
            unique_pixels, first_occurrence = torch.unique(
                combined_fragments['pixel_indices'][sorted_indices],
                return_inverse=True
            )
            
            # Filter to unique pixels
            for key in combined_fragments:
                combined_fragments[key] = combined_fragments[key][sorted_indices][first_occurrence]
        
        return combined_fragments
    
    def render_fragments_to_image(self,
                                 fragments: Dict[str, torch.Tensor],
                                 image_features: torch.Tensor) -> torch.Tensor:
        """
        Render fragments to final image.
        
        Args:
            fragments: Rasterization fragments
            image_features: Per-fragment image features [N, C]
            
        Returns:
            rendered_image: Final rendered image [H, W, C]
        """
        device = image_features.device
        height, width = self.config.image_height, self.config.image_width
        num_channels = image_features.shape[-1]
        
        # Initialize output image
        rendered_image = torch.zeros(height, width, num_channels, device=device)
        
        if len(fragments['pixel_indices']) == 0:
            return rendered_image
        
        # Convert linear indices to 2D coordinates
        pixel_y = fragments['pixel_indices'] // width
        pixel_x = fragments['pixel_indices'] % width
        
        # Scatter features to image
        rendered_image[pixel_y, pixel_x] = image_features
        
        return rendered_image 