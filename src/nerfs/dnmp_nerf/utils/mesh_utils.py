"""
Mesh processing utilities for DNMP.

This module provides functions for mesh manipulation, topology operations, and mesh quality metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import open3d as o3d


def compute_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute vertex normals from mesh vertices and faces.
    
    Args:
        vertices: Mesh vertices [V, 3]
        faces: Mesh faces [F, 3]
        
    Returns:
        vertex_normals: Vertex normals [V, 3]
    """
    device = vertices.device
    num_vertices = vertices.shape[0]
    
    # Initialize vertex normals
    vertex_normals = torch.zeros_like(vertices)
    
    # Compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_normals = face_normals / (torch.norm(face_normals, dim=1, keepdim=True) + 1e-8)
    
    # Accumulate face normals to vertices
    for i in range(3):
        vertex_indices = faces[:, i]
        vertex_normals.index_add_(0, vertex_indices, face_normals)
    
    # Normalize vertex normals
    vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-8)
    
    return vertex_normals


def compute_face_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute face areas.
    
    Args:
        vertices: Mesh vertices [V, 3]
        faces: Mesh faces [F, 3]
        
    Returns:
        face_areas: Face areas [F]
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    cross_product = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_areas = 0.5 * torch.norm(cross_product, dim=1)
    
    return face_areas


def compute_edge_lengths(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute edge lengths of mesh faces.
    
    Args:
        vertices: Mesh vertices [V, 3]
        faces: Mesh faces [F, 3]
        
    Returns:
        edge_lengths: Edge lengths [F, 3] (three edges per face)
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    edge_01 = torch.norm(v1 - v0, dim=1)
    edge_12 = torch.norm(v2 - v1, dim=1)
    edge_20 = torch.norm(v0 - v2, dim=1)
    
    edge_lengths = torch.stack([edge_01, edge_12, edge_20], dim=1)
    
    return edge_lengths


def laplacian_smoothing(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    lambda_smooth: float = 0.1
):
    """
    Apply Laplacian smoothing to mesh vertices.
    
    Args:
        vertices: Mesh vertices [V, 3]
        faces: Mesh faces [F, 3]
        lambda_smooth: Smoothing factor
        
    Returns:
        smoothed_vertices: Smoothed vertices [V, 3]
    """
    device = vertices.device
    num_vertices = vertices.shape[0]
    
    # Build adjacency matrix
    adjacency = torch.zeros(num_vertices, num_vertices, device=device)
    
    for face in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[face[i], face[j]] = 1.0
    
    # Compute vertex degrees
    degrees = adjacency.sum(dim=1)
    
    # Normalize adjacency matrix
    adjacency = adjacency / (degrees.unsqueeze(1) + 1e-8)
    
    # Apply Laplacian smoothing
    laplacian = adjacency @ vertices
    smoothed_vertices = vertices + lambda_smooth * (laplacian - vertices)
    
    return smoothed_vertices


def subdivide_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor
):
    """
    Subdivide mesh using loop subdivision (simplified version).
    
    Args:
        vertices: Mesh vertices [V, 3]
        faces: Mesh faces [F, 3]
        
    Returns:
        new_vertices: Subdivided vertices
        new_faces: Subdivided faces
    """
    device = vertices.device
    
    # Create edge-to-vertex mapping
    edges = {}
    edge_count = 0
    
    # Collect all edges
    for face in faces:
        for i in range(3):
            v1, v2 = face[i].item(), face[(i + 1) % 3].item()
            edge = tuple(sorted([v1, v2]))
            
            if edge not in edges:
                edges[edge] = len(vertices) + edge_count
                edge_count += 1
    
    # Create new vertices (original + edge midpoints)
    new_vertices = [vertices]
    
    for (v1, v2), _ in edges.items():
        midpoint = 0.5 * (vertices[v1] + vertices[v2])
        new_vertices.append(midpoint.unsqueeze(0))
    
    new_vertices = torch.cat(new_vertices, dim=0)
    
    # Create new faces
    new_faces = []
    
    for face in faces:
        v0, v1, v2 = face
        
        # Get edge vertices
        e01 = edges[tuple(sorted([v0.item(), v1.item()]))]
        e12 = edges[tuple(sorted([v1.item(), v2.item()]))]
        e20 = edges[tuple(sorted([v2.item(), v0.item()]))]
        
        # Create 4 new faces
        new_faces.extend([
            [v0, e01, e20], [v1, e12, e01], [v2, e20, e12], [e01, e12, e20]
        ])
    
    new_faces = torch.tensor(new_faces, device=device)
    
    return new_vertices, new_faces


def mesh_decimation(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    target_faces: int
):
    """
    Decimate mesh to reduce number of faces.
    
    Args:
        vertices: Mesh vertices [V, 3]
        faces: Mesh faces [F, 3]
        target_faces: Target number of faces
        
    Returns:
        decimated_vertices: Decimated vertices
        decimated_faces: Decimated faces
    """
    # Convert to Open3D mesh
    vertices_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy()
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
    mesh.triangles = o3d.utility.Vector3iVector(faces_np)
    
    # Perform decimation
    decimated_mesh = mesh.simplify_quadric_decimation(target_faces)
    
    # Convert back to tensors
    decimated_vertices = torch.from_numpy(np.asarray(decimated_mesh.vertices)).float()
    decimated_faces = torch.from_numpy(np.asarray(decimated_mesh.triangles)).long()
    
    return decimated_vertices, decimated_faces


def compute_mesh_quality_metrics(vertices: torch.Tensor, faces: torch.Tensor) -> dict[str, float]:
    """
    Compute various mesh quality metrics.
    
    Args:
        vertices: Mesh vertices [V, 3]
        faces: Mesh faces [F, 3]
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Face areas
    face_areas = compute_face_areas(vertices, faces)
    metrics['min_face_area'] = face_areas.min().item()
    metrics['max_face_area'] = face_areas.max().item()
    metrics['mean_face_area'] = face_areas.mean().item()
    
    # Edge lengths
    edge_lengths = compute_edge_lengths(vertices, faces)
    metrics['min_edge_length'] = edge_lengths.min().item()
    metrics['max_edge_length'] = edge_lengths.max().item()
    metrics['mean_edge_length'] = edge_lengths.mean().item()
    
    # Aspect ratios
    edge_lengths_per_face = edge_lengths
    min_edges = edge_lengths_per_face.min(dim=1)[0]
    max_edges = edge_lengths_per_face.max(dim=1)[0]
    aspect_ratios = max_edges / (min_edges + 1e-8)
    
    metrics['min_aspect_ratio'] = aspect_ratios.min().item()
    metrics['max_aspect_ratio'] = aspect_ratios.max().item()
    metrics['mean_aspect_ratio'] = aspect_ratios.mean().item()
    
    # Mesh bounds
    min_coords = vertices.min(dim=0)[0]
    max_coords = vertices.max(dim=0)[0]
    
    metrics['bbox_min'] = min_coords.tolist()
    metrics['bbox_max'] = max_coords.tolist()
    metrics['bbox_size'] = (max_coords - min_coords).tolist()
    
    return metrics


def remove_duplicate_vertices(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    tolerance: float = 1e-6
):
    """
    Remove duplicate vertices from mesh.
    
    Args:
        vertices: Mesh vertices [V, 3]
        faces: Mesh faces [F, 3]
        tolerance: Distance tolerance for duplicate detection
        
    Returns:
        clean_vertices: Vertices with duplicates removed
        clean_faces: Updated faces
    """
    device = vertices.device
    vertices_np = vertices.cpu().numpy()
    
    # Find unique vertices
    unique_vertices, inverse_indices = np.unique(
        vertices_np, axis=0, return_inverse=True
    )
    
    # Update faces
    clean_faces = faces.clone()
    for i in range(faces.shape[0]):
        for j in range(3):
            clean_faces[i, j] = inverse_indices[faces[i, j]]
    
    clean_vertices = torch.from_numpy(unique_vertices).float().to(device)
    
    return clean_vertices, clean_faces


def compute_mesh_volume(vertices: torch.Tensor, faces: torch.Tensor) -> float:
    """
    Compute mesh volume using divergence theorem.
    
    Args:
        vertices: Mesh vertices [V, 3]
        faces: Mesh faces [F, 3]
        
    Returns:
        volume: Mesh volume
    """
    volume = 0.0
    
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        # Compute signed volume of tetrahedron formed by origin and triangle
        volume += torch.dot(v0, torch.cross(v1, v2)) / 6.0
    
    return abs(volume.item())


def mesh_to_open3d(vertices: torch.Tensor, faces: torch.Tensor) -> o3d.geometry.TriangleMesh:
    """
    Convert PyTorch mesh to Open3D mesh.
    
    Args:
        vertices: Mesh vertices [V, 3]
        faces: Mesh faces [F, 3]
        
    Returns:
        Open3D triangle mesh
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
    
    return mesh


def open3d_to_mesh(mesh: o3d.geometry.TriangleMesh) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert Open3D mesh to PyTorch tensors.
    
    Args:
        mesh: Open3D triangle mesh
        
    Returns:
        vertices: Mesh vertices [V, 3]
        faces: Mesh faces [F, 3]
    """
    vertices = torch.from_numpy(np.asarray(mesh.vertices)).float()
    faces = torch.from_numpy(np.asarray(mesh.triangles)).long()
    
    return vertices, faces


def save_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    filename: str,
    vertex_colors: Optional[torch.Tensor] = None
):
    """
    Save mesh to file.
    
    Args:
        vertices: Mesh vertices [V, 3]
        faces: Mesh faces [F, 3]
        filename: Output filename
        vertex_colors: Optional vertex colors [V, 3]
    """
    mesh = mesh_to_open3d(vertices, faces)
    
    if vertex_colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.cpu().numpy())
    
    o3d.io.write_triangle_mesh(filename, mesh)


def load_mesh(filename: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load mesh from file.
    
    Args:
        filename: Input filename
        
    Returns:
        vertices: Mesh vertices [V, 3]
        faces: Mesh faces [F, 3]
    """
    mesh = o3d.io.read_triangle_mesh(filename)
    return open3d_to_mesh(mesh) 