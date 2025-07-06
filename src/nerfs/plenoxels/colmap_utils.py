"""
from __future__ import annotations

COLMAP Binary File Reader

This module provides utilities for reading COLMAP binary files:
- cameras.bin: Camera parameters
- images.bin: Image information and poses
- points3d.bin: 3D point cloud data

Based on COLMAP's binary file format:
https://colmap.github.io/format.html#binary-file-format
"""

import numpy as np
import struct
from pathlib import Path
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass
import quaternion  # numpy-quaternion package

@dataclass
class Camera:
    """Camera parameters from COLMAP."""
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

@dataclass
class Image:
    """Image information from COLMAP."""
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3d_ids: np.ndarray
    
    def qvec2rotmat(self) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        q = quaternion.quaternion(self.qvec[0], self.qvec[1], self.qvec[2], self.qvec[3])
        return quaternion.as_rotation_matrix(q)

@dataclass
class Point3D:
    """3D point information from COLMAP."""
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray

def read_next_bytes(
    fid,
    num_bytes: int,
    format_char_sequence: str,
    endian_character: str = "<"
) -> tuple:
    """Read and unpack the next bytes from a binary file.
    
    Args:
        fid: Open file object
        num_bytes: Number of bytes to read
        format_char_sequence: Format characters
        endian_character: Endian character ("<" for little endian)
        
    Returns:
        Tuple of read and unpacked values
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path: str | Path) -> Dict[int, Camera]:
    """Read COLMAP cameras.bin file.
    
    Args:
        path: Path to cameras.bin file
        
    Returns:
        Dictionary mapping camera IDs to Camera objects
    """
    cameras = {}
    
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            
            # Get camera model name
            camera_models = {
                0: "SIMPLE_PINHOLE",
                1: "PINHOLE",
                2: "SIMPLE_RADIAL",
                3: "RADIAL",
                4: "OPENCV",
                5: "OPENCV_FISHEYE",
                6: "FULL_OPENCV",
                7: "FOV",
                8: "SIMPLE_RADIAL_FISHEYE",
                9: "RADIAL_FISHEYE",
                10: "THIN_PRISM_FISHEYE"
            }
            model_name = camera_models.get(model_id, "UNKNOWN")
            
            # Read parameters
            num_params = read_next_bytes(fid, 8, "Q")[0]
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            params = np.array(params)
            
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=params
            )
    
    return cameras

def read_images_binary(path: str | Path) -> Dict[int, Image]:
    """Read COLMAP images.bin file.
    
    Args:
        path: Path to images.bin file
        
    Returns:
        Dictionary mapping image IDs to Image objects
    """
    images = {}
    
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        
        for _ in range(num_images):
            image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = image_properties[0]
            qvec = np.array(image_properties[1:5])
            tvec = np.array(image_properties[5:8])
            camera_id = image_properties[8]
            
            # Read image name
            name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            # Read observations
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            
            # Read point coordinates and 3D ids
            xys = np.zeros((num_points2D, 2))
            point3d_ids = np.zeros(num_points2D, dtype=np.int64)
            
            for i in range(num_points2D):
                xy = read_next_bytes(fid, 16, "dd")
                point3d_id = read_next_bytes(fid, 8, "q")[0]
                xys[i] = xy
                point3d_ids[i] = point3d_id
            
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
                xys=xys,
                point3d_ids=point3d_ids
            )
    
    return images

def read_points3d_binary(path: str | Path) -> Dict[int, Point3D]:
    """Read COLMAP points3D.bin file.
    
    Args:
        path: Path to points3D.bin file
        
    Returns:
        Dictionary mapping point IDs to Point3D objects
    """
    points3D = {}
    
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
            point_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            
            # Read track length
            track_length = read_next_bytes(fid, 8, "Q")[0]
            
            # Read track data
            track_data = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            image_ids = np.array([track_data[i] for i in range(0, len(track_data), 2)])
            point2D_idxs = np.array([track_data[i+1] for i in range(0, len(track_data), 2)])
            
            points3D[point_id] = Point3D(
                id=point_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs
            )
    
    return points3D 