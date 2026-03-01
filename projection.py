from __future__ import annotations

import math

import numpy as np


class Camera:
    """Pinhole camera model for 3D -> 2D projection."""

    def __init__(self, width: int, height: int, fov_deg: float = 60.0):
        self.width = width
        self.height = height
        self.fov_deg = fov_deg
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.focal = (height / 2.0) / math.tan(math.radians(fov_deg / 2.0))

    def project(self, points_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (N,3) world points to (N,2) screen coords and (N,) depths."""
        z = points_3d[:, 2]
        # Avoid division by zero
        safe_z = np.where(z > 1.0, z, 1.0)
        sx = self.focal * points_3d[:, 0] / safe_z + self.cx
        sy = self.focal * points_3d[:, 1] / safe_z + self.cy
        return np.column_stack([sx, sy]), z


def rotation_matrix_from_euler(rx: float, ry: float, rz: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from XYZ Euler angles (radians)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx


def unproject_screen_to_3d(
    sx: float, sy: float, depth: float, camera: Camera
) -> np.ndarray:
    """Convert 2D screen coordinate + depth back to 3D world point."""
    x = (sx - camera.cx) * depth / camera.focal
    y = (sy - camera.cy) * depth / camera.focal
    return np.array([x, y, depth], dtype=np.float64)
