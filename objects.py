from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from projection import Camera, rotation_matrix_from_euler


@dataclass
class Wireframe3DObject:
    # Shape definition
    local_vertices: np.ndarray  # (N, 3)
    edges: np.ndarray  # (M, 2) index pairs
    faces: np.ndarray  # (F, 3) triangle vertex indices
    bounding_radius: float = 50.0

    # Position (screen-space x/y, world z depth)
    x: float = 320.0
    y: float = 240.0
    z_depth: float = 400.0

    # Rotation (Euler angles in radians)
    rot_x: float = 0.0
    rot_y: float = 0.0
    rot_z: float = 0.0

    # Linear velocity (px/s)
    vx: float = 0.0
    vy: float = 0.0

    # Angular velocity (rad/s)
    angular_vx: float = 0.0
    angular_vy: float = 0.0
    angular_vz: float = 0.0

    # Interaction state
    grabbed: bool = False
    hover: bool = False

    # Visual (BGR)
    color: tuple[int, int, int] = (200, 200, 200)

    def get_world_vertices(
        self, camera: Camera
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rotate local verts, translate to world position, project to screen.

        Returns (screen_pts (N,2), depths (N,))
        """
        R = rotation_matrix_from_euler(self.rot_x, self.rot_y, self.rot_z)
        rotated = (R @ self.local_vertices.T).T  # (N, 3)

        # Convert screen position to world-space offset
        world_x = (self.x - camera.cx) * self.z_depth / camera.focal
        world_y = (self.y - camera.cy) * self.z_depth / camera.focal

        world_pts = rotated + np.array([world_x, world_y, self.z_depth])
        screen_pts, depths = camera.project(world_pts)
        return screen_pts, depths

    def contains(self, point: np.ndarray, camera: Camera) -> bool:
        """Hit-test using projected bounding circle."""
        screen_pts, _ = self.get_world_vertices(camera)
        center = screen_pts.mean(axis=0)
        scale = camera.focal / max(self.z_depth, 1.0)
        proj_radius = self.bounding_radius * scale
        dx = float(point[0]) - center[0]
        dy = float(point[1]) - center[1]
        return dx * dx + dy * dy <= proj_radius * proj_radius


# --- Shape Factories ---
# Each returns (vertices, edges, faces, bounding_radius)

_ShapeData = tuple[np.ndarray, np.ndarray, np.ndarray, float]


def make_cube(size: float = 50.0) -> _ShapeData:
    s = size / 2.0
    vertices = np.array(
        [
            [-s, -s, -s],  # 0
            [s, -s, -s],   # 1
            [s, s, -s],    # 2
            [-s, s, -s],   # 3
            [-s, -s, s],   # 4
            [s, -s, s],    # 5
            [s, s, s],     # 6
            [-s, s, s],    # 7
        ],
        dtype=np.float64,
    )
    edges = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ],
        dtype=np.int32,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],  # back
            [4, 6, 5], [4, 7, 6],  # front
            [0, 4, 5], [0, 5, 1],  # bottom
            [3, 2, 6], [3, 6, 7],  # top
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ],
        dtype=np.int32,
    )
    return vertices, edges, faces, size * math.sqrt(3) / 2.0


def make_octahedron(size: float = 50.0) -> _ShapeData:
    s = size
    vertices = np.array(
        [
            [s, 0, 0],    # 0 +X
            [-s, 0, 0],   # 1 -X
            [0, s, 0],    # 2 +Y
            [0, -s, 0],   # 3 -Y
            [0, 0, s],    # 4 +Z
            [0, 0, -s],   # 5 -Z
        ],
        dtype=np.float64,
    )
    edges = np.array(
        [
            [0, 2], [0, 3], [0, 4], [0, 5],
            [1, 2], [1, 3], [1, 4], [1, 5],
            [2, 4], [2, 5], [3, 4], [3, 5],
        ],
        dtype=np.int32,
    )
    faces = np.array(
        [
            [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
            [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5],
        ],
        dtype=np.int32,
    )
    return vertices, edges, faces, size


def make_tetrahedron(size: float = 50.0) -> _ShapeData:
    s = size
    vertices = np.array(
        [
            [s, s, s],
            [s, -s, -s],
            [-s, s, -s],
            [-s, -s, s],
        ],
        dtype=np.float64,
    )
    edges = np.array(
        [
            [0, 1], [0, 2], [0, 3],
            [1, 2], [1, 3], [2, 3],
        ],
        dtype=np.int32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2],
        ],
        dtype=np.int32,
    )
    return vertices, edges, faces, size * math.sqrt(3)


def make_icosahedron(size: float = 40.0) -> _ShapeData:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    s = size
    vertices = np.array(
        [
            [-s, phi * s, 0],     # 0
            [s, phi * s, 0],      # 1
            [-s, -phi * s, 0],    # 2
            [s, -phi * s, 0],     # 3
            [0, -s, phi * s],     # 4
            [0, s, phi * s],      # 5
            [0, -s, -phi * s],    # 6
            [0, s, -phi * s],     # 7
            [phi * s, 0, -s],     # 8
            [phi * s, 0, s],      # 9
            [-phi * s, 0, -s],    # 10
            [-phi * s, 0, s],     # 11
        ],
        dtype=np.float64,
    )
    edges = np.array(
        [
            [0, 1], [0, 5], [0, 7], [0, 10], [0, 11],
            [1, 5], [1, 7], [1, 8], [1, 9],
            [2, 3], [2, 4], [2, 6], [2, 10], [2, 11],
            [3, 4], [3, 6], [3, 8], [3, 9],
            [4, 5], [4, 9], [4, 11],
            [5, 9], [5, 11],
            [6, 7], [6, 8], [6, 10],
            [7, 8], [7, 10],
            [8, 9],
            [10, 11],
        ],
        dtype=np.int32,
    )
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int32,
    )
    return vertices, edges, faces, size * phi


def make_diamond(size: float = 50.0) -> _ShapeData:
    """Elongated octahedron (diamond shape)."""
    s = size
    h = size * 1.5
    vertices = np.array(
        [
            [s, 0, 0],    # 0
            [-s, 0, 0],   # 1
            [0, 0, s],    # 2
            [0, 0, -s],   # 3
            [0, h, 0],    # 4 top
            [0, -h, 0],   # 5 bottom
        ],
        dtype=np.float64,
    )
    edges = np.array(
        [
            [0, 2], [2, 1], [1, 3], [3, 0],
            [0, 4], [2, 4], [1, 4], [3, 4],
            [0, 5], [2, 5], [1, 5], [3, 5],
        ],
        dtype=np.int32,
    )
    faces = np.array(
        [
            [4, 0, 2], [4, 2, 1], [4, 1, 3], [4, 3, 0],
            [5, 2, 0], [5, 1, 2], [5, 3, 1], [5, 0, 3],
        ],
        dtype=np.int32,
    )
    return vertices, edges, faces, h


def create_default_objects() -> list[Wireframe3DObject]:
    cube_v, cube_e, cube_f, cube_r = make_cube(50.0)
    oct_v, oct_e, oct_f, oct_r = make_octahedron(45.0)
    tet_v, tet_e, tet_f, tet_r = make_tetrahedron(40.0)
    ico_v, ico_e, ico_f, ico_r = make_icosahedron(25.0)
    dia_v, dia_e, dia_f, dia_r = make_diamond(35.0)

    return [
        Wireframe3DObject(
            local_vertices=cube_v, edges=cube_e, faces=cube_f,
            bounding_radius=cube_r,
            x=120, y=150, z_depth=400,
            rot_x=0.3, rot_y=0.5, rot_z=0.0,
            angular_vy=0.3,
            color=(255, 180, 0),     # neon cyan
        ),
        Wireframe3DObject(
            local_vertices=oct_v, edges=oct_e, faces=oct_f,
            bounding_radius=oct_r,
            x=500, y=120, z_depth=400,
            rot_x=0.0, rot_y=0.3, rot_z=0.4,
            angular_vy=0.25,
            color=(0, 255, 200),     # neon green
        ),
        Wireframe3DObject(
            local_vertices=ico_v, edges=ico_e, faces=ico_f,
            bounding_radius=ico_r,
            x=320, y=300, z_depth=400,
            rot_x=0.2, rot_y=0.0, rot_z=0.3,
            angular_vy=0.2,
            color=(255, 100, 255),   # neon magenta-blue
        ),
        Wireframe3DObject(
            local_vertices=tet_v, edges=tet_e, faces=tet_f,
            bounding_radius=tet_r,
            x=200, y=380, z_depth=400,
            rot_x=0.5, rot_y=0.2, rot_z=0.0,
            angular_vy=0.35,
            color=(0, 230, 255),     # neon yellow-orange
        ),
        Wireframe3DObject(
            local_vertices=dia_v, edges=dia_e, faces=dia_f,
            bounding_radius=dia_r,
            x=450, y=350, z_depth=400,
            rot_x=0.0, rot_y=0.4, rot_z=0.2,
            angular_vy=0.28,
            color=(255, 0, 220),     # neon pink-purple
        ),
    ]
