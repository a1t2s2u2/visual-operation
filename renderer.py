from __future__ import annotations

import cv2
import numpy as np

from hand_tracker import (
    HandData,
    FINGER_CONNECTIONS,
    PALM_CONNECTIONS,
    FINGERTIP_IDS,
)
from gesture import HandState, GrabState
from objects import Wireframe3DObject
from projection import Camera

# Colors per finger group (BGR): Thumb, Index, Middle, Ring, Pinky
FINGER_COLORS = [
    (0, 100, 255),   # Thumb - orange
    (0, 255, 100),   # Index - green
    (255, 200, 0),   # Middle - cyan-blue
    (255, 0, 150),   # Ring - purple
    (100, 100, 255), # Pinky - pink
]

GRAB_BORDER_COLOR = (0, 255, 0)     # Green
HOVER_BORDER_COLOR = (255, 255, 0)  # Cyan
BORDER_THICKNESS = 2

# Depth shading range
MIN_BRIGHTNESS = 0.4
MAX_BRIGHTNESS = 1.0


class Renderer:
    def __init__(self, camera: Camera):
        self.camera = camera

    def draw_objects(
        self,
        frame: np.ndarray,
        objects: list[Wireframe3DObject],
    ) -> np.ndarray:
        # Sort by z_depth descending (far objects first = Painter's Algorithm)
        sorted_objs = sorted(objects, key=lambda o: o.z_depth, reverse=True)

        for obj in sorted_objs:
            screen_pts, depths = obj.get_world_vertices(self.camera)
            self._draw_wireframe(frame, obj, screen_pts, depths)

        return frame

    def _draw_wireframe(
        self,
        frame: np.ndarray,
        obj: Wireframe3DObject,
        screen_pts: np.ndarray,
        depths: np.ndarray,
    ) -> None:
        # Collect edges with their average depth for sorting
        edge_list: list[tuple[float, int, int]] = []
        for idx in range(len(obj.edges)):
            i, j = obj.edges[idx]
            avg_depth = (depths[i] + depths[j]) / 2.0
            edge_list.append((avg_depth, i, j))

        # Sort edges: far edges first (draw behind)
        edge_list.sort(key=lambda e: e[0], reverse=True)

        # Compute depth range for brightness mapping
        all_depths = depths
        d_min = float(all_depths.min())
        d_max = float(all_depths.max())
        d_range = d_max - d_min if d_max > d_min else 1.0

        base_b, base_g, base_r = obj.color

        for avg_depth, i, j in edge_list:
            p1 = (int(screen_pts[i, 0]), int(screen_pts[i, 1]))
            p2 = (int(screen_pts[j, 0]), int(screen_pts[j, 1]))

            # Depth-based brightness: closer = brighter
            t = 1.0 - (avg_depth - d_min) / d_range  # 0=far, 1=close
            brightness = MIN_BRIGHTNESS + t * (MAX_BRIGHTNESS - MIN_BRIGHTNESS)

            edge_color = (
                int(base_b * brightness),
                int(base_g * brightness),
                int(base_r * brightness),
            )

            # Black outline + colored line
            cv2.line(frame, p1, p2, (0, 0, 0), 3)
            cv2.line(frame, p1, p2, edge_color, 1)

        # Draw vertex dots for closer vertices
        for idx in range(len(screen_pts)):
            t = 1.0 - (depths[idx] - d_min) / d_range
            brightness = MIN_BRIGHTNESS + t * (MAX_BRIGHTNESS - MIN_BRIGHTNESS)
            dot_color = (
                int(base_b * brightness),
                int(base_g * brightness),
                int(base_r * brightness),
            )
            pt = (int(screen_pts[idx, 0]), int(screen_pts[idx, 1]))
            cv2.circle(frame, pt, 2, dot_color, -1)

        # Grab / hover bounding circle
        if obj.grabbed or obj.hover:
            center = screen_pts.mean(axis=0).astype(int)
            scale = self.camera.focal / max(obj.z_depth, 1.0)
            proj_radius = int(obj.bounding_radius * scale)
            color = GRAB_BORDER_COLOR if obj.grabbed else HOVER_BORDER_COLOR
            cv2.circle(frame, tuple(center), proj_radius, color, BORDER_THICKNESS)

    def draw_hand(self, frame: np.ndarray, hand: HandData) -> np.ndarray:
        pts = hand.landmarks_px.astype(np.int32)

        # Draw palm connections
        for i, j in PALM_CONNECTIONS:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]), (0, 0, 0), 4)
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]), (200, 200, 200), 2)

        # Draw finger connections with per-finger colors
        for finger_idx, connections in enumerate(FINGER_CONNECTIONS):
            color = FINGER_COLORS[finger_idx]
            for i, j in connections:
                cv2.line(frame, tuple(pts[i]), tuple(pts[j]), (0, 0, 0), 4)
                cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, 2)

        # Draw landmarks
        for idx in range(21):
            pos = tuple(pts[idx])
            radius = 6 if idx in FINGERTIP_IDS else 3
            cv2.circle(frame, pos, radius + 2, (0, 0, 0), -1)

            color = (200, 200, 200)
            for finger_idx, group_connections in enumerate(FINGER_CONNECTIONS):
                group_ids = set()
                for i, j in group_connections:
                    group_ids.add(i)
                    group_ids.add(j)
                if idx in group_ids and idx != 0:
                    color = FINGER_COLORS[finger_idx]
                    break
            cv2.circle(frame, pos, radius, color, -1)

        return frame

    def draw_pinch_indicator(
        self,
        frame: np.ndarray,
        hand: HandData,
        hand_state: HandState,
    ) -> np.ndarray:
        thumb_tip = hand.landmarks_px[4].astype(np.int32)
        index_tip = hand.landmarks_px[8].astype(np.int32)
        mid = ((thumb_tip + index_tip) // 2).astype(np.int32)

        if hand_state.state == GrabState.GRABBING:
            cv2.circle(frame, tuple(mid), 12, (0, 255, 0), 2)
            cv2.circle(frame, tuple(mid), 4, (0, 255, 0), -1)
        elif hand_state.pinch_distance < 80:
            alpha = max(0, 1.0 - hand_state.pinch_distance / 80)
            intensity = int(255 * alpha)
            color = (0, intensity, intensity)
            cv2.circle(frame, tuple(mid), 8, color, 1)

        return frame

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        text = f"FPS: {fps:.0f}"
        cv2.putText(
            frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3,
        )
        cv2.putText(
            frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2,
        )
        return frame

    def draw_help(self, frame: np.ndarray) -> np.ndarray:
        lines = [
            "Pinch: Grab object",
            "Drag: Move object",
            "Open: Release (with inertia)",
            "R: Reset  Q/ESC: Quit",
        ]
        h = frame.shape[0]
        y = h - 20 - (len(lines) - 1) * 22
        for line in lines:
            cv2.putText(
                frame, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2,
            )
            cv2.putText(
                frame, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1,
            )
            y += 22
        return frame
