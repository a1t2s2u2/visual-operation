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

GRAB_BORDER_COLOR = (0, 255, 100)     # Neon green
HOVER_BORDER_COLOR = (255, 255, 0)    # Cyan

# Face rendering
FACE_ALPHA = 0.45          # Semi-transparent holographic faces
FACE_ALPHA_GRABBED = 0.55  # Brighter when grabbed

# Edge glow
GLOW_THICKNESS = 2         # Outer glow width
EDGE_THICKNESS = 1         # Core bright line
GLOW_BRIGHTNESS = 0.5      # Glow dimness factor

# Depth shading range
MIN_BRIGHTNESS = 0.35
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
            self._draw_holographic(frame, obj, screen_pts, depths)

        return frame

    def _draw_holographic(
        self,
        frame: np.ndarray,
        obj: Wireframe3DObject,
        screen_pts: np.ndarray,
        depths: np.ndarray,
    ) -> None:
        d_min = float(depths.min())
        d_max = float(depths.max())
        d_range = d_max - d_min if d_max > d_min else 1.0

        base_b, base_g, base_r = obj.color

        # --- 1. Draw semi-transparent faces ---
        face_alpha = FACE_ALPHA_GRABBED if obj.grabbed else FACE_ALPHA
        if len(obj.faces) > 0:
            self._draw_faces(frame, obj, screen_pts, depths, d_min, d_range, face_alpha)

        # --- 2. Draw edge glow + core lines ---
        edge_list: list[tuple[float, int, int]] = []
        for idx in range(len(obj.edges)):
            i, j = obj.edges[idx]
            avg_depth = (depths[i] + depths[j]) / 2.0
            edge_list.append((avg_depth, i, j))

        # Sort edges: far first
        edge_list.sort(key=lambda e: e[0], reverse=True)

        for avg_depth, i, j in edge_list:
            p1 = (int(screen_pts[i, 0]), int(screen_pts[i, 1]))
            p2 = (int(screen_pts[j, 0]), int(screen_pts[j, 1]))

            t = 1.0 - (avg_depth - d_min) / d_range
            brightness = MIN_BRIGHTNESS + t * (MAX_BRIGHTNESS - MIN_BRIGHTNESS)

            # Glow layer (wider, dimmer)
            glow_color = (
                int(base_b * brightness * GLOW_BRIGHTNESS),
                int(base_g * brightness * GLOW_BRIGHTNESS),
                int(base_r * brightness * GLOW_BRIGHTNESS),
            )
            cv2.line(frame, p1, p2, glow_color, GLOW_THICKNESS, cv2.LINE_AA)

            # Core bright line
            core_color = (
                int(min(255, base_b * brightness)),
                int(min(255, base_g * brightness)),
                int(min(255, base_r * brightness)),
            )
            cv2.line(frame, p1, p2, core_color, EDGE_THICKNESS, cv2.LINE_AA)

        # --- 3. Vertex highlights (small bright dots on near vertices) ---
        for idx in range(len(screen_pts)):
            t = 1.0 - (depths[idx] - d_min) / d_range
            if t < 0.4:
                continue  # Skip far vertices
            brightness = MIN_BRIGHTNESS + t * (MAX_BRIGHTNESS - MIN_BRIGHTNESS)
            dot_color = (
                int(min(255, base_b * brightness * 1.2)),
                int(min(255, base_g * brightness * 1.2)),
                int(min(255, base_r * brightness * 1.2)),
            )
            pt = (int(screen_pts[idx, 0]), int(screen_pts[idx, 1]))
            cv2.circle(frame, pt, 2, dot_color, -1, cv2.LINE_AA)

        # --- 4. Grab / hover indicator ---
        if obj.grabbed or obj.hover:
            center = screen_pts.mean(axis=0).astype(int)
            scale = self.camera.focal / max(obj.z_depth, 1.0)
            proj_radius = int(obj.bounding_radius * scale)
            color = GRAB_BORDER_COLOR if obj.grabbed else HOVER_BORDER_COLOR
            # Dashed-look: draw thin circle
            cv2.circle(frame, tuple(center), proj_radius, color, 1, cv2.LINE_AA)

    def _draw_faces(
        self,
        frame: np.ndarray,
        obj: Wireframe3DObject,
        screen_pts: np.ndarray,
        depths: np.ndarray,
        d_min: float,
        d_range: float,
        face_alpha: float,
    ) -> None:
        base_b, base_g, base_r = obj.color

        # Sort faces by average depth (far first)
        face_depths: list[tuple[float, int]] = []
        for fi in range(len(obj.faces)):
            tri = obj.faces[fi]
            avg_d = (depths[tri[0]] + depths[tri[1]] + depths[tri[2]]) / 3.0
            face_depths.append((avg_d, fi))
        face_depths.sort(key=lambda x: x[0], reverse=True)

        overlay = frame.copy()
        for avg_d, fi in face_depths:
            tri = obj.faces[fi]
            pts = screen_pts[tri].astype(np.int32)

            t = 1.0 - (avg_d - d_min) / d_range
            brightness = MIN_BRIGHTNESS + t * (MAX_BRIGHTNESS - MIN_BRIGHTNESS)
            face_color = (
                int(base_b * brightness * 0.7),
                int(base_g * brightness * 0.7),
                int(base_r * brightness * 0.7),
            )
            cv2.fillPoly(overlay, [pts], face_color)

        cv2.addWeighted(overlay, face_alpha, frame, 1.0 - face_alpha, 0, frame)

    def draw_hand(self, frame: np.ndarray, hand: HandData) -> np.ndarray:
        pts = hand.landmarks_px.astype(np.int32)

        # Draw palm connections
        for i, j in PALM_CONNECTIONS:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]), (50, 50, 50), 3, cv2.LINE_AA)
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]), (200, 200, 200), 1, cv2.LINE_AA)

        # Draw finger connections with per-finger colors
        for finger_idx, connections in enumerate(FINGER_CONNECTIONS):
            color = FINGER_COLORS[finger_idx]
            for i, j in connections:
                # Dim glow
                glow = tuple(max(0, c // 3) for c in color)
                cv2.line(frame, tuple(pts[i]), tuple(pts[j]), glow, 3, cv2.LINE_AA)
                cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, 1, cv2.LINE_AA)

        # Draw landmarks
        for idx in range(21):
            pos = tuple(pts[idx])
            radius = 5 if idx in FINGERTIP_IDS else 2
            color = (200, 200, 200)
            for finger_idx, group_connections in enumerate(FINGER_CONNECTIONS):
                group_ids = set()
                for i, j in group_connections:
                    group_ids.add(i)
                    group_ids.add(j)
                if idx in group_ids and idx != 0:
                    color = FINGER_COLORS[finger_idx]
                    break
            cv2.circle(frame, pos, radius, color, -1, cv2.LINE_AA)

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
            cv2.circle(frame, tuple(mid), 12, (0, 255, 100), 2, cv2.LINE_AA)
            cv2.circle(frame, tuple(mid), 4, (0, 255, 100), -1, cv2.LINE_AA)
        elif hand_state.pinch_distance < 80:
            alpha = max(0, 1.0 - hand_state.pinch_distance / 80)
            intensity = int(255 * alpha)
            color = (0, intensity, intensity)
            cv2.circle(frame, tuple(mid), 8, color, 1, cv2.LINE_AA)

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
