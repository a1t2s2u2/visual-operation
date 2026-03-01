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
from objects import DraggableRect, DraggableCircle

# Colors per finger group (BGR): Thumb, Index, Middle, Ring, Pinky
FINGER_COLORS = [
    (0, 100, 255),   # Thumb - orange
    (0, 255, 100),   # Index - green
    (255, 200, 0),   # Middle - cyan-blue
    (255, 0, 150),   # Ring - purple
    (100, 100, 255), # Pinky - pink
]

OBJECT_ALPHA = 0.7
GRAB_BORDER_COLOR = (0, 255, 0)     # Green
HOVER_BORDER_COLOR = (255, 255, 0)  # Cyan
BORDER_THICKNESS = 3


class Renderer:
    def draw_objects(
        self,
        frame: np.ndarray,
        objects: list[DraggableRect | DraggableCircle],
    ) -> np.ndarray:
        overlay = frame.copy()

        for obj in objects:
            if isinstance(obj, DraggableRect):
                x1 = int(obj.x - obj.width / 2)
                y1 = int(obj.y - obj.height / 2)
                x2 = int(obj.x + obj.width / 2)
                y2 = int(obj.y + obj.height / 2)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), obj.color, -1)

                if obj.grabbed:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), GRAB_BORDER_COLOR, BORDER_THICKNESS)
                elif obj.hover:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), HOVER_BORDER_COLOR, BORDER_THICKNESS)

            elif isinstance(obj, DraggableCircle):
                center = (int(obj.x), int(obj.y))
                cv2.circle(overlay, center, int(obj.radius), obj.color, -1)

                if obj.grabbed:
                    cv2.circle(frame, center, int(obj.radius), GRAB_BORDER_COLOR, BORDER_THICKNESS)
                elif obj.hover:
                    cv2.circle(frame, center, int(obj.radius), HOVER_BORDER_COLOR, BORDER_THICKNESS)

        cv2.addWeighted(overlay, OBJECT_ALPHA, frame, 1 - OBJECT_ALPHA, 0, frame)
        return frame

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

            # Find which finger this landmark belongs to
            color = (200, 200, 200)  # Default (wrist)
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
            # Fading indicator as fingers approach
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
            "Open: Release",
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
