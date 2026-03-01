from __future__ import annotations

import time
from enum import Enum, auto

import numpy as np

from hand_tracker import HandData
from objects import Wireframe3DObject
from projection import Camera


class GrabState(Enum):
    IDLE = auto()
    HOVERING = auto()
    GRABBING = auto()


PINCH_GRAB_THRESHOLD = 40.0
PINCH_RELEASE_THRESHOLD = 60.0
EMA_ALPHA = 0.4
POSITION_HISTORY_SIZE = 5
ANGULAR_KICK_FACTOR = 0.015  # rad/s per px/s of release speed


class HandState:
    def __init__(self):
        self.state: GrabState = GrabState.IDLE
        self.grabbed_object: Wireframe3DObject | None = None
        self.grab_offset: np.ndarray = np.zeros(2, dtype=np.float32)
        self.smoothed_pinch: np.ndarray | None = None
        self.pinch_distance: float = 0.0
        self.position_history: list[tuple[float, np.ndarray]] = []


class GestureDetector:
    def __init__(self, camera: Camera):
        self._hand_states: dict[str, HandState] = {}
        self._camera = camera

    def _get_state(self, hand_id: str) -> HandState:
        if hand_id not in self._hand_states:
            self._hand_states[hand_id] = HandState()
        return self._hand_states[hand_id]

    def _make_hand_id(self, hand: HandData, index: int, seen: dict[str, int]) -> str:
        label = hand.handedness
        count = seen.get(label, 0)
        seen[label] = count + 1
        if count > 0:
            return f"{label}_{count}"
        return label

    def _compute_release_velocity(self, history: list[tuple[float, np.ndarray]]) -> np.ndarray:
        """Compute velocity from position history on release."""
        if len(history) < 2:
            return np.zeros(2, dtype=np.float32)

        # Use oldest and newest samples for more stable velocity
        t0, p0 = history[0]
        t1, p1 = history[-1]
        dt = t1 - t0
        if dt < 0.001:
            return np.zeros(2, dtype=np.float32)
        return (p1 - p0) / dt

    def update_all(
        self,
        hands: list[HandData],
        objects: list[Wireframe3DObject],
        current_time: float,
    ) -> list[HandState]:
        seen: dict[str, int] = {}
        hand_ids = [self._make_hand_id(h, i, seen) for i, h in enumerate(hands)]
        active_ids = set(hand_ids)

        # Release objects held by disappeared hands
        for hid in list(self._hand_states.keys()):
            if hid not in active_ids:
                hs = self._hand_states[hid]
                if hs.grabbed_object is not None:
                    hs.grabbed_object.grabbed = False
                    hs.grabbed_object = None
                del self._hand_states[hid]

        # Reset hover for non-grabbed objects
        for obj in objects:
            if not obj.grabbed:
                obj.hover = False

        # Collect already-grabbed objects
        grabbed_set: set[int] = set()
        for hs in self._hand_states.values():
            if hs.grabbed_object is not None:
                grabbed_set.add(id(hs.grabbed_object))

        results: list[HandState] = []
        for hand, hid in zip(hands, hand_ids):
            hs = self._get_state(hid)

            thumb_tip = hand.landmarks_px[4]
            index_tip = hand.landmarks_px[8]
            pinch_pos = (thumb_tip + index_tip) / 2.0
            hs.pinch_distance = float(np.linalg.norm(thumb_tip - index_tip))

            # EMA smoothing
            if hs.smoothed_pinch is None:
                hs.smoothed_pinch = pinch_pos.copy()
            else:
                hs.smoothed_pinch = (
                    EMA_ALPHA * pinch_pos + (1 - EMA_ALPHA) * hs.smoothed_pinch
                )

            pinch = hs.smoothed_pinch

            if hs.state in (GrabState.IDLE, GrabState.HOVERING):
                # Find topmost hittable object
                hovered = None
                for obj in reversed(objects):
                    if obj.contains(pinch, self._camera) and id(obj) not in grabbed_set:
                        hovered = obj
                        break

                if hovered:
                    hovered.hover = True
                    hs.state = GrabState.HOVERING
                else:
                    hs.state = GrabState.IDLE

                # Transition to grabbing
                if hs.pinch_distance < PINCH_GRAB_THRESHOLD and hovered:
                    hs.state = GrabState.GRABBING
                    hs.grabbed_object = hovered
                    hs.grab_offset = np.array(
                        [hovered.x - pinch[0], hovered.y - pinch[1]],
                        dtype=np.float32,
                    )
                    hovered.grabbed = True
                    hovered.hover = False
                    grabbed_set.add(id(hovered))
                    # Reset position history
                    hs.position_history = [(current_time, pinch.copy())]

            elif hs.state == GrabState.GRABBING:
                if hs.pinch_distance > PINCH_RELEASE_THRESHOLD:
                    # --- Release with momentum ---
                    if hs.grabbed_object is not None:
                        vel = self._compute_release_velocity(hs.position_history)
                        hs.grabbed_object.vx = float(vel[0])
                        hs.grabbed_object.vy = float(vel[1])

                        # Angular velocity kick proportional to release speed
                        speed = float(np.linalg.norm(vel))
                        hs.grabbed_object.angular_vx += float(vel[1]) * ANGULAR_KICK_FACTOR
                        hs.grabbed_object.angular_vy += float(vel[0]) * ANGULAR_KICK_FACTOR
                        hs.grabbed_object.angular_vz += speed * ANGULAR_KICK_FACTOR * 0.3

                        grabbed_set.discard(id(hs.grabbed_object))
                        hs.grabbed_object.grabbed = False
                        hs.grabbed_object = None
                    hs.state = GrabState.IDLE
                    hs.position_history = []
                else:
                    # Move grabbed object
                    if hs.grabbed_object is not None:
                        hs.grabbed_object.x = float(pinch[0] + hs.grab_offset[0])
                        hs.grabbed_object.y = float(pinch[1] + hs.grab_offset[1])

                    # Record position history
                    hs.position_history.append((current_time, pinch.copy()))
                    if len(hs.position_history) > POSITION_HISTORY_SIZE:
                        hs.position_history.pop(0)

            results.append(hs)

        return results

    def release_all(self):
        for hs in self._hand_states.values():
            if hs.grabbed_object is not None:
                hs.grabbed_object.grabbed = False
                hs.grabbed_object = None
            hs.state = GrabState.IDLE
            hs.smoothed_pinch = None
            hs.position_history = []
        self._hand_states.clear()
