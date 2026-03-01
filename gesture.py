from __future__ import annotations

from enum import Enum, auto

import numpy as np

from hand_tracker import HandData
from objects import DraggableRect, DraggableCircle


class GrabState(Enum):
    IDLE = auto()
    HOVERING = auto()
    GRABBING = auto()


PINCH_GRAB_THRESHOLD = 40.0
PINCH_RELEASE_THRESHOLD = 60.0
EMA_ALPHA = 0.4


class HandState:
    def __init__(self):
        self.state: GrabState = GrabState.IDLE
        self.grabbed_object: DraggableRect | DraggableCircle | None = None
        self.grab_offset: np.ndarray = np.zeros(2, dtype=np.float32)
        self.smoothed_pinch: np.ndarray | None = None
        self.pinch_distance: float = 0.0


class GestureDetector:
    def __init__(self):
        self._hand_states: dict[str, HandState] = {}

    def _get_state(self, hand_id: str) -> HandState:
        if hand_id not in self._hand_states:
            self._hand_states[hand_id] = HandState()
        return self._hand_states[hand_id]

    def update(
        self,
        hand: HandData,
        objects: list[DraggableRect | DraggableCircle],
    ) -> HandState:
        hs = self._get_state(hand.handedness)

        thumb_tip = hand.landmarks_px[4]
        index_tip = hand.landmarks_px[8]

        pinch_pos = (thumb_tip + index_tip) / 2.0
        hs.pinch_distance = float(np.linalg.norm(thumb_tip - index_tip))

        # EMA smoothing
        if hs.smoothed_pinch is None:
            hs.smoothed_pinch = pinch_pos.copy()
        else:
            hs.smoothed_pinch = EMA_ALPHA * pinch_pos + (1 - EMA_ALPHA) * hs.smoothed_pinch

        pinch = hs.smoothed_pinch

        if hs.state == GrabState.IDLE or hs.state == GrabState.HOVERING:
            # Check hover
            hovered = None
            for obj in reversed(objects):
                if obj.contains(pinch):
                    hovered = obj
                    break

            # Reset all hover flags for this hand
            for obj in objects:
                if not obj.grabbed:
                    obj.hover = False
            if hovered and not hovered.grabbed:
                hovered.hover = True

            if hovered:
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

        elif hs.state == GrabState.GRABBING:
            # Release check
            if hs.pinch_distance > PINCH_RELEASE_THRESHOLD:
                if hs.grabbed_object is not None:
                    hs.grabbed_object.grabbed = False
                    hs.grabbed_object = None
                hs.state = GrabState.IDLE
            else:
                # Move object
                if hs.grabbed_object is not None:
                    hs.grabbed_object.x = float(pinch[0] + hs.grab_offset[0])
                    hs.grabbed_object.y = float(pinch[1] + hs.grab_offset[1])

        return hs

    def release_all(self):
        for hs in self._hand_states.values():
            if hs.grabbed_object is not None:
                hs.grabbed_object.grabbed = False
                hs.grabbed_object = None
            hs.state = GrabState.IDLE
            hs.smoothed_pinch = None
        self._hand_states.clear()
