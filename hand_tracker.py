from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mediapipe as mp

# Finger landmark indices
THUMB = [1, 2, 3, 4]
INDEX = [5, 6, 7, 8]
MIDDLE = [9, 10, 11, 12]
RING = [13, 14, 15, 16]
PINKY = [17, 18, 19, 20]

FINGER_GROUPS = [THUMB, INDEX, MIDDLE, RING, PINKY]

# Connections for drawing hand skeleton (grouped by finger)
FINGER_CONNECTIONS = [
    # Thumb
    [(0, 1), (1, 2), (2, 3), (3, 4)],
    # Index
    [(0, 5), (5, 6), (6, 7), (7, 8)],
    # Middle
    [(0, 9), (9, 10), (10, 11), (11, 12)],
    # Ring
    [(0, 13), (13, 14), (14, 15), (15, 16)],
    # Pinky
    [(0, 17), (17, 18), (18, 19), (19, 20)],
]

# Palm connections
PALM_CONNECTIONS = [(5, 9), (9, 13), (13, 17)]

FINGERTIP_IDS = [4, 8, 12, 16, 20]

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"


@dataclass
class HandData:
    landmarks_px: np.ndarray  # (21, 2) pixel coordinates
    landmarks_norm: np.ndarray  # (21, 3) normalized coordinates
    handedness: str  # "Left" or "Right"


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
    ):
        vision = mp.tasks.vision
        base_options = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray) -> list[HandData]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self._timestamp_ms += 33  # ~30fps
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        hands: list[HandData] = []
        if not result.hand_landmarks:
            return hands

        for hand_landmarks, handedness_list in zip(
            result.hand_landmarks, result.handedness
        ):
            landmarks_norm = np.array(
                [(lm.x, lm.y, lm.z) for lm in hand_landmarks],
                dtype=np.float32,
            )
            landmarks_px = np.array(
                [(lm.x * w, lm.y * h) for lm in hand_landmarks],
                dtype=np.float32,
            )
            label = handedness_list[0].category_name
            hands.append(HandData(landmarks_px, landmarks_norm, label))

        return hands

    def close(self):
        self._landmarker.close()
