import time

import cv2
import numpy as np

from hand_tracker import HandTracker
from gesture import GestureDetector
from objects import create_default_objects
from renderer import Renderer


class App:
    WINDOW_NAME = "Hand Tracking - Visual Operation"
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    def __init__(self):
        self.tracker = HandTracker()
        self.gesture = GestureDetector()
        self.renderer = Renderer()
        self.objects = create_default_objects()
        self.fps = 0.0
        self.fps_alpha = 0.3

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        prev_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Mirror display
                frame = cv2.flip(frame, 1)

                # Detect hands
                hands = self.tracker.process(frame)

                # Update gestures for all hands at once
                hand_states = self.gesture.update_all(hands, self.objects)

                # Draw objects (semi-transparent)
                self.renderer.draw_objects(frame, self.objects)

                # Draw hand skeletons and pinch indicators
                for hand, hs in zip(hands, hand_states):
                    self.renderer.draw_hand(frame, hand)
                    self.renderer.draw_pinch_indicator(frame, hand, hs)

                # FPS calculation (EMA smoothed)
                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    instant_fps = 1.0 / dt
                    self.fps = self.fps_alpha * instant_fps + (1 - self.fps_alpha) * self.fps

                self.renderer.draw_fps(frame, self.fps)
                self.renderer.draw_help(frame)

                cv2.imshow(self.WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord("r"):
                    self.gesture.release_all()
                    self.objects = create_default_objects()

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.tracker.close()


if __name__ == "__main__":
    App().run()
