import cv2
import numpy as np
import mediapipe as mp
from cores.utils import compute_ear, get_screen_quadrant

class GazeTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

        self.left_eye_ids = [474, 475, 476, 477]
        self.right_eye_ids = [469, 470, 471, 472]

        self.left_eye_top = 386
        self.left_eye_bottom = 374
        self.left_eye_left = 263
        self.left_eye_right = 362

        self.right_eye_top = 159
        self.right_eye_bottom = 145
        self.right_eye_left = 133
        self.right_eye_right = 33

        self.blink_threshold = 0.25
        self.blink_counter = 0
        self.both_eyes_closed = False

    def process_frame(self, frame):
        h, w, _ = frame.shape
        left_gaze = None
        right_gaze = None
        screen_section = "Unknown"

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                landmarks = face.landmark

                left_coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.left_eye_ids]
                right_coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.right_eye_ids]

                if left_coords:
                    left_gaze = (int(np.mean([p[0] for p in left_coords])), int(np.mean([p[1] for p in left_coords])))
                if right_coords:
                    right_gaze = (int(np.mean([p[0] for p in right_coords])), int(np.mean([p[1] for p in right_coords])))

                if left_gaze and right_gaze:
                    avg_x = (left_gaze[0] + right_gaze[0]) // 2
                    avg_y = (left_gaze[1] + right_gaze[1]) // 2
                    screen_section = get_screen_quadrant(avg_x, avg_y, w, h)

                # for blink detection
                left_ear = compute_ear(landmarks, self.left_eye_top, self.left_eye_bottom, self.left_eye_left, self.left_eye_right, w, h)
                right_ear = compute_ear(landmarks, self.right_eye_top, self.right_eye_bottom, self.right_eye_left, self.right_eye_right, w, h)

                both_closed = (left_ear < self.blink_threshold) and (right_ear < self.blink_threshold)

                if both_closed:
                    if not self.both_eyes_closed:
                        self.both_eyes_closed = True
                else:
                    if self.both_eyes_closed:
                        self.both_eyes_closed = False
                        self.blink_counter += 1

        return frame, screen_section, left_gaze, right_gaze, self.blink_counter
