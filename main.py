import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont

class EyeTrackerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Tracker with Blink Detection")
        self.setGeometry(100, 100, 800, 600)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)

        # iris landmarks
        self.left_eye_ids = [474, 475, 476, 477]
        self.right_eye_ids = [469, 470, 471, 472]

        # eyelid for EAR calculation (approximate)
        self.left_eye_top = 386
        self.left_eye_bottom = 374
        self.left_eye_left = 263
        self.left_eye_right = 362

        self.right_eye_top = 159
        self.right_eye_bottom = 145
        self.right_eye_left = 133
        self.right_eye_right = 33

        # blink detection parameters
        self.blink_threshold = 0.25  # EAR threshold for closed eye
        self.blink_counter = 0
        self.both_eyes_closed = False

    def eye_aspect_ratio(self, landmarks, top, bottom, left, right, width, height):
        # vertical distance
        top_point = np.array([landmarks[top].x * width, landmarks[top].y * height])
        bottom_point = np.array([landmarks[bottom].x * width, landmarks[bottom].y * height])
        vertical_dist = np.linalg.norm(top_point - bottom_point)

        # horizontal distance
        left_point = np.array([landmarks[left].x * width, landmarks[left].y * height])
        right_point = np.array([landmarks[right].x * width, landmarks[right].y * height])
        horizontal_dist = np.linalg.norm(left_point - right_point)

        ear = vertical_dist / horizontal_dist
        return ear

    def get_screen_quadrant(self, x, y, width, height):
        col = 'Left' if x < width / 2 else 'Right'
        row = 'Top' if y < height / 2 else 'Bottom'
        return f"{row}-{col}"

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        left_gaze = None
        right_gaze = None
        screen_section = "Unknown"

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = face.landmark

                # eye dots
                left_coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.left_eye_ids]
                right_coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.right_eye_ids]

                if left_coords:
                    left_gaze = (int(np.mean([p[0] for p in left_coords])), int(np.mean([p[1] for p in left_coords])))
                if right_coords:
                    right_gaze = (int(np.mean([p[0] for p in right_coords])), int(np.mean([p[1] for p in right_coords])))

                # screen
                if left_gaze and right_gaze:
                    avg_x = (left_gaze[0] + right_gaze[0]) // 2
                    avg_y = (left_gaze[1] + right_gaze[1]) // 2
                    screen_section = self.get_screen_quadrant(avg_x, avg_y, w, h)

                # Blink detection (EAR)
                left_ear = self.eye_aspect_ratio(landmarks, self.left_eye_top, self.left_eye_bottom,
                                                 self.left_eye_left, self.left_eye_right, w, h)
                right_ear = self.eye_aspect_ratio(landmarks, self.right_eye_top, self.right_eye_bottom,
                                                  self.right_eye_left, self.right_eye_right, w, h)

                # if both eyes are closed
                both_closed = (left_ear < self.blink_threshold) and (right_ear < self.blink_threshold)

                if both_closed:
                    if not self.both_eyes_closed:
                        self.both_eyes_closed = True
                else:
                    if self.both_eyes_closed:
                        self.both_eyes_closed = False
                        self.blink_counter += 1

        #convert to Qt format
        image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        painter = QPainter(pixmap)
        painter.setBrush(QColor(255, 0, 0))
        painter.setPen(Qt.NoPen)
        if left_gaze:
            painter.drawEllipse(left_gaze[0] - 10, left_gaze[1] - 10, 20, 20)
        if right_gaze:
            painter.drawEllipse(right_gaze[0] - 10, right_gaze[1] - 10, 20, 20)

        #text info
        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 20))
        painter.drawText(10, 30, f"Looking at: {screen_section}")
        painter.drawText(10, 60, f"Blinks: {self.blink_counter}")
        painter.end()

        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EyeTrackerApp()
    window.show()
    sys.exit(app.exec_())
