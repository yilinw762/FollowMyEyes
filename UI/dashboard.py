from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont
import cv2
import numpy as np
from core.tracker import GazeTracker

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

        self.tracker = GazeTracker()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame, screen_section, left_gaze, right_gaze, blink_count = self.tracker.process_frame(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        painter = QPainter(pixmap)
        painter.setBrush(QColor(255, 0, 0))
        painter.setPen(Qt.NoPen)

        if left_gaze:
            painter.drawEllipse(left_gaze[0] - 10, left_gaze[1] - 10, 20, 20)
        if right_gaze:
            painter.drawEllipse(right_gaze[0] - 10, right_gaze[1] - 10, 20, 20)

        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 20))
        painter.drawText(10, 30, f"Looking at: {screen_section}")
        painter.drawText(10, 60, f"Blinks: {blink_count}")
        painter.end()

        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()
