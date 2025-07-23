# FollowMyEyes
**Eye Tracker with Blink Detection**

A Python desktop application that uses a webcam to detect where you're looking and how often you blink — all in real-time. Built with PyQt5 and OpenCV, this tool overlays gaze tracking and blink information directly onto the video feed.

**Features**
Real-time gaze tracking (left and right eye)

Blink detection with live count

Visual overlay of gaze points and activity

Intuitive PyQt5 interface

**Requirements**
Python 3.7+

OpenCV (opencv-python)

PyQt5

NumPy

Install dependencies using pip:
pip install -r requirements.txt
Or manually:
pip install opencv-python PyQt5 numpy

**How to Run**
python main.py
Make sure your webcam is enabled and accessible.

**How It Works**
core/tracker.py: Uses facial landmarks to track eye movement and blinks.

ui/dashboard.py: Displays live webcam feed, overlays gaze points and blink count.

Gaze points are marked with red circles; blink count updates in real-time.

**To-Do / Future Ideas**
Save gaze/blink data to CSV

Add sound or popup alerts for fatigue

Calibrate screen sections for better accuracy

Heatmap of gaze over time


