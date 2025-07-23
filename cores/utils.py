import numpy as np

def compute_ear(landmarks, top, bottom, left, right, width, height):
    top_point = np.array([landmarks[top].x * width, landmarks[top].y * height])
    bottom_point = np.array([landmarks[bottom].x * width, landmarks[bottom].y * height])
    vertical_dist = np.linalg.norm(top_point - bottom_point)

    left_point = np.array([landmarks[left].x * width, landmarks[left].y * height])
    right_point = np.array([landmarks[right].x * width, landmarks[right].y * height])
    horizontal_dist = np.linalg.norm(left_point - right_point)

    return vertical_dist / horizontal_dist

def get_screen_quadrant(x, y, width, height):
    col = 'Left' if x < width / 2 else 'Right'
    row = 'Top' if y < height / 2 else 'Bottom'
    return f"{row}-{col}"
