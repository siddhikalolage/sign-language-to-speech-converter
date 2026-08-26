import cv2
import mediapipe as mp
import glob
import os
import numpy as np

pose = mp.solutions.pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.1
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
files = sorted(glob.glob(os.path.join(ROOT, "images for phrases", "again", "*.png")))
vals = []

for p in files:
    img = cv2.imread(p)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        left = lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST].y
        right = lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        right = right.y
        vals.append((os.path.basename(p), left, right))
    else:
        vals.append((os.path.basename(p), None, None))

pose.close()

valid = [
    x for x in vals
    if x[1] is not None and x[2] is not None
]

print("Pose detected:", len(valid), "/", len(vals))
print(
    "Both wrists >= 0.95:",
    sum(1 for _, l, r in valid if l >= 0.95 and r >= 0.95)
)
print(
    "Either wrist >= 0.95:",
    sum(1 for _, l, r in valid if l >= 0.95 or r >= 0.95)
)
print(
    "Mean left wrist Y:",
    round(float(np.mean([l for _, l, _ in valid])), 3)
)
print(
    "Mean right wrist Y:",
    round(float(np.mean([r for _, _, r in valid])), 3)
)

