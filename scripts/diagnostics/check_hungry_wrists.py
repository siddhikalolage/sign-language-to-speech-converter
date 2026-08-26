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
files = sorted(glob.glob(os.path.join(ROOT, "images for phrases", "hungry", "*.png")))
vals = []

for p in files:
    img = cv2.imread(p)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark

        left = lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST].y
        right = lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].y

        vals.append((os.path.basename(p), left, right))
    else:
        vals.append((os.path.basename(p), None, None))

pose.close()

print("File                 LeftWristY   RightWristY")
print("-" * 50)

for name, left, right in vals:
    print(
        "{:<20} {:<12} {:<12}".format(
            name,
            "NA" if left is None else round(left, 3),
            "NA" if right is None else round(right, 3)
        )
    )

valid = [
    x for x in vals
    if x[1] is not None and x[2] is not None
]

print()
print("Pose detected:", len(valid), "/", len(vals))

if valid:
    both = sum(
        1 for _, left, right in valid
        if left >= 0.95 and right >= 0.95
    )

    either = sum(
        1 for _, left, right in valid
        if left >= 0.95 or right >= 0.95
    )

    mean_left = np.mean([left for _, left, _ in valid])
    mean_right = np.mean([right for _, _, right in valid])

    print("Both wrists >= 0.95:", both)
    print("Either wrist >= 0.95:", either)
    print("Mean left wrist Y:", round(float(mean_left), 3))
    print("Mean right wrist Y:", round(float(mean_right), 3))

