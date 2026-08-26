from pathlib import Path
import cv2

from gesture_pipeline import (
    create_face_hands_landmarker,
    detect_holistic_landmarks,
    normalize_landmark_points,
    landmark_quality_score,
    has_landmark_signal,
    extract_landmarks,
)

ROOT = Path(__file__).resolve().parents[2]
folder = ROOT / "images for phrases" / "hungry"

if not folder.exists():
    raise FileNotFoundError(f"Folder not found: {folder}")

landmarker = create_face_hands_landmarker(static_image_mode=True)

try:
    print("=" * 90)
    print("HUNGRY LANDMARK DIAGNOSTIC")
    print("=" * 90)
    print(f"Folder: {folder}")
    print()

    for image_path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if not image_path.is_file():
            continue

        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"{image_path.name:20} IMAGE READ FAILED")
            continue

        results = detect_holistic_landmarks(landmarker, frame)

        face = normalize_landmark_points(
            getattr(results, "face_landmarks", None)
        )

        left = normalize_landmark_points(
            getattr(results, "left_hand_landmarks", None)
        )

        right = normalize_landmark_points(
            getattr(results, "right_hand_landmarks", None)
        )

        landmarks = extract_landmarks(results)

        quality = (
            landmark_quality_score(results)
            if has_landmark_signal(landmarks)
            else 0.0
        )

        print(
            f"{image_path.name:20} "
            f"Face={len(face):3} "
            f"Left={len(left):2} "
            f"Right={len(right):2} "
            f"Quality={quality:.3f}"
        )

finally:
    landmarker.close()

print()
print("=" * 90)
print("DIAGNOSTIC COMPLETE")
print("=" * 90)

