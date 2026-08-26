from pathlib import Path

import cv2

from gesture_pipeline import (
    create_face_hands_landmarker,
    detect_holistic_landmarks,
    extract_landmarks,
    hand_landmark_counts,
    landmark_quality_score,
    has_landmark_signal,
)


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "images for phrases" / "hungry"


def main():
    images = sorted(
        [
            p
            for p in IMAGE_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )

    print("=" * 75)
    print("HUNGRY LANDMARK DETECTION DIAGNOSTIC")
    print("=" * 75)

    landmarker = create_face_hands_landmarker(static_image_mode=True)

    try:
        for image_path in images:
            frame = cv2.imread(str(image_path))

            if frame is None:
                print(f"{image_path.name:30} IMAGE_READ_FAILED")
                continue

            results = detect_holistic_landmarks(landmarker, frame)

            left, right = hand_landmark_counts(results)
            quality = landmark_quality_score(results)

            landmarks = extract_landmarks(results)

            face_present = any(abs(x) > 1e-9 for x in landmarks[:18])

            status = (
                "PASS"
                if has_landmark_signal(landmarks) and quality >= 0.35
                else "REJECT"
            )

            print(
                f"{image_path.name:30} "
                f"left={left:2d} "
                f"right={right:2d} "
                f"face={'YES' if face_present else 'NO ':3} "
                f"quality={quality:.3f} "
                f"{status}"
            )

    finally:
        landmarker.close()


if __name__ == "__main__":

    main()
