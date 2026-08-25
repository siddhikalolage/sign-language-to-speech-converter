import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gesture_pipeline import DEFAULT_MIN_LANDMARK_QUALITY
from gesture_pipeline import EXPECTED_FEATURE_COUNT
from gesture_pipeline import create_face_hands_landmarker
from gesture_pipeline import detect_holistic_landmarks
from gesture_pipeline import extract_landmarks
from gesture_pipeline import has_landmark_signal
from gesture_pipeline import landmark_quality_score
from gesture_pipeline import sanitize_label


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES_FOLDER = PROJECT_ROOT / "images for phrases"
DEFAULT_OUTPUT_FOLDER = PROJECT_ROOT / "text_phrase_image_data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageExtractionStats:
    labels_processed: int = 0
    images_checked: int = 0
    images_saved: int = 0
    images_skipped: int = 0
    low_quality_rejected: int = 0


def iter_image_files(folder: Path) -> list[Path]:
    return sorted(
        [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: (str(path.parent).lower(), path.name.lower()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract landmark CSVs from labeled gesture images."
    )
    parser.add_argument(
        "--images-folder",
        default=str(DEFAULT_IMAGES_FOLDER),
        help="Folder containing per-label image subdirectories.",
    )
    parser.add_argument(
        "--data-folder",
        default=str(DEFAULT_OUTPUT_FOLDER),
        help="Folder where extracted landmark CSVs will be written.",
    )
    parser.add_argument(
        "--min-landmark-quality",
        type=float,
        default=0.35,
        help="Reject images with incomplete face or hand coverage below this score.",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        help="Limit how many label folders are processed. Useful for a quick smoke test.",
    )
    parser.add_argument(
        "--max-images-per-label",
        type=int,
        help="Limit how many images are processed per label. Useful for a quick smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite CSVs that already exist.",
    )
    parser.add_argument(
        "--mirror-input",
        action="store_true",
        help="Horizontally mirror images before extracting landmarks. Leave off for normal web/upload alignment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_folder = Path(args.images_folder)
    output_folder = Path(args.data_folder)

    if not images_folder.exists() or not images_folder.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)
    stats = ImageExtractionStats()
    landmarker = create_face_hands_landmarker(static_image_mode=True)

    try:
        processed_labels = 0
        for label_dir in sorted(images_folder.iterdir(), key=lambda path: path.name.lower()):
            if not label_dir.is_dir():
                continue
            if args.max_labels and processed_labels >= args.max_labels:
                break

            label = sanitize_label(label_dir.name)
            image_files = iter_image_files(label_dir)
            if args.max_images_per_label:
                image_files = image_files[: args.max_images_per_label]
            if not image_files:
                continue

            processed_labels += 1
            stats.labels_processed += 1
            label_output_dir = output_folder / label
            label_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[info] Processing label '{label}' from {len(image_files)} image(s)")

            for image_path in image_files:
                stats.images_checked += 1
                output_path = label_output_dir / f"{image_path.stem}.csv"
                if output_path.exists() and not args.overwrite:
                    stats.images_skipped += 1
                    continue

                frame = cv2.imread(str(image_path))
                if frame is None:
                    print(f"[warn] Could not read image: {image_path}")
                    stats.images_skipped += 1
                    continue

                if args.mirror_input:
                    frame = cv2.flip(frame, 1)
                results = detect_holistic_landmarks(landmarker, frame)
                landmarks = extract_landmarks(results)
                quality = landmark_quality_score(results) if has_landmark_signal(landmarks) else 0.0
                if not has_landmark_signal(landmarks) or quality < args.min_landmark_quality:
                    stats.low_quality_rejected += 1
                    continue
                if len(landmarks) != EXPECTED_FEATURE_COUNT:
                    print(
                        f"[warn] Skipping {image_path.name}: expected {EXPECTED_FEATURE_COUNT} features, "
                        f"found {len(landmarks)}"
                    )
                    stats.images_skipped += 1
                    continue

                np.savetxt(output_path, [np.asarray(landmarks, dtype=np.float32)], delimiter=",")
                stats.images_saved += 1
                print(
                    "   - "
                    f"{image_path.name}: saved, quality={quality:.2f}"
                )
    finally:
        landmarker.close()

    print("\n[done] Image extraction finished.")
    print(f"[done] Output folder: {output_folder}")
    print(f"[done] Labels processed: {stats.labels_processed}")
    print(f"[done] Images checked: {stats.images_checked}")
    print(f"[done] Images saved: {stats.images_saved}")
    print(f"[done] Images skipped: {stats.images_skipped}")
    print(f"[done] Low-quality rejected: {stats.low_quality_rejected}")


if __name__ == "__main__":
    main()
