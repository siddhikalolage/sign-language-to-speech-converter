import argparse
import json
import random
import shutil
from pathlib import Path

from gesture_pipeline import sanitize_label

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_image_files(folder: Path) -> list[Path]:
    return sorted(
        [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a YOLO classification dataset from labeled image folders."
    )
    parser.add_argument(
        "--source-folder",
        default=str(PROJECT_ROOT / "images for phrases"),
        help="Folder containing one subfolder per class with source images inside.",
    )
    parser.add_argument(
        "--output-folder",
        default=str(PROJECT_ROOT / "yolo_phrase_dataset"),
        help="Folder where the train/val/test split will be written.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of each class reserved for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of each class reserved for test.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used when shuffling images.",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        help="Limit how many classes are processed. Useful for a smoke test.",
    )
    parser.add_argument(
        "--max-images-per-label",
        type=int,
        help="Limit how many images are processed per class. Useful for a smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing prepared dataset folder before rebuilding it.",
    )
    return parser.parse_args()


def split_counts(total_count: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    val_count = int(round(total_count * val_ratio))
    test_count = int(round(total_count * test_ratio))
    train_count = total_count - val_count - test_count

    if train_count <= 0:
        train_count = max(1, total_count - val_count - test_count)
    while train_count + val_count + test_count > total_count:
        if test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            break
    while train_count + val_count + test_count < total_count:
        train_count += 1
    return train_count, val_count, test_count


def main() -> None:
    args = parse_args()
    source_folder = Path(args.source_folder)
    output_folder = Path(args.output_folder)

    if not source_folder.exists() or not source_folder.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_folder}")

    if output_folder.exists() and args.overwrite:
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    random.seed(args.random_state)
    class_name_map: dict[str, str] = {}
    split_totals = {"train": 0, "val": 0, "test": 0}
    processed_labels = 0

    for label_dir in sorted(source_folder.iterdir(), key=lambda path: path.name.lower()):
        if not label_dir.is_dir():
            continue
        if args.max_labels and processed_labels >= args.max_labels:
            break

        image_files = iter_image_files(label_dir)
        if args.max_images_per_label:
            image_files = image_files[: args.max_images_per_label]
        if not image_files:
            continue

        processed_labels += 1
        class_name = sanitize_label(label_dir.name)
        class_name_map[class_name] = label_dir.name

        shuffled = list(image_files)
        random.shuffle(shuffled)
        train_count, val_count, test_count = split_counts(
            len(shuffled),
            max(0.0, min(0.9, args.val_ratio)),
            max(0.0, min(0.9, args.test_ratio)),
        )
        split_slices = {
            "train": shuffled[:train_count],
            "val": shuffled[train_count : train_count + val_count],
            "test": shuffled[train_count + val_count : train_count + val_count + test_count],
        }

        print(
            f"[info] {label_dir.name} -> {class_name}: "
            f"train={len(split_slices['train'])}, "
            f"val={len(split_slices['val'])}, "
            f"test={len(split_slices['test'])}"
        )
        for split_name, files in split_slices.items():
            split_totals[split_name] += len(files)
            split_dir = output_folder / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for image_path in files:
                shutil.copy2(image_path, split_dir / image_path.name)

    mapping_path = output_folder / "class_name_map.json"
    mapping_path.write_text(json.dumps(class_name_map, indent=2), encoding="utf-8")

    print("\n[done] YOLO dataset prepared.")
    print(f"[done] Output folder: {output_folder}")
    print(f"[done] Class mapping: {mapping_path}")
    print(f"[done] Classes processed: {processed_labels}")
    print(f"[done] Train images: {split_totals['train']}")
    print(f"[done] Val images: {split_totals['val']}")
    print(f"[done] Test images: {split_totals['test']}")


if __name__ == "__main__":
    main()
