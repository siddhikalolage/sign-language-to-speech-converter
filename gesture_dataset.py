import os
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split

from gesture_pipeline import DATA_FOLDER, EXPECTED_FEATURE_COUNT


def infer_sample_group(csv_path: Path) -> str:
    stem = csv_path.stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def load_dataset(data_folder: Path | str = DATA_FOLDER) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_folder = Path(data_folder)
    if not data_folder.is_dir():
        raise FileNotFoundError(
            f"Gesture data folder not found: {data_folder}\n"
            "Create the extracted landmark dataset first, or specify --data-folder."
        )

    samples: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []

    for label in sorted(os.listdir(data_folder)):
        label_folder = data_folder / label
        if not label_folder.is_dir():
            continue

        for csv_path in sorted(label_folder.glob("*.csv")):
            try:
                landmarks = np.loadtxt(csv_path, delimiter=",", ndmin=1)
            except Exception as exc:
                print(f"[warn] Failed to load {csv_path}: {exc}")
                continue

            if landmarks.size != EXPECTED_FEATURE_COUNT:
                print(
                    f"[warn] Skipping {csv_path}: expected {EXPECTED_FEATURE_COUNT} "
                    f"features, found {landmarks.size}"
                )
                continue

            samples.append(np.asarray(landmarks, dtype=np.float32))
            labels.append(label)
            groups.append(f"{label}/{infer_sample_group(csv_path)}")

    if not samples:
        raise ValueError(
            "No training samples were found. "
            "The current extracted landmark dataset is empty, so create the image or video CSV data first."
        )

    label_counts = Counter(labels)
    if len(label_counts) < 2:
        raise ValueError(
            "At least two gesture labels are required to train a classifier. "
            f"Found only: {sorted(label_counts)}"
        )

    print(f"[info] Loaded {len(samples)} sample(s) across {len(label_counts)} label(s)")
    print(f"[info] Smallest class size: {min(label_counts.values())}")
    print(f"[info] Largest class size: {max(label_counts.values())}")
    return np.stack(samples), np.array(labels), np.array(groups)


def split_dataset_indices(
    X: np.ndarray,
    y_encoded: np.ndarray,
    groups: np.ndarray,
    y_labels: np.ndarray,
    *,
    split_strategy: str = "video",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if split_strategy == "video":
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        train_indices, test_indices = next(splitter.split(X, y_encoded, groups))
        print("[info] Using source-video-aware evaluation to reduce train/test leakage.")
        return train_indices, test_indices

    label_counts = Counter(y_labels)
    can_stratify = min(label_counts.values()) >= 2
    stratify_labels = y_encoded if can_stratify else None
    if not can_stratify:
        print(
            "[warn] Some labels have fewer than 2 samples, so the frame split "
            "is not stratified."
        )

    adjusted_test_size = test_size
    class_count = len(np.unique(y_encoded))
    sample_count = len(X)
    if isinstance(adjusted_test_size, float) and adjusted_test_size < 1:
        minimum_fraction = class_count / sample_count
        if can_stratify and adjusted_test_size < minimum_fraction:
            print(
                f"[warn] Increasing test_size from {adjusted_test_size:.2f} to "
                f"{minimum_fraction:.2f} so every class appears in the test split."
            )
            adjusted_test_size = minimum_fraction
    elif can_stratify and adjusted_test_size < class_count:
        print(
            f"[warn] Increasing test_size from {adjusted_test_size} to {class_count} "
            "so every class appears in the test split."
        )
        adjusted_test_size = class_count

    return train_test_split(
        np.arange(len(X)),
        test_size=adjusted_test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
