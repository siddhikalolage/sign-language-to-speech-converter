import os
import re
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FOLDER = Path("text_gesture_data")
VIDEO_DATA_FOLDER = Path("video dataset")
DEFAULT_MODEL_PATH = PROJECT_ROOT / "holistic_landmarker.task"
DEFAULT_FRAME_SKIP = 5
DEFAULT_HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
)
SELECTED_FACE_INDICES = [1, 33, 61, 199, 263, 291]
FACE_LANDMARK_COUNT = len(SELECTED_FACE_INDICES)
HAND_LANDMARK_COUNT = 21
HAND_FEATURE_COUNT = HAND_LANDMARK_COUNT * 3
DEFAULT_TEMPORAL_WINDOW = 5
DEFAULT_MIN_LANDMARK_QUALITY = 0.45
DEFAULT_MIN_HAND_POINTS = 12
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4"}
EXPECTED_FEATURE_COUNT = FACE_LANDMARK_COUNT * 3 + HAND_FEATURE_COUNT + HAND_FEATURE_COUNT


def sanitize_label(label: str) -> str:
    label = re.sub(r"^\s*\d+\s*[.)_-]*\s*", "", label.strip())
    label = label.lower()
    label = re.sub(r"[^a-z0-9]+", "_", label)
    label = re.sub(r"_+", "_", label)
    return label.strip("_") or "unknown"


def format_label(label: str) -> str:
    return label.replace("_", " ")


def iter_video_files(label_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in label_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ],
        key=lambda path: (str(path.parent).lower(), path.name.lower()),
    )


def download_holistic_model(
    destination: Path,
    download_url: str = DEFAULT_HOLISTIC_MODEL_URL,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(download_url, destination)
    return destination


def resolve_holistic_model_path(
    model_path: str | Path | None = None,
    *,
    download: bool = False,
    download_url: str = DEFAULT_HOLISTIC_MODEL_URL,
) -> Path:
    candidates: list[Path] = []
    if model_path:
        candidates.append(Path(model_path))

    env_model = os.getenv("MEDIAPIPE_HOLISTIC_MODEL")
    if env_model:
        candidates.append(Path(env_model))

    candidates.append(DEFAULT_MODEL_PATH)

    seen: set[Path] = set()
    ordered_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            ordered_candidates.append(resolved)

    for candidate in ordered_candidates:
        if candidate.exists():
            return candidate

    if download:
        target = ordered_candidates[0] if ordered_candidates else DEFAULT_MODEL_PATH
        return download_holistic_model(target, download_url=download_url)

    searched = ", ".join(str(path) for path in ordered_candidates)
    raise FileNotFoundError(
        "Holistic landmarker model not found. "
        f"Searched: {searched}. "
        "Place 'holistic_landmarker.task' in the project root, set "
        "MEDIAPIPE_HOLISTIC_MODEL, pass --model-path, or rerun with "
        "--download-model."
    )


def create_holistic_landmarker(
    model_path: str | Path | None = None,
    *,
    download_model: bool = False,
    download_url: str = DEFAULT_HOLISTIC_MODEL_URL,
    running_mode: str = "image",
):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    resolved_model_path = resolve_holistic_model_path(
        model_path,
        download=download_model,
        download_url=download_url,
    )
    option_kwargs = dict(
        base_options=python.BaseOptions(model_asset_path=str(resolved_model_path)),
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    running_mode_enum = getattr(vision, "RunningMode", None)
    normalized_running_mode = str(running_mode or "image").strip().upper()
    if running_mode_enum is not None and hasattr(running_mode_enum, normalized_running_mode):
        option_kwargs["running_mode"] = getattr(
            running_mode_enum,
            normalized_running_mode,
        )

    options = vision.HolisticLandmarkerOptions(**option_kwargs)
    landmarker = vision.HolisticLandmarker.create_from_options(options)
    return landmarker, resolved_model_path


def detect_holistic_landmarks(
    landmarker,
    frame,
    *,
    frame_timestamp_ms: int | None = None,
):
    if hasattr(landmarker, "process"):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return landmarker.process(rgb_frame)

    mp_image = to_mp_image(frame)
    if frame_timestamp_ms is not None and hasattr(landmarker, "detect_for_video"):
        return landmarker.detect_for_video(mp_image, int(max(0, frame_timestamp_ms)))
    return landmarker.detect(mp_image)


def create_legacy_holistic_landmarker(*, static_image_mode: bool = True):
    import mediapipe as mp

    return mp.solutions.holistic.Holistic(
        static_image_mode=static_image_mode,
        model_complexity=2,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


class FaceHandsLandmarker:
    def __init__(self, *, static_image_mode: bool = True):
        import mediapipe as mp

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, rgb_frame):
        face_results = self._face_mesh.process(rgb_frame)
        hand_results = self._hands.process(rgb_frame)

        face_landmarks = None
        if getattr(face_results, "multi_face_landmarks", None):
            face_landmarks = face_results.multi_face_landmarks[0].landmark

        left_hand_landmarks = None
        right_hand_landmarks = None
        hand_landmarks = getattr(hand_results, "multi_hand_landmarks", None) or []
        handedness = getattr(hand_results, "multi_handedness", None) or []
        for landmarks, hand_info in zip(hand_landmarks, handedness):
            label = hand_info.classification[0].label.lower()
            if label == "left" and left_hand_landmarks is None:
                left_hand_landmarks = landmarks.landmark
            elif label == "right" and right_hand_landmarks is None:
                right_hand_landmarks = landmarks.landmark

        return SimpleNamespace(
            face_landmarks=face_landmarks,
            left_hand_landmarks=left_hand_landmarks,
            right_hand_landmarks=right_hand_landmarks,
        )

    def close(self) -> None:
        self._face_mesh.close()
        self._hands.close()


class HandsOnlyLandmarker:
    def __init__(self, *, static_image_mode: bool = True):
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, rgb_frame):
        hand_results = self._hands.process(rgb_frame)

        left_hand_landmarks = None
        right_hand_landmarks = None
        hand_landmarks = getattr(hand_results, "multi_hand_landmarks", None) or []
        handedness = getattr(hand_results, "multi_handedness", None) or []
        for landmarks, hand_info in zip(hand_landmarks, handedness):
            label = hand_info.classification[0].label.lower()
            if label == "left" and left_hand_landmarks is None:
                left_hand_landmarks = landmarks.landmark
            elif label == "right" and right_hand_landmarks is None:
                right_hand_landmarks = landmarks.landmark

        return SimpleNamespace(
            face_landmarks=None,
            left_hand_landmarks=left_hand_landmarks,
            right_hand_landmarks=right_hand_landmarks,
        )

    def close(self) -> None:
        self._hands.close()


def create_face_hands_landmarker(*, static_image_mode: bool = True):
    try:
        return FaceHandsLandmarker(static_image_mode=static_image_mode)
    except Exception:
        running_mode = "image" if static_image_mode else "video"
        landmarker, _ = create_holistic_landmarker(running_mode=running_mode)
        return landmarker


def create_hands_landmarker(*, static_image_mode: bool = True):
    return HandsOnlyLandmarker(static_image_mode=static_image_mode)


def normalize_landmark_points(landmarks) -> list:
    if landmarks is None:
        return []
    if hasattr(landmarks, "landmark"):
        return list(landmarks.landmark)
    return list(landmarks)


def extract_landmarks(results) -> list[float]:
    landmarks: list[float] = []

    face_landmarks = normalize_landmark_points(getattr(results, "face_landmarks", None))
    if face_landmarks:
        for index in SELECTED_FACE_INDICES:
            point = face_landmarks[index]
            landmarks.extend([point.x, point.y, point.z])
    else:
        landmarks.extend([0.0] * len(SELECTED_FACE_INDICES) * 3)

    for hand_landmarks in (
        normalize_landmark_points(getattr(results, "left_hand_landmarks", None)),
        normalize_landmark_points(getattr(results, "right_hand_landmarks", None)),
    ):
        if hand_landmarks:
            for point in hand_landmarks:
                landmarks.extend([point.x, point.y, point.z])
        else:
            landmarks.extend([0.0] * 63)

    return landmarks


def has_landmark_signal(landmarks: list[float]) -> bool:
    return any(abs(value) > 1e-9 for value in landmarks)


def landmark_quality_score(results) -> float:
    face_landmarks = normalize_landmark_points(getattr(results, "face_landmarks", None))
    face_hits = sum(1 for index in SELECTED_FACE_INDICES if index < len(face_landmarks))
    left_hand_hits = len(normalize_landmark_points(getattr(results, "left_hand_landmarks", None)))
    right_hand_hits = len(normalize_landmark_points(getattr(results, "right_hand_landmarks", None)))

    face_ratio = face_hits / FACE_LANDMARK_COUNT
    left_ratio = min(left_hand_hits, HAND_LANDMARK_COUNT) / HAND_LANDMARK_COUNT
    right_ratio = min(right_hand_hits, HAND_LANDMARK_COUNT) / HAND_LANDMARK_COUNT

    return float(0.2 * face_ratio + 0.4 * left_ratio + 0.4 * right_ratio)


def hand_landmark_counts(results) -> tuple[int, int]:
    left_hand_hits = len(normalize_landmark_points(getattr(results, "left_hand_landmarks", None)))
    right_hand_hits = len(normalize_landmark_points(getattr(results, "right_hand_landmarks", None)))
    return left_hand_hits, right_hand_hits


def has_clear_hand_signal(results, *, min_hand_points: int = DEFAULT_MIN_HAND_POINTS) -> bool:
    left_hand_hits, right_hand_hits = hand_landmark_counts(results)
    return max(left_hand_hits, right_hand_hits) >= min_hand_points


def hand_visibility_score(results) -> float:
    left_hand_hits, right_hand_hits = hand_landmark_counts(results)
    return float(max(left_hand_hits, right_hand_hits) / HAND_LANDMARK_COUNT)


def hand_bounding_boxes(results, *, padding: float = 0.04) -> list[dict[str, float | str]]:
    boxes: list[dict[str, float | str]] = []
    for label, landmarks in (
        ("left", normalize_landmark_points(getattr(results, "left_hand_landmarks", None))),
        ("right", normalize_landmark_points(getattr(results, "right_hand_landmarks", None))),
    ):
        if not landmarks:
            continue
        xs = [point.x for point in landmarks]
        ys = [point.y for point in landmarks]
        boxes.append(
            {
                "hand": label,
                "x": max(0.0, min(xs) - padding),
                "y": max(0.0, min(ys) - padding),
                "width": min(1.0, max(xs) + padding) - max(0.0, min(xs) - padding),
                "height": min(1.0, max(ys) + padding) - max(0.0, min(ys) - padding),
            }
        )
    return boxes


def smooth_landmark_sequence(landmark_frames) -> np.ndarray:
    stacked = np.asarray(landmark_frames, dtype=np.float32)
    if stacked.ndim != 2 or stacked.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected shape (*, {EXPECTED_FEATURE_COUNT}) for smoothing, got "
            f"{stacked.shape}"
        )

    # Median smoothing is robust to occasional jittery detections in a video window.
    return np.median(stacked, axis=0)


def draw_landmarks(frame, results) -> None:
    frame_height, frame_width = frame.shape[:2]

    def draw_points(points, color) -> None:
        for point in points:
            x = int(point.x * frame_width)
            y = int(point.y * frame_height)
            cv2.circle(frame, (x, y), 2, color, -1)

    face_landmarks = normalize_landmark_points(getattr(results, "face_landmarks", None))
    left_hand_landmarks = normalize_landmark_points(getattr(results, "left_hand_landmarks", None))
    right_hand_landmarks = normalize_landmark_points(getattr(results, "right_hand_landmarks", None))

    if face_landmarks:
        draw_points(
            [face_landmarks[index] for index in SELECTED_FACE_INDICES],
            (0, 255, 0),
        )

    if left_hand_landmarks:
        draw_points(left_hand_landmarks, (255, 128, 0))

    if right_hand_landmarks:
        draw_points(right_hand_landmarks, (0, 128, 255))


def to_mp_image(frame):
    import mediapipe as mp

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
