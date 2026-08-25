import argparse
import math
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from gesture_pipeline import (
    DEFAULT_MIN_HAND_POINTS,
    EXPECTED_FEATURE_COUNT,
    create_hands_landmarker,
    detect_holistic_landmarks,
    draw_landmarks,
    extract_landmarks,
    format_label,
    hand_landmark_counts,
    hand_visibility_score,
    has_clear_hand_signal,
    has_landmark_signal,
)
from speech_output import SpeechWorker


PROJECT_ROOT = Path(__file__).resolve().parent
FACE_FEATURE_COUNT = 18
HAND_FEATURE_COUNT = 63


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict one isolated gesture at a time from the webcam."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--classifier-path",
        default=str(PROJECT_ROOT / "text_phrase_image_model.pkl"),
        help="Path to the trained sklearn gesture model.",
    )
    parser.add_argument(
        "--label-encoder-path",
        default=str(PROJECT_ROOT / "text_phrase_image_label_encoder.pkl"),
        help="Path to the fitted label encoder.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="Only allow capture when the averaged class confidence reaches this threshold.",
    )
    parser.add_argument(
        "--min-landmark-quality",
        type=float,
        default=0.55,
        help="Ignore frames where hand tracking quality is below this threshold.",
    )
    parser.add_argument(
        "--min-hand-points",
        type=int,
        default=DEFAULT_MIN_HAND_POINTS,
        help="Require at least this many tracked points on one hand before predicting.",
    )
    parser.add_argument(
        "--stability-frames",
        type=int,
        default=4,
        help="Require this many stable high-quality frames before a capture is considered ready.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Average prediction probabilities over this many recent frames.",
    )
    parser.add_argument(
        "--max-motion",
        type=float,
        default=0.035,
        help="Treat frames above this motion level as unstable.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.25,
        help="Require the best class to beat the runner-up by at least this margin.",
    )
    parser.add_argument(
        "--consensus-window",
        type=int,
        default=6,
        help="How many recent stable frame labels to consider before committing a word.",
    )
    parser.add_argument(
        "--consensus-ratio",
        type=float,
        default=0.7,
        help="Portion of recent stable labels that must agree before a word is shown.",
    )
    parser.add_argument(
        "--release-frames",
        type=int,
        default=3,
        help="Require this many weak frames before unlocking for the next word.",
    )
    parser.add_argument(
        "--show-landmarks",
        action="store_true",
        help="Draw debug landmarks on top of the camera feed.",
    )
    parser.add_argument(
        "--debug-ui",
        action="store_true",
        help="Show tracking and confidence details on screen.",
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="Speak each accepted word and allow the full sentence to be replayed.",
    )
    return parser.parse_args()


def canonicalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 3).copy()
    mask = np.any(np.abs(points) > 1e-6, axis=1)
    if not np.any(mask):
        return points.reshape(-1)

    face = points[: FACE_FEATURE_COUNT // 3]
    face_mask = mask[: FACE_FEATURE_COUNT // 3]
    center = None
    scale = 1.0

    if np.any(face_mask):
        visible_face = face[face_mask]
        center = visible_face.mean(axis=0, keepdims=True)
        if face_mask[1] and face_mask[4]:
            scale = float(np.linalg.norm(face[1, :2] - face[4, :2]))

    if center is None:
        visible_points = points[mask]
        center = visible_points.mean(axis=0, keepdims=True)

    if scale < 1e-6:
        visible_points = points[mask]
        point_extent = visible_points.max(axis=0) - visible_points.min(axis=0)
        scale = float(max(point_extent[0], point_extent[1], 1.0))

    points[mask] = (points[mask] - center) / max(scale, 1e-6)
    return points.reshape(-1)


def compute_motion(previous_landmarks: np.ndarray | None, current_landmarks: np.ndarray) -> float:
    if previous_landmarks is None:
        return 0.0
    previous_landmarks = np.asarray(previous_landmarks, dtype=np.float32)
    current_landmarks = np.asarray(current_landmarks, dtype=np.float32)
    return float(np.mean(np.abs(current_landmarks - previous_landmarks)))


def top_prediction_with_margin(probabilities: np.ndarray) -> tuple[int, float, float]:
    ordered = np.argsort(probabilities)[::-1]
    top_index = int(ordered[0])
    top_confidence = float(probabilities[top_index])
    runner_up = float(probabilities[int(ordered[1])]) if len(ordered) > 1 else 0.0
    return top_index, top_confidence, top_confidence - runner_up


def wrap_text_to_width(text: str, font, scale: float, thickness: int, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_prediction_only(frame, prediction: str, sentence: str = "") -> None:
    if not prediction and not sentence:
        return

    label = format_label(prediction) if prediction else "Waiting"
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_height, frame_width = frame.shape[:2]
    max_text_width = max(120, frame_width - 48)

    title_scale = 1.0
    title_thickness = 3
    sentence_scale = 0.72
    sentence_thickness = 2
    sentence_lines = wrap_text_to_width(sentence, font, sentence_scale, sentence_thickness, max_text_width)[-2:]

    (title_width, title_height), title_baseline = cv2.getTextSize(
        label,
        font,
        title_scale,
        title_thickness,
    )
    line_height = cv2.getTextSize("Ag", font, sentence_scale, sentence_thickness)[0][1] + 12
    box_height = 54 + title_height + title_baseline + len(sentence_lines) * line_height
    y1 = max(12, frame_height - box_height - 18)
    y2 = frame_height - 18

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (18, y1),
        (frame_width - 18, y2),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    x = max(24, (frame_width - title_width) // 2)
    y = y1 + 28 + title_height
    cv2.putText(
        frame,
        label,
        (x, y),
        font,
        title_scale,
        (255, 255, 255),
        title_thickness,
        cv2.LINE_AA,
    )

    sentence_y = y + title_baseline + 24
    for line in sentence_lines:
        cv2.putText(
            frame,
            line,
            (30, sentence_y),
            font,
            sentence_scale,
            (214, 236, 255),
            sentence_thickness,
            cv2.LINE_AA,
        )
        sentence_y += line_height


def main() -> None:
    args = parse_args()
    classifier_path = Path(args.classifier_path)
    label_encoder_path = Path(args.label_encoder_path)

    model = joblib.load(classifier_path)
    classifier = getattr(model, "named_steps", {}).get("classifier")
    if classifier is not None and hasattr(classifier, "n_jobs"):
        classifier.n_jobs = 1
    label_encoder = joblib.load(label_encoder_path)
    expected_model_features = getattr(model, "n_features_in_", EXPECTED_FEATURE_COUNT)
    if expected_model_features != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Model expects {expected_model_features} features, but this pipeline extracts "
            f"{EXPECTED_FEATURE_COUNT}. Use a classifier trained from the current landmarks."
        )

    landmarker = create_hands_landmarker(static_image_mode=False)
    print(f"[info] Using classifier: {classifier_path}")
    print("[info] Using face/hands landmark runtime aligned with the web app.")
    print("[info] Hold one gesture steady. A word appears only after stable consensus.")
    print("[info] Press B to undo, S to speak the sentence, C to clear, Q to quit.")

    speech_worker = SpeechWorker(enabled=args.speak)
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        speech_worker.close()
        landmarker.close()
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    probability_window: deque[np.ndarray] = deque(maxlen=max(1, args.smoothing_window))
    landmark_window: deque[np.ndarray] = deque(maxlen=max(1, args.smoothing_window))
    label_window: deque[str] = deque(maxlen=max(1, args.consensus_window))
    previous_signature: np.ndarray | None = None
    stable_frames = 0
    release_frames = 0
    candidate_label = ""
    candidate_confidence = 0.0
    candidate_margin = 0.0
    current_prediction = ""
    locked_prediction = ""
    sentence_words: list[str] = []
    status_text = "Show one gesture and hold it"

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Keep camera orientation aligned with the image dataset and web app.
            timestamp_ms = time.monotonic_ns() // 1_000_000
            results = detect_holistic_landmarks(
                landmarker,
                frame,
                frame_timestamp_ms=timestamp_ms,
            )
            if args.show_landmarks:
                draw_landmarks(frame, results)

            landmarks = extract_landmarks(results)
            has_signal = has_landmark_signal(landmarks)
            has_hand_signal = has_clear_hand_signal(results, min_hand_points=args.min_hand_points)
            left_hand_points, right_hand_points = hand_landmark_counts(results)
            tracking_quality = hand_visibility_score(results) if has_signal else 0.0
            landmark_array = np.asarray(landmarks, dtype=np.float32)
            landmark_signature = canonicalize_landmarks(landmark_array) if has_signal else None
            motion = (
                compute_motion(previous_signature, landmark_signature)
                if landmark_signature is not None
                else 0.0
            )
            previous_signature = landmark_signature

            ready = False
            if (
                has_signal
                and has_hand_signal
                and tracking_quality >= args.min_landmark_quality
                and motion <= args.max_motion
            ):
                landmark_window.append(landmark_array)
                smoothed_landmarks = np.median(np.asarray(landmark_window, dtype=np.float32), axis=0)
                probabilities = model.predict_proba([smoothed_landmarks.tolist()])[0]
                probability_window.append(probabilities)
                averaged = np.mean(probability_window, axis=0)
                prediction, candidate_confidence, candidate_margin = top_prediction_with_margin(averaged)
                candidate_label = str(label_encoder.inverse_transform([prediction])[0])
                if candidate_margin >= args.min_margin:
                    label_window.append(candidate_label)
                else:
                    label_window.append("")
                stable_frames += 1
                release_frames = 0
                if locked_prediction:
                    current_prediction = locked_prediction
                    status_text = "Hold until release"
                else:
                    required_votes = max(
                        1,
                        math.ceil(len(label_window) * min(max(args.consensus_ratio, 0.0), 1.0)),
                    )
                    consensus_hits = sum(1 for label in label_window if label == candidate_label)
                    ready = (
                        stable_frames >= args.stability_frames
                        and candidate_confidence >= args.min_confidence
                        and candidate_margin >= args.min_margin
                        and consensus_hits >= required_votes
                    )
                    status_text = "Stable" if ready else "Hold steady"
                    if ready and candidate_label:
                        current_prediction = candidate_label
                        locked_prediction = candidate_label
                        spoken_label = format_label(candidate_label)
                        sentence_words.append(spoken_label)
                        print(
                            f"[done] Predicted: {candidate_label} "
                            f"({candidate_confidence * 100:.1f}%)"
                        )
                        print(f"[done] Sentence: {' '.join(sentence_words)}")
                        if args.speak:
                            speech_worker.say(spoken_label)
            else:
                probability_window.clear()
                landmark_window.clear()
                label_window.clear()
                stable_frames = 0
                candidate_label = ""
                candidate_confidence = 0.0
                candidate_margin = 0.0
                if has_signal:
                    status_text = "Show your hand clearly" if not has_hand_signal else "Hold steadier"
                else:
                    status_text = "Show one gesture"
                if locked_prediction:
                    release_frames += 1
                    if release_frames >= max(1, args.release_frames):
                        locked_prediction = ""
                else:
                    release_frames = 0

            if args.debug_ui:
                cv2.putText(
                    frame,
                    f"Status: {status_text}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0) if ready else (0, 215, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Candidate: {format_label(candidate_label) if candidate_label else '-'}",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Confidence: {candidate_confidence * 100:.0f}%",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Margin: {candidate_margin * 100:.0f}%",
                    (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Tracking: {tracking_quality * 100:.0f}%",
                    (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Hand pts: L{left_hand_points} R{right_hand_points}",
                    (10, 215),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 220, 160),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Stable: {stable_frames}",
                    (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (200, 255, 200),
                    2,
                    cv2.LINE_AA,
                )
            else:
                draw_prediction_only(frame, current_prediction, " ".join(sentence_words))

            cv2.imshow("Single Gesture Prediction", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                current_prediction = ""
                locked_prediction = ""
                sentence_words.clear()
                release_frames = 0
            elif key == ord("b"):
                if sentence_words:
                    sentence_words.pop()
                    print(f"[done] Sentence: {' '.join(sentence_words) or '(empty)'}")
            elif key == ord("s"):
                speech_worker.say(" ".join(sentence_words))
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        speech_worker.close()
        landmarker.close()


if __name__ == "__main__":
    main()
