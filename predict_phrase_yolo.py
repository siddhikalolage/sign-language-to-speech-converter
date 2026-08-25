import argparse
import json
import os
import time
from pathlib import Path

import cv2

from speech_output import SpeechWorker

PROJECT_ROOT = Path(__file__).resolve().parent


def configure_ultralytics_workspace() -> Path:
    config_dir = (PROJECT_ROOT / ".ultralytics").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    return config_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict one phrase at a time from the webcam using a YOLO classification model."
    )
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "text_phrase_yolo_cls.pt"),
        help="Path to the trained YOLO classification checkpoint.",
    )
    parser.add_argument(
        "--class-map",
        default=str(PROJECT_ROOT / "yolo_phrase_dataset" / "class_name_map.json"),
        help="JSON mapping from sanitized class names back to user-facing phrases.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.40,
        help="Only accept predictions above this confidence threshold.",
    )
    parser.add_argument(
        "--predict-every-ms",
        type=int,
        default=120,
        help="Run the classifier at most once this often.",
    )
    parser.add_argument(
        "--stability-frames",
        type=int,
        default=2,
        help="Require this many confident matching predictions before showing a phrase.",
    )
    parser.add_argument(
        "--release-frames",
        type=int,
        default=2,
        help="Require this many weak frames before unlocking for the next phrase.",
    )
    parser.add_argument(
        "--debug-ui",
        action="store_true",
        help="Show confidence and candidate overlays on screen.",
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="Speak each accepted word and allow the full sentence to be replayed.",
    )
    return parser.parse_args()


def load_class_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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

    font = cv2.FONT_HERSHEY_SIMPLEX
    label = prediction.replace("_", " ") if prediction else "Waiting"
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
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    configure_ultralytics_workspace()
    from ultralytics import YOLO

    class_map = load_class_map(Path(args.class_map))
    model = YOLO(str(model_path))
    speech_worker = SpeechWorker(enabled=args.speak)
    print("[info] Press B to undo, S to speak the sentence, C to clear, Q to quit.")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        speech_worker.close()
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    last_predict_at = 0.0
    candidate_label = ""
    candidate_confidence = 0.0
    previous_candidate = ""
    stable_frames = 0
    release_frames = 0
    current_prediction = ""
    locked_prediction = ""
    sentence_words: list[str] = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Keep camera orientation aligned with the image dataset and web app.
            now = time.perf_counter()
            if (now - last_predict_at) * 1000.0 >= max(1, args.predict_every_ms):
                last_predict_at = now
                results = model.predict(frame, verbose=False)
                if results:
                    probs = results[0].probs
                    if probs is not None:
                        top_index = int(probs.top1)
                        top_conf = float(probs.top1conf.item())
                        raw_label = str(results[0].names[top_index])
                        candidate_label = class_map.get(raw_label, raw_label).replace("_", " ")
                        candidate_confidence = top_conf
                        if top_conf >= args.min_confidence:
                            if candidate_label == locked_prediction:
                                current_prediction = locked_prediction
                                stable_frames = max(stable_frames, 1)
                            else:
                                if candidate_label == previous_candidate:
                                    stable_frames += 1
                                else:
                                    previous_candidate = candidate_label
                                    stable_frames = 1
                                if stable_frames >= args.stability_frames:
                                    current_prediction = candidate_label
                                    locked_prediction = candidate_label
                                    sentence_words.append(candidate_label)
                                    print(
                                        f"[done] Predicted: {candidate_label} "
                                        f"({candidate_confidence * 100:.1f}%)"
                                    )
                                    print(f"[done] Sentence: {' '.join(sentence_words)}")
                                    speech_worker.say(candidate_label)
                            release_frames = 0
                        else:
                            previous_candidate = ""
                            stable_frames = 0
                            if locked_prediction:
                                release_frames += 1
                                if release_frames >= max(1, args.release_frames):
                                    locked_prediction = ""
                            else:
                                release_frames = 0

            if args.debug_ui:
                cv2.putText(
                    frame,
                    f"Candidate: {candidate_label or '-'}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Confidence: {candidate_confidence * 100:.0f}%",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if current_prediction:
                    cv2.putText(
                        frame,
                        f"Shown: {current_prediction}",
                        (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (200, 255, 200),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                draw_prediction_only(frame, current_prediction, " ".join(sentence_words))

            cv2.imshow("YOLO Phrase Prediction", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                current_prediction = ""
                locked_prediction = ""
                sentence_words.clear()
                previous_candidate = ""
                stable_frames = 0
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


if __name__ == "__main__":
    main()
