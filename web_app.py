import base64
import os
import time
from pathlib import Path
from threading import Lock

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gesture_pipeline import (
    DEFAULT_MIN_HAND_POINTS,
    create_hands_landmarker,
    detect_holistic_landmarks,
    extract_landmarks,
    format_label,
    hand_bounding_boxes,
    hand_landmark_counts,
    hand_visibility_score,
    has_clear_hand_signal,
    has_landmark_signal,
)

PROJECT_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PROJECT_ROOT / "web"
YOLO_CONFIG_DIR = PROJECT_ROOT / ".ultralytics"

LANDMARK_MODEL_PATH = PROJECT_ROOT / "text_phrase_image_model.pkl"
LANDMARK_ENCODER_PATH = PROJECT_ROOT / "text_phrase_image_label_encoder.pkl"
YOLO_MODEL_PATH = PROJECT_ROOT / "text_phrase_yolo_cls.pt"
YOLO_CLASS_MAP_PATH = PROJECT_ROOT / "yolo_phrase_dataset" / "class_name_map.json"
LANDMARK_MIN_CONFIDENCE = 0.55
LANDMARK_MIN_MARGIN = 0.25
LANDMARK_MIN_QUALITY = 0.55
LANDMARK_MIN_HAND_POINTS = DEFAULT_MIN_HAND_POINTS
LANDMARK_MAX_FRAME_WIDTH = 320
YOLO_MIN_CONFIDENCE = 0.40
AUTO_YOLO_MIN_CONFIDENCE = 0.90


class PredictRequest(BaseModel):
    image_data: str
    mode: str = "auto"


class LandmarkRuntime:
    def __init__(self) -> None:
        self.model = joblib.load(LANDMARK_MODEL_PATH)
        classifier = getattr(self.model, "named_steps", {}).get("classifier")
        if classifier is not None and hasattr(classifier, "n_jobs"):
            classifier.n_jobs = 1
        self.label_encoder = joblib.load(LANDMARK_ENCODER_PATH)
        self.landmarker = create_hands_landmarker(static_image_mode=True)
        self.lock = Lock()
        try:
            detect_holistic_landmarks(self.landmarker, np.zeros((224, 224, 3), dtype=np.uint8))
        except Exception:
            pass

    def predict(self, frame: np.ndarray, *, orientation: str = "normal") -> dict:
        frame = resize_for_landmarks(frame)
        with self.lock:
            results = detect_holistic_landmarks(self.landmarker, frame)

        landmarks = extract_landmarks(results)
        if not has_landmark_signal(landmarks):
            raise HTTPException(status_code=422, detail="No face or hand landmarks were detected.")
        if not has_clear_hand_signal(results, min_hand_points=LANDMARK_MIN_HAND_POINTS):
            raise HTTPException(status_code=422, detail="No clear hand landmarks were detected.")

        quality = hand_visibility_score(results)
        left_hand_points, right_hand_points = hand_landmark_counts(results)
        probabilities = self.model.predict_proba([landmarks])[0]
        ordered = np.argsort(probabilities)[::-1]
        top_index = int(ordered[0])
        confidence = float(probabilities[top_index])
        runner_up = float(probabilities[int(ordered[1])]) if len(ordered) > 1 else 0.0
        label = str(self.label_encoder.inverse_transform([top_index])[0])
        top = []
        for index in ordered[:5]:
            raw = str(self.label_encoder.inverse_transform([int(index)])[0])
            top.append(
                {
                    "label": format_label(raw),
                    "confidence": float(probabilities[int(index)]),
                }
            )
        return {
            "label": format_label(label),
            "confidence": confidence,
            "quality": quality,
            "margin": confidence - runner_up,
            "hand_points": max(left_hand_points, right_hand_points),
            "left_hand_points": left_hand_points,
            "right_hand_points": right_hand_points,
            "hand_boxes": hand_bounding_boxes(results),
            "backend": "landmark",
            "orientation": orientation,
            "top": top,
        }


class YoloRuntime:
    def __init__(self) -> None:
        os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR.resolve()))
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "YOLO runtime is unavailable. Install the web requirements to use YOLO mode."
            ) from exc

        self.model = YOLO(str(YOLO_MODEL_PATH))
        if YOLO_CLASS_MAP_PATH.exists():
            import json

            self.class_map = json.loads(YOLO_CLASS_MAP_PATH.read_text(encoding="utf-8"))
        else:
            self.class_map = {}
        self.lock = Lock()
        try:
            self.model.predict(np.zeros((224, 224, 3), dtype=np.uint8), verbose=False)
        except Exception:
            pass

    def predict(self, frame: np.ndarray, *, orientation: str = "normal") -> dict:
        with self.lock:
            results = self.model.predict(frame, verbose=False)

        if not results or results[0].probs is None:
            raise HTTPException(status_code=422, detail="YOLO could not classify the image.")

        probs = results[0].probs
        top_index = int(probs.top1)
        confidence = float(probs.top1conf.item())
        raw_label = str(results[0].names[top_index])
        label = self.class_map.get(raw_label, raw_label).replace("_", " ")
        top = []
        prob_values = probs.data.detach().cpu().numpy()
        ordered = np.argsort(prob_values)[::-1]
        for index in ordered[:5]:
            raw = str(results[0].names[int(index)])
            top.append(
                {
                    "label": self.class_map.get(raw, raw).replace("_", " "),
                    "confidence": float(prob_values[int(index)]),
                }
            )
        return {
            "label": label,
            "confidence": confidence,
            "quality": None,
            "margin": None,
            "hand_points": None,
            "left_hand_points": None,
            "right_hand_points": None,
            "hand_boxes": [],
            "backend": "yolo",
            "orientation": orientation,
            "top": top,
        }


class RuntimeRegistry:
    def __init__(self) -> None:
        self._landmark: LandmarkRuntime | None = None
        self._yolo: YoloRuntime | None = None

    @property
    def landmark(self) -> LandmarkRuntime:
        if self._landmark is None:
            self._landmark = LandmarkRuntime()
        return self._landmark

    @property
    def yolo(self) -> YoloRuntime:
        if self._yolo is None:
            self._yolo = YoloRuntime()
        return self._yolo


def decode_image(image_data: str) -> np.ndarray:
    payload = image_data
    if "," in image_data:
        _, payload = image_data.split(",", 1)
    try:
        image_bytes = base64.b64decode(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image data.")
    return frame


def resize_for_landmarks(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= LANDMARK_MAX_FRAME_WIDTH:
        return frame
    scale = LANDMARK_MAX_FRAME_WIDTH / float(width)
    return cv2.resize(
        frame,
        (LANDMARK_MAX_FRAME_WIDTH, max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def is_strong_landmark_result(result: dict) -> bool:
    quality = float(result.get("quality") or 0.0)
    confidence = float(result.get("confidence") or 0.0)
    margin = float(result.get("margin") or 0.0)
    hand_points = int(result.get("hand_points") or 0)
    return (
        quality >= LANDMARK_MIN_QUALITY
        and confidence >= LANDMARK_MIN_CONFIDENCE
        and margin >= LANDMARK_MIN_MARGIN
        and hand_points >= LANDMARK_MIN_HAND_POINTS
    )


def can_use_yolo_result(result: dict, *, min_confidence: float = YOLO_MIN_CONFIDENCE) -> bool:
    return float(result.get("confidence") or 0.0) >= min_confidence


def mirrored_frame(frame: np.ndarray) -> np.ndarray:
    return cv2.flip(frame, 1)


def prediction_score(result: dict) -> float:
    confidence = float(result.get("confidence") or 0.0)
    margin = float(result.get("margin") or 0.0)
    quality = float(result.get("quality") or 0.0)
    if result.get("backend") == "landmark":
        return confidence + 0.25 * min(max(margin, 0.0), 1.0) + 0.10 * min(max(quality, 0.0), 1.0)
    return confidence


def choose_best(results: list[dict]) -> dict | None:
    if not results:
        return None
    return max(results, key=prediction_score)


def landmark_candidates(frame: np.ndarray) -> list[dict]:
    candidates: list[dict] = []
    try:
        candidates.append(runtime.landmark.predict(frame, orientation="normal"))
    except HTTPException:
        pass
    return candidates


def yolo_candidates(frame: np.ndarray) -> list[dict]:
    candidates: list[dict] = []
    try:
        candidates.append(runtime.yolo.predict(frame, orientation="normal"))
    except HTTPException:
        pass
    return candidates


runtime = RuntimeRegistry()
app = FastAPI(title="Sign Language Converter")
app.mount("/web", StaticFiles(directory=WEB_ROOT), name="web")


@app.on_event("startup")
def warm_prediction_runtimes() -> None:
    runtime.landmark


@app.get("/")
def serve_index():
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/api/health")
def health_check():
    yolo_runtime_ready = True
    try:
        import ultralytics  # noqa: F401
    except Exception:
        yolo_runtime_ready = False

    return {
        "status": "ok",
        "landmark_model": LANDMARK_MODEL_PATH.exists(),
        "yolo_model": YOLO_MODEL_PATH.exists(),
        "yolo_runtime_ready": yolo_runtime_ready,
    }


@app.post("/api/predict")
def predict(payload: PredictRequest):
    started_at = time.perf_counter()
    try:
        frame = decode_image(payload.image_data)
        mode = payload.mode.strip().lower()

        if mode in {"landmark", "auto"}:
            landmark_options = landmark_candidates(frame)
            strong_landmarks = [item for item in landmark_options if is_strong_landmark_result(item)]
            result = choose_best(strong_landmarks)

            if result is None and mode == "auto":
                yolo_options = [
                    item
                    for item in yolo_candidates(frame)
                    if can_use_yolo_result(item, min_confidence=AUTO_YOLO_MIN_CONFIDENCE)
                ]
                yolo_result = choose_best(yolo_options)
                if yolo_result is not None:
                    result = {
                        **yolo_result,
                        "requested_mode": mode,
                        "backend": "yolo-fallback",
                    }

            if result is None:
                raise HTTPException(
                    status_code=422,
                    detail="Gesture is not clear enough for a reliable prediction.",
                )
        elif mode == "yolo":
            yolo_options = [item for item in yolo_candidates(frame) if can_use_yolo_result(item)]
            result = choose_best(yolo_options)
            if result is None:
                raise HTTPException(
                    status_code=422,
                    detail="YOLO confidence is too low for a reliable prediction.",
                )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported mode: {payload.mode}")

        return {
            "mode": result.get("backend", mode),
            "requested_mode": mode,
            "label": result["label"],
            "confidence": result["confidence"],
            "quality": result["quality"],
            "margin": result.get("margin"),
            "hand_points": result.get("hand_points"),
            "left_hand_points": result.get("left_hand_points"),
            "right_hand_points": result.get("right_hand_points"),
            "hand_boxes": result.get("hand_boxes", []),
            "orientation": result.get("orientation"),
            "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 1),
            "top": result.get("top", []),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


def main() -> None:
    import uvicorn

    uvicorn.run(
        "web_app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
