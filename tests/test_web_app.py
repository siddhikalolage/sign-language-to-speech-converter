import base64

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

import web_app


def test_health_contract():
    body = web_app.health_check()

    assert body["status"] == "ok"
    assert isinstance(body["landmark_model"], bool)
    assert isinstance(body["yolo_model"], bool)
    assert isinstance(body["yolo_runtime_ready"], bool)


def test_invalid_image_payload():
    payload = web_app.PredictRequest(
        image_data="not-valid-base64",
        mode="auto",
    )

    with pytest.raises(HTTPException) as exc_info:
        web_app.predict(payload)

    assert exc_info.value.status_code == 400
    assert "Invalid image payload" in str(exc_info.value.detail)


def test_invalid_image_bytes():
    encoded = base64.b64encode(b"this-is-not-an-image").decode("ascii")

    payload = web_app.PredictRequest(
        image_data=encoded,
        mode="auto",
    )

    with pytest.raises(HTTPException) as exc_info:
        web_app.predict(payload)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Could not decode image data."


def test_unsupported_prediction_mode():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    payload = web_app.PredictRequest(
        image_data=base64.b64encode(encoded.tobytes()).decode("ascii"),
        mode="unsupported-mode",
    )

    with pytest.raises(HTTPException) as exc_info:
        web_app.predict(payload)

    assert exc_info.value.status_code == 400
    assert "Unsupported mode" in str(exc_info.value.detail)
