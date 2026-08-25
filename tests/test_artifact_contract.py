from pathlib import Path
import json

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "model-manifest.json"
MODEL = ROOT / "text_phrase_image_model.pkl"
LABEL_ENCODER = ROOT / "text_phrase_image_label_encoder.pkl"


def load_manifest():
    assert MANIFEST.exists(), "Production model manifest is missing."
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def load_production_model():
    assert MODEL.exists(), (
        "Production model artifact is missing. "
        "Place text_phrase_image_model.pkl in the project root."
    )
    return joblib.load(MODEL)


def load_label_encoder():
    assert LABEL_ENCODER.exists(), (
        "Production label encoder is missing. "
        "Place text_phrase_image_label_encoder.pkl in the project root."
    )
    return joblib.load(LABEL_ENCODER)


def test_model_manifest_exists():
    assert MANIFEST.exists()


def test_model_manifest_is_valid_json():
    data = load_manifest()

    assert data["schema_version"] == "1.0"
    assert data["project"] == "sign-language-to-speech-converter"


def test_production_model_contract():
    production = load_manifest()["production_model"]

    assert production["artifact"] == "text_phrase_image_model.pkl"
    assert production["size_bytes"] == 31561578
    assert production["framework"] == "scikit-learn"
    assert production["framework_version_validated"] == "1.6.1"
    assert production["input_features"] == 144
    assert production["engineered_features"] == 328
    assert production["output_classes"] == 43

    classifier = production["classifier"]

    assert classifier["type"] == "ExtraTreesClassifier"
    assert classifier["n_estimators"] == 250
    assert classifier["random_state"] == 42


def test_label_encoder_contract():
    encoder = load_manifest()["label_encoder"]

    assert encoder["artifact"] == "text_phrase_image_label_encoder.pkl"
    assert encoder["status"] == "tracked_artifact"
    assert encoder["classes"] == 43


def test_yolo_artifact_contract():
    yolo = load_manifest()["yolo_model"]

    assert yolo["artifact"] == "text_phrase_yolo_cls.pt"
    assert yolo["status"] == "tracked_artifact"


def test_mediapipe_artifact_contract():
    landmarker = load_manifest()["mediapipe_landmarker"]

    assert landmarker["artifact"] == "holistic_landmarker.task"
    assert landmarker["status"] == "tracked_artifact"


def test_feature_contract():
    contract = load_manifest()["feature_contract"]

    assert contract["raw_input_features"] == 144
    assert contract["engineered_output_features"] == 328
    assert contract["include_raw"] is False


def test_production_model_can_be_loaded():
    model = load_production_model()

    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_production_model_input_contract():
    model = load_production_model()

    sample = np.zeros((1, 144), dtype=np.float32)

    probabilities = model.predict_proba(sample)

    assert probabilities.shape == (1, 43)
    assert np.isfinite(probabilities).all()


def test_production_model_probability_contract():
    model = load_production_model()

    sample = np.zeros((1, 144), dtype=np.float32)

    probabilities = model.predict_proba(sample)

    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)

    probability_sum = probabilities.sum(axis=1)

    assert np.allclose(probability_sum, 1.0, atol=1e-6)


def test_production_model_prediction_contract():
    model = load_production_model()

    sample = np.zeros((1, 144), dtype=np.float32)

    prediction = model.predict(sample)

    assert prediction.shape == (1,)
    assert isinstance(prediction[0], (int, np.integer))


def test_label_encoder_can_be_loaded():
    encoder = load_label_encoder()

    assert encoder is not None
    assert hasattr(encoder, "classes_")
    assert len(encoder.classes_) == 43


def test_model_and_label_encoder_class_alignment():
    model = load_production_model()
    encoder = load_label_encoder()

    sample = np.zeros((1, 144), dtype=np.float32)

    prediction = model.predict(sample)
    decoded = encoder.inverse_transform(prediction)

    assert decoded.shape == (1,)
    assert isinstance(decoded[0], str)
    assert decoded[0] in encoder.classes_