from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import joblib
import numpy as np


MANIFEST_PATH = ROOT / "models" / "model-manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def check_file(
    relative_path: str,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> Path:
    path = ROOT / relative_path

    if not path.exists():
        fail(
            f"Missing required artifact: {relative_path}\n"
            f"Expected location: {path}"
        )

    actual_size = path.stat().st_size

    if expected_size is not None and actual_size != expected_size:
        fail(
            f"Size mismatch for {relative_path}\n"
            f"Expected: {expected_size}\n"
            f"Actual:   {actual_size}"
        )

    if expected_sha256 is not None:
        actual_hash = sha256(path)

        if actual_hash != expected_sha256:
            fail(
                f"SHA-256 mismatch for {relative_path}\n"
                f"Expected: {expected_sha256}\n"
                f"Actual:   {actual_hash}"
            )

    print(f"[PASS] {relative_path}")
    return path


def validate_pipeline_contract(
    model,
    encoder,
    expected_input: int,
    expected_engineered: int,
    expected_classes: int,
) -> None:
    if not hasattr(model, "n_features_in_"):
        fail("Production model does not expose n_features_in_.")

    if model.n_features_in_ != expected_input:
        fail(
            "Production pipeline input mismatch: "
            f"expected {expected_input}, "
            f"got {model.n_features_in_}"
        )

    print(
        "[PASS] Production pipeline input contract: "
        f"{model.n_features_in_} features"
    )

    if not hasattr(model, "named_steps"):
        fail(
            "Production artifact is not the expected sklearn Pipeline."
        )

    required_steps = {"features", "classifier"}

    missing_steps = required_steps.difference(model.named_steps)

    if missing_steps:
        fail(
            "Pipeline is missing required steps: "
            f"{sorted(missing_steps)}"
        )

    feature_extractor = model.named_steps["features"]
    classifier = model.named_steps["classifier"]

    if type(feature_extractor).__name__ != "GestureFeatureExtractor":
        fail(
            "Unexpected feature extractor: "
            f"{type(feature_extractor).__name__}"
        )

    print(
        "[PASS] Pipeline contains GestureFeatureExtractor"
    )

    if type(classifier).__name__ != "ExtraTreesClassifier":
        fail(
            "Unexpected classifier: "
            f"{type(classifier).__name__}"
        )

    print(
        "[PASS] Pipeline contains ExtraTreesClassifier"
    )

    sample = np.zeros(
        (1, expected_input),
        dtype=np.float32,
    )

    transformed = feature_extractor.transform(sample)

    if transformed.shape != (
        1,
        expected_engineered,
    ):
        fail(
            "Feature-engineering contract mismatch: "
            f"expected {(1, expected_engineered)}, "
            f"got {transformed.shape}"
        )

    print(
        "[PASS] Feature extractor output contract: "
        f"{transformed.shape[1]} engineered features"
    )

    classifier_features = getattr(
        classifier,
        "n_features_in_",
        None,
    )

    if classifier_features != expected_engineered:
        fail(
            "Classifier input contract mismatch: "
            f"expected {expected_engineered}, "
            f"got {classifier_features}"
        )

    print(
        "[PASS] Classifier input contract: "
        f"{classifier_features} features"
    )

    classifier_classes = getattr(
        classifier,
        "n_classes_",
        None,
    )

    if classifier_classes != expected_classes:
        fail(
            "Classifier class-count mismatch: "
            f"expected {expected_classes}, "
            f"got {classifier_classes}"
        )

    print(
        "[PASS] Classifier output contract: "
        f"{classifier_classes} classes"
    )

    prediction = model.predict(sample)
    probabilities = model.predict_proba(sample)

    if probabilities.shape != (
        1,
        expected_classes,
    ):
        fail(
            "Probability output mismatch: "
            f"expected {(1, expected_classes)}, "
            f"got {probabilities.shape}"
        )

    print(
        "[PASS] Inference probability shape: "
        f"{probabilities.shape}"
    )

    if len(encoder.classes_) != expected_classes:
        fail(
            "Label encoder class-count mismatch: "
            f"expected {expected_classes}, "
            f"got {len(encoder.classes_)}"
        )

    print(
        "[PASS] Label encoder class count: "
        f"{len(encoder.classes_)} classes"
    )

    expected_indices = np.arange(expected_classes)

    if not np.array_equal(
        np.asarray(classifier.classes_),
        expected_indices,
    ):
        fail(
            "Classifier/label-encoder class-index contract mismatch.\n"
            f"Classifier classes: {classifier.classes_}\n"
            f"Expected indices:   {expected_indices}"
        )

    print(
        "[PASS] Classifier/label-encoder index contract: "
        f"{expected_classes} classes"
    )

    decoded = encoder.inverse_transform(prediction)

    print(
        "[PASS] Prediction successfully decoded: "
        f"{decoded[0]}"
    )


def main() -> None:
    print("=" * 70)
    print("SIGN LANGUAGE TO SPEECH - ARTIFACT VALIDATION")
    print("=" * 70)

    if not MANIFEST_PATH.exists():
        fail(f"Missing manifest: {MANIFEST_PATH}")

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    production = manifest["production_model"]
    encoder_info = manifest["label_encoder"]
    yolo_info = manifest["yolo_model"]
    landmarker_info = manifest["mediapipe_landmarker"]
    feature_contract = manifest["feature_contract"]

    print()
    print("Checking artifact integrity...")

    model_path = check_file(
        production["artifact"],
        production["size_bytes"],
        production["sha256"],
    )

    encoder_path = check_file(
        encoder_info["artifact"],
        encoder_info["size_bytes"],
        encoder_info["sha256"],
    )

    check_file(
        yolo_info["artifact"],
        yolo_info["size_bytes"],
        yolo_info["sha256"],
    )

    check_file(
        landmarker_info["artifact"],
        landmarker_info["size_bytes"],
        landmarker_info["sha256"],
    )

    print()
    print("Loading serialized production artifacts...")

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    expected_input = feature_contract["raw_input_features"]
    expected_engineered = feature_contract["engineered_output_features"]
    expected_classes = production["output_classes"]

    print()
    print("Validating production pipeline contract...")

    validate_pipeline_contract(
        model=model,
        encoder=encoder,
        expected_input=expected_input,
        expected_engineered=expected_engineered,
        expected_classes=expected_classes,
    )

    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Model artifact:   {production['artifact']}")
    print(f"Input features:   {expected_input}")
    print(f"Engineered:       {expected_engineered}")
    print(f"Classes:          {expected_classes}")
    print(f"Prediction:       {model.predict(np.zeros((1, expected_input), dtype=np.float32))[0]}")
    print("Artifact status:  VALID")
    print("Pipeline status:  VALID")
    print("Status:           PASS")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FAIL] Unexpected validation error: {exc}")
        sys.exit(1)