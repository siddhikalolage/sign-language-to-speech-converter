import numpy as np
import pytest

from gesture_features import (
    GestureFeatureExtractor,
    make_gesture_model_pipeline,
)
from gesture_pipeline import EXPECTED_FEATURE_COUNT


RAW_FEATURE_COUNT = 144
ENGINEERED_FEATURE_COUNT = 328


def make_valid_landmarks(
    samples: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """
    Create deterministic synthetic MediaPipe-style landmark vectors.

    Layout:
        6 face landmarks  -> 18 values
        21 left-hand      -> 63 values
        21 right-hand     -> 63 values

    Total:
        18 + 63 + 63 = 144
    """
    rng = np.random.default_rng(seed)

    landmarks = rng.uniform(
        low=0.1,
        high=0.9,
        size=(samples, RAW_FEATURE_COUNT),
    ).astype(np.float32)

    return landmarks


def make_missing_hand_landmarks(
    samples: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """
    Create valid landmarks where the right hand is completely absent.
    """
    landmarks = make_valid_landmarks(samples=samples, seed=seed)

    # Right hand occupies the final 63 values.
    landmarks[:, 81:] = 0.0

    return landmarks


def test_raw_feature_contract():
    """The extractor must accept exactly 144 raw landmark features."""
    X = make_valid_landmarks()

    extractor = GestureFeatureExtractor()

    extractor.fit(X)

    assert extractor.n_features_in_ == EXPECTED_FEATURE_COUNT
    assert X.shape == (8, RAW_FEATURE_COUNT)


def test_engineered_feature_count():
    """The production feature extractor must produce exactly 328 features."""
    X = make_valid_landmarks()

    extractor = GestureFeatureExtractor()

    transformed = extractor.fit_transform(X)

    assert transformed.shape == (
        X.shape[0],
        ENGINEERED_FEATURE_COUNT,
    )


def test_engineered_features_are_finite():
    """
    Feature engineering must never produce NaN or infinite values.

    This protects the downstream ExtraTrees classifier from invalid
    numerical input caused by missing or degenerate landmarks.
    """
    X = make_valid_landmarks()

    extractor = GestureFeatureExtractor()

    transformed = extractor.fit_transform(X)

    assert np.isfinite(transformed).all()


def test_missing_hand_is_supported():
    """
    A completely missing hand should produce a valid feature vector
    rather than crashing or producing NaN/inf.
    """
    X = make_missing_hand_landmarks()

    extractor = GestureFeatureExtractor()

    transformed = extractor.fit_transform(X)

    assert transformed.shape == (
        X.shape[0],
        ENGINEERED_FEATURE_COUNT,
    )

    assert np.isfinite(transformed).all()


def test_invalid_input_feature_count_is_rejected():
    """Incorrect landmark dimensionality must fail loudly."""
    X = np.zeros((4, RAW_FEATURE_COUNT - 1), dtype=np.float32)

    extractor = GestureFeatureExtractor()

    with pytest.raises(ValueError, match="Expected input shape"):
        extractor.fit(X)


def test_pipeline_can_train_and_predict():
    """
    Verify that the engineered features are compatible with the
    ExtraTrees production-style pipeline.
    """
    X = make_valid_landmarks(samples=12)

    y = np.array(
        [
            "hello",
            "hello",
            "hello",
            "hello",
            "hello",
            "hello",
            "thanks",
            "thanks",
            "thanks",
            "thanks",
            "thanks",
            "thanks",
        ]
    )

    pipeline = make_gesture_model_pipeline(
        random_state=42,
        n_estimators=25,
        n_jobs=1,
    )

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    assert predictions.shape == (12,)
    assert probabilities.shape == (12, 2)

    assert np.isfinite(probabilities).all()
    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    )


def test_pipeline_is_deterministic():
    """
    With a fixed random state, two independently trained pipelines
    should produce identical predictions on the same data.
    """
    X = make_valid_landmarks(samples=16)

    y = np.array(
        [
            "hello",
            "hello",
            "hello",
            "hello",
            "hello",
            "hello",
            "hello",
            "hello",
            "thanks",
            "thanks",
            "thanks",
            "thanks",
            "thanks",
            "thanks",
            "thanks",
            "thanks",
        ]
    )

    pipeline_a = make_gesture_model_pipeline(
        random_state=42,
        n_estimators=25,
        n_jobs=1,
    )

    pipeline_b = make_gesture_model_pipeline(
        random_state=42,
        n_estimators=25,
        n_jobs=1,
    )

    pipeline_a.fit(X, y)
    pipeline_b.fit(X, y)

    predictions_a = pipeline_a.predict(X)
    predictions_b = pipeline_b.predict(X)

    assert np.array_equal(
        predictions_a,
        predictions_b,
    )