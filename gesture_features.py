import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline

from gesture_pipeline import EXPECTED_FEATURE_COUNT

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
HAND_ANGLE_TRIPLETS = [
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
    (0, 5, 6),
    (5, 6, 7),
    (6, 7, 8),
    (0, 9, 10),
    (9, 10, 11),
    (10, 11, 12),
    (0, 13, 14),
    (13, 14, 15),
    (14, 15, 16),
    (0, 17, 18),
    (17, 18, 19),
    (18, 19, 20),
]


def _safe_norm(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.where(norms < 1e-6, 1.0, norms)


class GestureFeatureExtractor(BaseEstimator, TransformerMixin):
    """Build hand-centered features from landmarks while ignoring image background."""

    def __init__(self, include_raw: bool = False):
        self.include_raw = include_raw

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Expected input shape (*, {EXPECTED_FEATURE_COUNT}), got {array.shape}"
            )
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X) -> np.ndarray:
        array = np.asarray(X, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Expected input shape (*, {EXPECTED_FEATURE_COUNT}), got {array.shape}"
            )

        face = array[:, :18].reshape(-1, 6, 3)
        left_hand = array[:, 18:81].reshape(-1, 21, 3)
        right_hand = array[:, 81:].reshape(-1, 21, 3)

        reference_center, hand_scale = self._reference_frame(face, left_hand, right_hand)
        left_features = self._hand_features(left_hand, reference_center, hand_scale)
        right_features = self._hand_features(right_hand, reference_center, hand_scale)

        feature_blocks = [self._normalized_raw(array, reference_center, hand_scale)] if self.include_raw else []
        feature_blocks.append(
            np.concatenate(
                [
                    left_features,
                    right_features,
                    self._two_hand_features(left_hand, right_hand, hand_scale),
                ],
                axis=1,
            )
        )
        return np.concatenate(feature_blocks, axis=1)

    def _reference_frame(
        self,
        face: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        hand_points = np.concatenate([left_hand, right_hand], axis=1)
        hand_mask = self._point_mask(hand_points)
        center = self._masked_mean(hand_points, hand_mask)
        scale = self._masked_extent(hand_points, hand_mask)
        return center, np.where(scale < 1e-6, 1.0, scale)

    def _normalized_raw(
        self,
        array: np.ndarray,
        reference_center: np.ndarray,
        hand_scale: np.ndarray,
    ) -> np.ndarray:
        points = array.reshape(array.shape[0], -1, 3)
        mask = self._point_mask(points)
        normalized = (points - reference_center[:, None, :]) / hand_scale[:, None, :]
        normalized = np.where(mask[:, :, None], normalized, 0.0)
        return normalized.reshape(array.shape[0], -1)

    def _hand_features(
        self,
        hand: np.ndarray,
        reference_center: np.ndarray,
        hand_scale: np.ndarray,
    ) -> np.ndarray:
        point_mask = self._point_mask(hand)
        hand_present = point_mask.any(axis=1, keepdims=True).astype(np.float32)
        wrist = hand[:, 0:1, :]
        centroid = self._masked_mean(hand, point_mask)
        tips = hand[:, FINGERTIP_INDICES, :]
        hand_extent = self._masked_extent(hand, point_mask)

        hand_relative_to_reference = (hand - reference_center[:, None, :]) / hand_scale[:, None, :]
        hand_relative_to_reference = np.where(point_mask[:, :, None], hand_relative_to_reference, 0.0)
        hand_relative_to_wrist = (hand - wrist) / hand_extent[:, None, :]
        hand_relative_to_wrist = np.where(point_mask[:, :, None], hand_relative_to_wrist, 0.0)

        centroid_relative = (centroid - reference_center) / hand_scale
        wrist_relative = (wrist[:, 0, :] - reference_center) / hand_scale
        tip_wrist_distances = np.linalg.norm(tips - wrist, axis=2) / hand_extent
        tip_wrist_distances = np.where(point_mask[:, FINGERTIP_INDICES], tip_wrist_distances, 0.0)
        joint_angles = self._joint_angle_features(hand)
        features = np.concatenate(
            [
                hand_present,
                centroid[:, :2],
                wrist[:, 0, :2],
                hand_relative_to_reference.reshape(hand.shape[0], -1),
                hand_relative_to_wrist.reshape(hand.shape[0], -1),
                centroid_relative,
                wrist_relative,
                hand_extent,
                tip_wrist_distances,
                joint_angles,
            ],
            axis=1,
        )
        return np.where(hand_present, features, 0.0)

    def _two_hand_features(
        self,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        body_scale: np.ndarray,
    ) -> np.ndarray:
        left_mask = self._point_mask(left_hand)
        right_mask = self._point_mask(right_hand)
        both_present = (left_mask.any(axis=1) & right_mask.any(axis=1))[:, None]

        left_wrist = left_hand[:, 0, :]
        right_wrist = right_hand[:, 0, :]
        left_centroid = self._masked_mean(left_hand, left_mask)
        right_centroid = self._masked_mean(right_hand, right_mask)
        left_tips = left_hand[:, FINGERTIP_INDICES, :]
        right_tips = right_hand[:, FINGERTIP_INDICES, :]

        wrist_delta = (left_wrist - right_wrist) / body_scale
        centroid_delta = (left_centroid - right_centroid) / body_scale
        tip_distances = np.linalg.norm(left_tips - right_tips, axis=2) / body_scale
        features = np.concatenate(
            [
                both_present.astype(np.float32),
                wrist_delta,
                centroid_delta,
                tip_distances,
            ],
            axis=1,
        )
        return np.where(both_present, features, 0.0)

    def _point_mask(self, points: np.ndarray) -> np.ndarray:
        return np.any(np.abs(points) > 1e-6, axis=2)

    def _masked_mean(self, points: np.ndarray, mask: np.ndarray) -> np.ndarray:
        weights = mask.astype(np.float32)
        count = weights.sum(axis=1, keepdims=True)
        count = np.where(count < 1.0, 1.0, count)
        return (points * weights[:, :, None]).sum(axis=1) / count

    def _masked_extent(self, points: np.ndarray, mask: np.ndarray) -> np.ndarray:
        has_points = mask.any(axis=1, keepdims=True)
        mins = np.min(np.where(mask[:, :, None], points, np.inf), axis=1)
        maxes = np.max(np.where(mask[:, :, None], points, -np.inf), axis=1)
        extents = np.where(has_points, maxes - mins, 1.0)
        scale = np.maximum.reduce([extents[:, 0], extents[:, 1], np.full(points.shape[0], 1e-3)])
        return scale.reshape(-1, 1)

    def _joint_angle_features(self, hand: np.ndarray) -> np.ndarray:
        features: list[np.ndarray] = []
        mask = self._point_mask(hand)
        for start_index, pivot_index, end_index in HAND_ANGLE_TRIPLETS:
            incoming = hand[:, start_index] - hand[:, pivot_index]
            outgoing = hand[:, end_index] - hand[:, pivot_index]
            valid = mask[:, start_index] & mask[:, pivot_index] & mask[:, end_index]
            cosine = (incoming * outgoing).sum(axis=1, keepdims=True) / (
                _safe_norm(incoming) * _safe_norm(outgoing)
            )
            features.append(np.where(valid[:, None], np.clip(cosine, -1.0, 1.0), 0.0))
        return np.concatenate(features, axis=1)


def make_gesture_model_pipeline(
    *,
    random_state: int = 42,
    n_estimators: int = 350,
    n_jobs: int = 1,
    include_raw: bool = False,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", GestureFeatureExtractor(include_raw=include_raw)),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )
