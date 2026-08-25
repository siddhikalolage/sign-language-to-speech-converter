from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from gesture_features import GestureFeatureExtractor
from gesture_pipeline import EXPECTED_FEATURE_COUNT

DEFAULT_TORCH_MODEL_PATH = Path("text_gesture_model.pt")


def parse_hidden_dims(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(int(item) for item in value)
    dims = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not dims:
        raise ValueError("At least one hidden layer size is required.")
    return tuple(dims)


class GestureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], class_count: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, class_count))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def compute_feature_matrix(landmarks: np.ndarray, *, include_raw: bool = True) -> np.ndarray:
    extractor = GestureFeatureExtractor(include_raw=include_raw)
    extractor.fit(landmarks)
    return extractor.transform(landmarks).astype(np.float32)


def compute_standardization_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def standardize_features(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32)


def choose_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TorchTrainingArtifacts:
    model: GestureMLP
    feature_mean: np.ndarray
    feature_std: np.ndarray
    input_dim: int
    hidden_dims: tuple[int, ...]
    dropout: float
    include_raw: bool
    device: str


def train_torch_gesture_model(
    landmarks: np.ndarray,
    targets: np.ndarray,
    *,
    hidden_dims: tuple[int, ...] = (256, 128),
    dropout: float = 0.25,
    batch_size: int = 256,
    epochs: int = 35,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    include_raw: bool = True,
    device: str = "auto",
    random_state: int = 42,
    verbose: bool = True,
) -> TorchTrainingArtifacts:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    resolved_device = choose_device(device)
    feature_matrix = compute_feature_matrix(landmarks, include_raw=include_raw)
    feature_mean, feature_std = compute_standardization_stats(feature_matrix)
    normalized_features = standardize_features(feature_matrix, feature_mean, feature_std)

    dataset = TensorDataset(
        torch.from_numpy(normalized_features),
        torch.from_numpy(targets.astype(np.int64)),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        drop_last=False,
    )

    hidden_dims = parse_hidden_dims(hidden_dims)
    model = GestureMLP(
        input_dim=normalized_features.shape[1],
        hidden_dims=hidden_dims,
        class_count=int(np.max(targets)) + 1,
        dropout=dropout,
    ).to(resolved_device)

    class_counts = np.bincount(targets.astype(np.int64))
    class_weights = len(targets) / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=resolved_device),
        label_smoothing=0.05,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
    )

    for epoch_index in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        sample_count = 0

        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(resolved_device)
            batch_targets = batch_targets.to(resolved_device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch_targets)
            correct += int((logits.argmax(dim=1) == batch_targets).sum().item())
            sample_count += len(batch_targets)

        scheduler.step()
        if verbose and (epoch_index == 0 or (epoch_index + 1) % 5 == 0 or epoch_index + 1 == epochs):
            print(
                f"[info] Epoch {epoch_index + 1:02d}/{epochs}: "
                f"loss={total_loss / max(1, sample_count):.4f}, "
                f"train_acc={correct / max(1, sample_count) * 100:.2f}%"
            )

    return TorchTrainingArtifacts(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        input_dim=normalized_features.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
        include_raw=include_raw,
        device=resolved_device,
    )


@torch.inference_mode()
def predict_proba_numpy(
    model: GestureMLP,
    landmarks: np.ndarray,
    *,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    include_raw: bool = True,
    device: str = "cpu",
    batch_size: int = 1024,
) -> np.ndarray:
    feature_matrix = compute_feature_matrix(landmarks, include_raw=include_raw)
    normalized_features = standardize_features(feature_matrix, feature_mean, feature_std)

    model.eval()
    probabilities: list[np.ndarray] = []
    for start_index in range(0, len(normalized_features), batch_size):
        batch = torch.from_numpy(normalized_features[start_index : start_index + batch_size]).to(device)
        logits = model(batch)
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def save_torch_gesture_checkpoint(
    path: str | Path,
    artifacts: TorchTrainingArtifacts,
    *,
    class_names: np.ndarray,
) -> None:
    checkpoint = {
        "backend": "torch",
        "expected_landmark_features": EXPECTED_FEATURE_COUNT,
        "input_dim": artifacts.input_dim,
        "hidden_dims": list(artifacts.hidden_dims),
        "dropout": artifacts.dropout,
        "include_raw": artifacts.include_raw,
        "class_names": list(class_names),
        "feature_mean": artifacts.feature_mean,
        "feature_std": artifacts.feature_std,
        "state_dict": {key: value.detach().cpu() for key, value in artifacts.model.state_dict().items()},
    }
    torch.save(checkpoint, path)


class TorchGestureRuntime:
    def __init__(self, checkpoint: dict, *, device: str = "auto"):
        self.device = choose_device(device)
        self.class_names = np.array(checkpoint["class_names"])
        self.include_raw = bool(checkpoint.get("include_raw", True))
        self.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        self.n_features_in_ = int(
            checkpoint.get("expected_landmark_features", EXPECTED_FEATURE_COUNT)
        )
        self.model = GestureMLP(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dims=parse_hidden_dims(checkpoint["hidden_dims"]),
            class_count=len(self.class_names),
            dropout=float(checkpoint["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict_proba(self, landmarks) -> np.ndarray:
        array = np.asarray(landmarks, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return predict_proba_numpy(
            self.model,
            array,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
            include_raw=self.include_raw,
            device=self.device,
        )

    def predict(self, landmarks) -> np.ndarray:
        probabilities = self.predict_proba(landmarks)
        return probabilities.argmax(axis=1)


def load_torch_gesture_runtime(path: str | Path, *, device: str = "auto") -> TorchGestureRuntime:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return TorchGestureRuntime(checkpoint, device=device)
