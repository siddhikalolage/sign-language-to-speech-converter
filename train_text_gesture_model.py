import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from gesture_dataset import load_dataset
from gesture_dataset import split_dataset_indices
from gesture_features import make_gesture_model_pipeline
from torch_gesture_model import DEFAULT_TORCH_MODEL_PATH
from torch_gesture_model import predict_proba_numpy
from torch_gesture_model import save_torch_gesture_checkpoint
from torch_gesture_model import train_torch_gesture_model

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a gesture recognition model from extracted landmark CSV files."
    )
    parser.add_argument(
        "--data-folder",
        default=str(PROJECT_ROOT / "text_phrase_image_data"),
        help="Folder containing the extracted gesture CSVs.",
    )
    parser.add_argument(
        "--backend",
        choices=("torch", "sklearn"),
        default="sklearn",
        help="Classifier backend to train.",
    )
    parser.add_argument(
        "--model-output",
        help="Where to save the trained classifier. Defaults to .pt for torch and .pkl for sklearn.",
    )
    parser.add_argument(
        "--encoder-output",
        default=str(PROJECT_ROOT / "text_phrase_image_label_encoder.pkl"),
        help="Where to save the fitted label encoder.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split and classifier.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("video", "frame"),
        default="frame",
        help="Use a frame split for image data or a video-aware split for grouped video samples.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=350,
        help="Number of trees in the sklearn ensemble.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
        help="Number of training epochs for the torch backend.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for the torch backend.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the torch backend.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay for the torch backend.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.25,
        help="Dropout used by the torch MLP.",
    )
    parser.add_argument(
        "--hidden-dims",
        default="256,128",
        help="Comma-separated hidden layer sizes for the torch MLP.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use: auto, cpu, or cuda.",
    )
    return parser.parse_args()


def resolve_model_output(args: argparse.Namespace) -> str:
    if args.model_output:
        return args.model_output
    if args.backend == "torch":
        return str(DEFAULT_TORCH_MODEL_PATH)
    return str(PROJECT_ROOT / "text_phrase_image_model.pkl")


def evaluate_sklearn_backend(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    random_state: int,
    n_estimators: int,
) -> float:
    model = make_gesture_model_pipeline(
        random_state=random_state,
        n_estimators=n_estimators,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return float(accuracy_score(y_test, model.predict(X_test)))


def evaluate_torch_backend(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
) -> float:
    artifacts = train_torch_gesture_model(
        X_train,
        y_train,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        random_state=args.random_state,
        verbose=True,
    )
    probabilities = predict_proba_numpy(
        artifacts.model,
        X_test,
        feature_mean=artifacts.feature_mean,
        feature_std=artifacts.feature_std,
        include_raw=artifacts.include_raw,
        device=artifacts.device,
    )
    predictions = probabilities.argmax(axis=1)
    return float(accuracy_score(y_test, predictions))


def train_final_sklearn_model(
    X: np.ndarray,
    y_encoded: np.ndarray,
    *,
    random_state: int,
    n_estimators: int,
):
    model = make_gesture_model_pipeline(
        random_state=random_state,
        n_estimators=n_estimators,
        n_jobs=1,
    )
    model.fit(X, y_encoded)
    return model


def main() -> None:
    args = parse_args()
    model_output = resolve_model_output(args)
    data_folder = Path(args.data_folder)
    X, y, groups = load_dataset(data_folder)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    train_indices, test_indices = split_dataset_indices(
        X,
        y_encoded,
        groups,
        y,
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y_encoded[train_indices]
    y_test = y_encoded[test_indices]

    if args.backend == "torch":
        accuracy = evaluate_torch_backend(X_train, X_test, y_train, y_test, args)
    else:
        accuracy = evaluate_sklearn_backend(
            X_train,
            X_test,
            y_train,
            y_test,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
        )

    print("[done] Evaluation finished.")
    print(f"[done] Test accuracy: {accuracy * 100:.2f}%")

    if args.backend == "torch":
        final_artifacts = train_torch_gesture_model(
            X,
            y_encoded,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            random_state=args.random_state,
            verbose=True,
        )
        save_torch_gesture_checkpoint(
            model_output,
            final_artifacts,
            class_names=label_encoder.classes_,
        )
    else:
        final_model = train_final_sklearn_model(
            X,
            y_encoded,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
        )
        joblib.dump(final_model, model_output)

    joblib.dump(label_encoder, args.encoder_output)
    print(f"[done] Saved model to: {model_output}")
    print(f"[done] Saved label encoder to: {args.encoder_output}")


if __name__ == "__main__":
    main()
