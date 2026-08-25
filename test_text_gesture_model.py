import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels

from gesture_dataset import load_dataset
from gesture_dataset import split_dataset_indices
from gesture_features import make_gesture_model_pipeline
from torch_gesture_model import predict_proba_numpy
from torch_gesture_model import train_torch_gesture_model

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the gesture model on extracted CSV data."
    )
    parser.add_argument(
        "--data-folder",
        default=str(PROJECT_ROOT / "text_phrase_image_data"),
        help="Folder containing extracted gesture CSV data.",
    )
    parser.add_argument(
        "--backend",
        choices=("torch", "sklearn"),
        default="sklearn",
        help="Classifier backend to evaluate.",
    )
    parser.add_argument(
        "--save-plot",
        default=str(PROJECT_ROOT / "confusion_matrix.png"),
        help="Where to save the confusion matrix image.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the confusion matrix window after saving it.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("video", "frame"),
        default="frame",
        help="Use a frame split for image data or a video-aware split for grouped video samples.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Average prediction probabilities over this many grouped samples when using video-aware evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the evaluation split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of samples reserved for evaluation.",
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


def train_and_predict_probabilities(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.backend == "sklearn":
        model = make_gesture_model_pipeline(
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_test)
        predictions = model.predict(X_test)
        eye = np.eye(len(np.unique(y_train)), dtype=np.float32)
        return eye[predictions]

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
    return predict_proba_numpy(
        artifacts.model,
        X_test,
        feature_mean=artifacts.feature_mean,
        feature_std=artifacts.feature_std,
        include_raw=artifacts.include_raw,
        device=artifacts.device,
    )


def main() -> None:
    args = parse_args()
    X, y, groups = load_dataset(args.data_folder)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    train_idx, test_idx = split_dataset_indices(
        X,
        y_encoded,
        groups,
        y,
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    probabilities = train_and_predict_probabilities(X_train, X_test, y_train, args)
    y_pred = probabilities.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"[done] Frame accuracy: {acc * 100:.2f}%")

    video_report_ready = args.split_strategy == "video"
    if video_report_ready:
        grouped_probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
        grouped_labels: dict[str, int] = {}
        for group, target, probability in zip(groups[test_idx], y_test, probabilities):
            grouped_probabilities[group].append(probability)
            grouped_labels[group] = int(target)

        video_truth = []
        video_pred = []
        first_window_truth = []
        first_window_pred = []
        for group in sorted(grouped_probabilities):
            video_truth.append(grouped_labels[group])
            mean_probability = np.mean(grouped_probabilities[group], axis=0)
            video_pred.append(int(mean_probability.argmax()))

            first_window_truth.append(grouped_labels[group])
            window_probability = np.mean(
                grouped_probabilities[group][: args.smoothing_window],
                axis=0,
            )
            first_window_pred.append(int(window_probability.argmax()))

        print(
            "[done] Full-video averaged accuracy: "
            f"{accuracy_score(video_truth, video_pred) * 100:.2f}%"
        )
        print(
            f"[done] First-{args.smoothing_window}-frame averaged accuracy: "
            f"{accuracy_score(first_window_truth, first_window_pred) * 100:.2f}%"
        )

    present_labels = unique_labels(y_test, y_pred)
    print("[done] Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=present_labels,
            target_names=label_encoder.inverse_transform(present_labels),
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=present_labels)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        xticklabels=label_encoder.inverse_transform(present_labels),
        yticklabels=label_encoder.inverse_transform(present_labels),
        cmap="Blues",
        fmt="d",
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(args.save_plot, dpi=200)
    print(f"[done] Saved confusion matrix to: {args.save_plot}")

    if args.show_plot:
        plt.show()


if __name__ == "__main__":
    main()
