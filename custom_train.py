import argparse
import subprocess
import sys
from pathlib import Path

from gesture_pipeline import sanitize_label

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "custom_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train landmark, YOLO, or both models from a custom image dataset."
    )
    parser.add_argument(
        "--images-folder",
        help="Folder containing one subfolder per phrase/class with images inside.",
    )
    parser.add_argument(
        "--mode",
        choices=("landmark", "yolo", "both"),
        help="Which training pipeline to run. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--project-name",
        help="Short name used to build output file names.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Folder where custom datasets and trained models will be saved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing prepared datasets with the same output names.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="YOLO training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="YOLO image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="YOLO batch size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="YOLO device, for example cpu or cuda.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Use defaults for any missing prompts.",
    )
    return parser.parse_args()


def prompt_value(prompt_text: str, default: str) -> str:
    entered = input(f"{prompt_text} [{default}]: ").strip()
    return entered or default


def resolve_images_folder(args: argparse.Namespace) -> Path:
    if args.images_folder:
        images_folder = Path(args.images_folder).expanduser().resolve()
    elif args.yes:
        images_folder = (PROJECT_ROOT / "images for phrases").resolve()
    else:
        images_folder = Path(
            prompt_value(
                "Enter the custom images folder",
                str((PROJECT_ROOT / "images for phrases").resolve()),
            )
        ).expanduser().resolve()

    if not images_folder.exists() or not images_folder.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_folder}")
    return images_folder


def resolve_mode(args: argparse.Namespace) -> str:
    if args.mode:
        return args.mode
    if args.yes:
        return "both"
    return prompt_value("Choose training mode: landmark, yolo, or both", "both").lower()


def resolve_project_name(args: argparse.Namespace, images_folder: Path) -> str:
    if args.project_name:
        return sanitize_label(args.project_name)
    if args.yes:
        return sanitize_label(images_folder.name)
    return sanitize_label(
        prompt_value("Project name", sanitize_label(images_folder.name))
    )


def run_step(command: list[str]) -> None:
    print(f"[run] {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def train_landmark(
    *,
    images_folder: Path,
    output_root: Path,
    project_name: str,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    data_folder = output_root / f"{project_name}_landmark_data"
    model_output = output_root / f"{project_name}_landmark_model.pkl"
    encoder_output = output_root / f"{project_name}_landmark_label_encoder.pkl"

    command = [
        sys.executable,
        "create_text_gesture_data_from_images.py",
        "--images-folder",
        str(images_folder),
        "--data-folder",
        str(data_folder),
    ]
    if overwrite:
        command.append("--overwrite")
    run_step(command)

    run_step(
        [
            sys.executable,
            "train_text_gesture_model.py",
            "--data-folder",
            str(data_folder),
            "--backend",
            "sklearn",
            "--split-strategy",
            "frame",
            "--model-output",
            str(model_output),
            "--encoder-output",
            str(encoder_output),
        ]
    )
    return data_folder, model_output, encoder_output


def train_yolo(
    *,
    images_folder: Path,
    output_root: Path,
    project_name: str,
    overwrite: bool,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
) -> tuple[Path, Path]:
    dataset_folder = output_root / f"{project_name}_yolo_dataset"
    model_output = output_root / f"{project_name}_yolo_model.pt"

    command = [
        sys.executable,
        "prepare_yolo_classification_dataset.py",
        "--source-folder",
        str(images_folder),
        "--output-folder",
        str(dataset_folder),
    ]
    if overwrite:
        command.append("--overwrite")
    run_step(command)

    run_step(
        [
            sys.executable,
            "train_yolo_phrase_model.py",
            "--dataset-folder",
            str(dataset_folder),
            "--model",
            "yolov8n-cls.yaml",
            "--epochs",
            str(epochs),
            "--imgsz",
            str(imgsz),
            "--batch",
            str(batch),
            "--device",
            str(device),
            "--save-model-to",
            str(model_output),
        ]
    )
    return dataset_folder, model_output


def main() -> None:
    args = parse_args()
    images_folder = resolve_images_folder(args)
    mode = resolve_mode(args)
    if mode not in {"landmark", "yolo", "both"}:
        raise ValueError(f"Unsupported mode: {mode}")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    project_name = resolve_project_name(args, images_folder)

    print(f"[info] Images folder: {images_folder}")
    print(f"[info] Mode: {mode}")
    print(f"[info] Project name: {project_name}")
    print(f"[info] Output root: {output_root}")

    landmark_outputs = None
    yolo_outputs = None

    if mode in {"landmark", "both"}:
        landmark_outputs = train_landmark(
            images_folder=images_folder,
            output_root=output_root,
            project_name=project_name,
            overwrite=args.overwrite,
        )

    if mode in {"yolo", "both"}:
        yolo_outputs = train_yolo(
            images_folder=images_folder,
            output_root=output_root,
            project_name=project_name,
            overwrite=args.overwrite,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )

    print("\n[done] Custom training finished.")
    if landmark_outputs is not None:
        data_folder, model_output, encoder_output = landmark_outputs
        print("[done] Landmark outputs:")
        print(f"       data   : {data_folder}")
        print(f"       model  : {model_output}")
        print(f"       encoder: {encoder_output}")
        print(
            "[done] Run landmark prediction with:\n"
            f"       python predict_single_gesture.py --classifier-path \"{model_output}\" "
            f"--label-encoder-path \"{encoder_output}\""
        )
    if yolo_outputs is not None:
        dataset_folder, model_output = yolo_outputs
        class_map = dataset_folder / "class_name_map.json"
        print("[done] YOLO outputs:")
        print(f"       dataset : {dataset_folder}")
        print(f"       model   : {model_output}")
        print(f"       classmap: {class_map}")
        print(
            "[done] Run YOLO prediction with:\n"
            f"       python predict_phrase_yolo.py --model-path \"{model_output}\" "
            f"--class-map \"{class_map}\""
        )


if __name__ == "__main__":
    main()
