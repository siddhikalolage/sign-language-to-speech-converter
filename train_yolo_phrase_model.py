import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def configure_ultralytics_workspace() -> Path:
    config_dir = (PROJECT_ROOT / ".ultralytics").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    return config_dir


def patch_ultralytics_verify_images() -> None:
    """Avoid Windows thread-pool verification issues in this workspace."""
    import ultralytics.data.dataset as dataset_mod
    from ultralytics.data.utils import (
        load_dataset_cache_file,
        save_dataset_cache_file,
        verify_image,
    )

    def verify_images_sequential(self):
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")

        try:
            dataset_mod.check_file_speeds([file for (file, _) in self.samples[:5]], prefix=self.prefix)
            cache = load_dataset_cache_file(path)
            assert cache["version"] == dataset_mod.DATASET_CACHE_VERSION
            assert cache["hash"] == dataset_mod.get_hash([x[0] for x in self.samples])
            nf, nc, n, samples = cache.pop("results")
            if dataset_mod.LOCAL_RANK in {-1, 0}:
                status = f"{desc} {nf} images, {nc} corrupt"
                dataset_mod.TQDM(None, desc=status, total=n, initial=n)
                if cache["msgs"]:
                    dataset_mod.LOGGER.info("\n".join(cache["msgs"]))
            return samples
        except (FileNotFoundError, AssertionError, AttributeError, KeyError):
            nf, nc, msgs, samples, cache_payload = 0, 0, [], [], {}
            iterator = dataset_mod.TQDM(self.samples, desc=desc, total=len(self.samples))
            for sample in iterator:
                verified_sample, nf_f, nc_f, msg = verify_image((sample, self.prefix))
                if nf_f:
                    samples.append(verified_sample)
                if msg:
                    msgs.append(msg)
                nf += nf_f
                nc += nc_f
                iterator.desc = f"{desc} {nf} images, {nc} corrupt"
            iterator.close()
            if msgs:
                dataset_mod.LOGGER.info("\n".join(msgs))
            cache_payload["hash"] = dataset_mod.get_hash([x[0] for x in self.samples])
            cache_payload["results"] = nf, nc, len(samples), samples
            cache_payload["msgs"] = msgs
            save_dataset_cache_file(self.prefix, path, cache_payload, dataset_mod.DATASET_CACHE_VERSION)
            return samples

    dataset_mod.ClassificationDataset.verify_images = verify_images_sequential


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 classification model on a prepared image dataset."
    )
    parser.add_argument(
        "--dataset-folder",
        default=str(PROJECT_ROOT / "yolo_phrase_dataset"),
        help="Prepared YOLO classification dataset root with train/val folders.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n-cls.yaml",
        help="YOLO classification architecture or checkpoint to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Image size used for classification training.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Training device. Use cpu or a CUDA device id if available.",
    )
    parser.add_argument(
        "--project",
        default=str(PROJECT_ROOT / "runs" / "yolo_phrase_cls"),
        help="Folder where YOLO training runs are stored.",
    )
    parser.add_argument(
        "--name",
        default="phrase_yolo",
        help="Run name inside the YOLO project folder.",
    )
    parser.add_argument(
        "--save-model-to",
        default=str(PROJECT_ROOT / "text_phrase_yolo_cls.pt"),
        help="Where to copy the best YOLO checkpoint after training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from a previous Ultralytics checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_folder = Path(args.dataset_folder)
    if not dataset_folder.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {dataset_folder}\n"
            "Run prepare_yolo_classification_dataset.py first."
        )

    config_dir = configure_ultralytics_workspace()
    from ultralytics import YOLO

    patch_ultralytics_verify_images()
    print(f"[info] Using Ultralytics config dir: {config_dir}")
    model = YOLO(args.model)
    train_kwargs = {
        "data": str(dataset_folder),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "verbose": True,
    }
    if args.resume:
        train_kwargs = {"resume": True}

    train_results = model.train(**train_kwargs)

    best_ckpt = Path(train_results.save_dir) / "weights" / "best.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")

    target_path = Path(args.save_model_to)
    target_path.write_bytes(best_ckpt.read_bytes())

    metrics = model.val(
        data=str(dataset_folder),
        split="test" if (dataset_folder / "test").exists() else "val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
    )
    metrics_path = target_path.with_suffix(".metrics.json")
    metrics_payload = {
        "top1": float(getattr(metrics, "top1", 0.0)),
        "top5": float(getattr(metrics, "top5", 0.0)),
        "fitness": float(getattr(metrics, "fitness", 0.0)),
        "save_dir": str(getattr(metrics, "save_dir", "")),
        "source_run": str(train_results.save_dir),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("[done] YOLO training finished.")
    print(f"[done] Best checkpoint copied to: {target_path}")
    print(f"[done] Metrics saved to: {metrics_path}")
    print(f"[done] Top-1 accuracy: {metrics_payload['top1']:.4f}")
    print(f"[done] Top-5 accuracy: {metrics_payload['top5']:.4f}")


if __name__ == "__main__":
    main()
