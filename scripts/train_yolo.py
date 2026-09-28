import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=416
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4
    )

    parser.add_argument(
        "--device",
        default="cpu"
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    data_yaml = root / "data" / "yolo" / "data.yaml"

    models_dir = root / "models"

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Could not find {data_yaml}"
        )

    print("\n====================================")
    print("YOLOv8 Muzzle Training")
    print("====================================")

    print(f"Device : {args.device}")
    print(f"Epochs : {args.epochs}")
    print(f"Image  : {args.imgsz}")
    print(f"Batch  : {args.batch}")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,

        # CPU / Windows friendly
        workers=0,

        project=str(root / "runs"),
        name="muzzle_yolov8n",

        pretrained=True,

        patience=10,

        save=True,
        plots=True,

        verbose=True,
    )

    best_model = (
        root
        / "runs"
        / "muzzle_yolov8n"
        / "weights"
        / "best.pt"
    )

    destination = models_dir / "yolo_muzzle.pt"

    if best_model.exists():

        shutil.copy2(
            best_model,
            destination
        )

        print("\nBest YOLO model copied to:")
        print(destination)

    else:

        print(
            "\nWARNING: best.pt was not found."
        )


if __name__ == "__main__":
    main()