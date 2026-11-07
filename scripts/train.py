"""Fine-tune YOLOv8 on RDD2022 road damage (run on a GPU: Colab, Kaggle, or local).

    python -m skytrack.rdd2022 /data/RDD2022 /data/rdd-yolo --countries India Japan Czech Norway
    python scripts/train.py --data /data/rdd-yolo/rdd2022.yaml --model yolov8s.pt --export-onnx

For the Jetson, export the trained weights on the device itself, since TensorRT
engines are tied to the GPU and TensorRT version they were built with:

    yolo export model=best.pt format=engine int8=True data=/data/rdd-yolo/rdd2022.yaml imgsz=640
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", default="runs/rdd2022")
    p.add_argument("--export-onnx", action="store_true")
    a = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.model)
    model.train(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, project=a.project,
        # Road damage has no canonical orientation from above, so vertical flips are valid augmentation.
        flipud=0.5, fliplr=0.5, degrees=10.0, mosaic=1.0, close_mosaic=10,
        hsv_v=0.5,  # brightness swings: shade from trees, wet asphalt
        patience=30, cos_lr=True, seed=0, deterministic=True,
    )
    metrics = model.val(data=a.data, imgsz=a.imgsz, split="val")
    summary = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_map50_95": {model.names[i]: float(v) for i, v in enumerate(metrics.box.maps)},
        "model": a.model, "imgsz": a.imgsz, "epochs": a.epochs,
    }
    out = Path(a.project) / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if a.export_onnx:
        print("exported:", model.export(format="onnx", imgsz=a.imgsz, simplify=True, dynamic=False))


if __name__ == "__main__":
    main()
