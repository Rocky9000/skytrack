"""Fine-tune a YOLO detector on VisDrone (run on a GPU: Colab, Kaggle, or local).

    python -m skytrack.visdrone /data/VisDrone /data/visdrone-yolo
    python scripts/train.py --data /data/visdrone-yolo/visdrone.yaml --model yolo11s.pt --imgsz 1024

High input resolution matters more than model size on VisDrone because most
objects are under 32 px. ``--imgsz 1024`` with a small model is a better
trade-off than a large model at 640.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", default="runs/visdrone")
    p.add_argument("--export-onnx", action="store_true")
    a = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.model)
    model.train(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, project=a.project,
        close_mosaic=10,      # turn mosaic off for the final epochs so boxes match real scale
        mosaic=1.0, mixup=0.1,
        patience=30, cos_lr=True, seed=0, deterministic=True,
    )
    metrics = model.val(data=a.data, imgsz=a.imgsz, split="val")
    summary = {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
               "model": a.model, "imgsz": a.imgsz, "epochs": a.epochs}
    out = Path(a.project) / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if a.export_onnx:
        print("exported:", model.export(format="onnx", imgsz=a.imgsz, simplify=True, dynamic=False))


if __name__ == "__main__":
    main()
