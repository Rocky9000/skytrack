"""Convert VisDrone2019-DET annotations to YOLO format.

VisDrone rows: ``left,top,width,height,score,category,truncation,occlusion``.
Category 0 marks "ignored regions" and 11 is "others"; both are dropped, and
rows with score 0 are ignored regions too. Categories 1-10 map to 0-9.

Download the dataset from https://github.com/VisDrone/VisDrone-Dataset and
unpack it so that ``ROOT/VisDrone2019-DET-train/{images,annotations}`` exists.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
           "awning-tricycle", "bus", "motor"]
SPLITS = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val",
          "test": "VisDrone2019-DET-test-dev"}


def parse_annotation(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (boxes xywh pixels, class ids 0-9) for the kept objects."""
    boxes, classes = [], []
    for line in text.strip().splitlines():
        fields = [f for f in line.strip().split(",") if f != ""]
        if len(fields) < 6:
            continue
        x, y, w, h, score, cat = (int(float(v)) for v in fields[:6])
        if score == 0 or not 1 <= cat <= 10 or w <= 0 or h <= 0:
            continue
        boxes.append((x, y, w, h))
        classes.append(cat - 1)
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4), np.asarray(classes, dtype=np.int64)


def to_yolo_lines(boxes: np.ndarray, classes: np.ndarray, img_w: int, img_h: int) -> list[str]:
    """Normalised ``cls cx cy w h`` lines, clipped to the image."""
    lines = []
    for (x, y, w, h), c in zip(boxes, classes, strict=True):
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(img_w), x + w), min(float(img_h), y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        cx, cy = (x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h
        lines.append(f"{c} {cx:.6f} {cy:.6f} {(x1 - x0) / img_w:.6f} {(y1 - y0) / img_h:.6f}")
    return lines


def _image_size(path: Path) -> tuple[int, int]:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"cannot read {path}")
    return img.shape[1], img.shape[0]


def convert(root: str | Path, out: str | Path, copy: bool = False) -> Path:
    root, out = Path(root), Path(out)
    for split, folder in SPLITS.items():
        src = root / folder
        if not src.exists():
            print(f"skip {split}: {src} not found")
            continue
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_obj = 0
        images = sorted((src / "images").glob("*.jpg"))
        for img in images:
            ann = src / "annotations" / f"{img.stem}.txt"
            boxes, classes = parse_annotation(ann.read_text()) if ann.exists() else (np.zeros((0, 4)), np.zeros(0))
            w, h = _image_size(img)
            lines = to_yolo_lines(boxes, classes.astype(np.int64), w, h)
            n_obj += len(lines)
            (out / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines))
            dst = out / "images" / split / img.name
            if not dst.exists():
                shutil.copy2(img, dst) if copy else os.symlink(img.resolve(), dst)
        print(f"{split}: {len(images)} images, {n_obj} objects")
    yaml = out / "visdrone.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    yaml.write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n"
    )
    return yaml


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("out")
    p.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    a = p.parse_args()
    print("wrote", convert(a.root, a.out, a.copy))


if __name__ == "__main__":
    main()
