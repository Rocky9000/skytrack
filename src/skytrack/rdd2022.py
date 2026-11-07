"""Convert RDD2022 (Road Damage Dataset 2022) annotations to YOLO format.

RDD2022 ships Pascal VOC XML per image with damage codes D00 (longitudinal
crack), D10 (transverse crack), D20 (alligator crack) and D40 (pothole). Some
country subsets add other labels (repairs, block cracks); those are dropped.

    python -m skytrack.rdd2022 /data/RDD2022 /data/rdd-yolo --countries India Japan
    python -m skytrack.rdd2022 /data/RDD2022 /data/rdd-yolo --potholes-only

Expected input layout: ``<root>/<Country>/train/{images,annotations/xmls}``.
The test splits have no public labels, so a validation split is carved out of train.
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ["D00", "D10", "D20", "D40"]
NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]


def parse_voc(xml_text: str, classes: list[str] = CLASSES) -> tuple[int, int, list[tuple]]:
    """Returns (width, height, [(class_index, x0, y0, x1, y1), ...]) keeping only known classes."""
    root = ET.fromstring(xml_text)
    w = int(float(root.findtext("size/width")))
    h = int(float(root.findtext("size/height")))
    objs = []
    for o in root.iter("object"):
        name = (o.findtext("name") or "").strip()
        if name not in classes:
            continue
        b = o.find("bndbox")
        x0, y0, x1, y1 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        if x1 > x0 and y1 > y0:
            objs.append((classes.index(name), x0, y0, x1, y1))
    return w, h, objs


def to_yolo(w: int, h: int, objs) -> list[str]:
    lines = []
    for c, x0, y0, x1, y1 in objs:
        x0, x1 = max(0.0, x0), min(float(w), x1)
        y0, y1 = max(0.0, y0), min(float(h), y1)
        lines.append(f"{c} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return lines


def convert(src: Path, dst: Path, countries: list[str] | None, potholes_only: bool, val_frac: float = 0.1,
            seed: int = 0) -> dict:
    classes = ["D40"] if potholes_only else CLASSES
    names = ["pothole"] if potholes_only else NAMES
    items = []
    for country in sorted(p for p in src.iterdir() if p.is_dir()):
        if countries and country.name not in countries:
            continue
        for xml in sorted((country / "train" / "annotations" / "xmls").glob("*.xml")):
            img = country / "train" / "images" / (xml.stem + ".jpg")
            if img.exists():
                items.append((img, xml))
    random.Random(seed).shuffle(items)
    n_val = int(len(items) * val_frac)
    counts = {"train": 0, "val": 0, "boxes": 0}
    for i, (img, xml) in enumerate(items):
        split = "val" if i < n_val else "train"
        w, h, objs = parse_voc(xml.read_text(), classes)
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst / "images" / split / img.name)
        (dst / "labels" / split / f"{img.stem}.txt").write_text("\n".join(to_yolo(w, h, objs)))
        counts[split] += 1
        counts["boxes"] += len(objs)
    yaml = [f"path: {dst.resolve()}", "train: images/train", "val: images/val", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
    (dst / "rdd2022.yaml").write_text("\n".join(yaml) + "\n")
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--countries", nargs="*", help="subset, e.g. India Japan Czech Norway")
    p.add_argument("--potholes-only", action="store_true", help="keep only D40")
    p.add_argument("--val-frac", type=float, default=0.1)
    a = p.parse_args()
    print(convert(a.src, a.dst, a.countries, a.potholes_only, a.val_frac))


if __name__ == "__main__":
    main()
