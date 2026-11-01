"""Detect + track a video and write an annotated copy with per-class unique counts."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from .tiling import sliced_predict
from .tracking import ByteTracker
from .visdrone import CLASSES


def _color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(64, 256, size=3))


def process(source: str, out: str, detector, tile: int | None = None, names=CLASSES) -> dict:
    import cv2

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise OSError(f"cannot open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    tracker = ByteTracker()
    seen: dict[int, set[int]] = defaultdict(set)
    frames, t0 = 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = sliced_predict(frame, detector, tile) if tile else detector(frame)
        for t in tracker.update(dets):
            seen[t.cls].add(t.id)
            x0, y0, x1, y1 = (int(v) for v in t.box)
            cv2.rectangle(frame, (x0, y0), (x1, y1), _color(t.id), 2)
            label = f"{names[t.cls] if t.cls < len(names) else t.cls} #{t.id}"
            cv2.putText(frame, label, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _color(t.id), 1)
        writer.write(frame)
        frames += 1
    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t0
    return {
        "frames": frames,
        "fps": frames / elapsed if elapsed else 0.0,
        "unique_counts": {names[c] if c < len(names) else str(c): len(ids) for c, ids in sorted(seen.items())},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("--out", default="tracked.mp4")
    p.add_argument("--weights", required=True, help=".onnx for ONNX Runtime, otherwise an Ultralytics .pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.1, help="keep low scores: the tracker uses them")
    p.add_argument("--tile", type=int, default=None, help="enable sliced inference with this tile size")
    a = p.parse_args()
    if a.weights.endswith(".onnx"):
        from .detectors import OnnxDetector

        det = OnnxDetector(a.weights, a.imgsz, a.conf)
    else:
        from .detectors import UltralyticsDetector

        det = UltralyticsDetector(a.weights, a.imgsz, a.conf)
    print(process(a.source, a.out, det, a.tile))


if __name__ == "__main__":
    main()
