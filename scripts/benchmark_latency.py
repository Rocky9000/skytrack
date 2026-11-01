"""Latency of full-frame vs sliced inference for an exported ONNX model.

    python scripts/benchmark_latency.py --weights best.onnx --image frame.jpg --imgsz 640 --tile 640
"""

from __future__ import annotations

import argparse
import statistics
import time

import cv2

from skytrack.detectors import OnnxDetector
from skytrack.tiling import make_tiles, sliced_predict


def timed(fn, runs: int) -> tuple[float, float, int]:
    fn()  # warm-up
    times, n = [], 0
    for _ in range(runs):
        t0 = time.perf_counter()
        n = len(fn())
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), sorted(times)[int(0.95 * (runs - 1))], n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--runs", type=int, default=20)
    a = p.parse_args()

    img = cv2.imread(a.image)
    det = OnnxDetector(a.weights, a.imgsz, conf=0.25)
    tiles = len(make_tiles(img.shape[1], img.shape[0], a.tile, a.overlap))
    print(f"image {img.shape[1]}x{img.shape[0]}, {tiles} tiles\n")
    print("| Mode | p50 ms | p95 ms | detections |\n|---|---|---|---|")
    for name, fn in [("full frame", lambda: det(img)),
                     (f"sliced ({tiles} tiles + full)", lambda: sliced_predict(img, det, a.tile, a.overlap))]:
        p50, p95, n = timed(fn, a.runs)
        print(f"| {name} | {p50:.1f} | {p95:.1f} | {n} |")


if __name__ == "__main__":
    main()
