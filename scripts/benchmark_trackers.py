"""Tracker ablation on synthetic aerial scenes (CPU, ~1 minute, fully reproducible).

Compares:
  * IoU tracker           - greedy frame-to-frame matching, no motion model
  * SORT (Kalman only)    - ByteTracker with the low-score pass disabled
  * ByteTrack (ours)      - Kalman + Hungarian + low-score second association

Usage: python scripts/benchmark_trackers.py [--seeds 10]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from skytrack.metrics import mot_metrics
from skytrack.synthetic import SceneConfig, make_scene, run_tracker
from skytrack.tracking import ByteTracker, IoUTracker

TRACKERS = {
    "IoU tracker": lambda: IoUTracker(iou_thresh=0.3, score_thresh=0.5),
    "SORT (Kalman, no low-score pass)": lambda: ByteTracker(high_thresh=0.5, low_thresh=0.5),
    "ByteTrack (Kalman + low-score pass)": lambda: ByteTracker(high_thresh=0.5, low_thresh=0.1),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--objects", type=int, default=25)
    a = p.parse_args()

    cfg = SceneConfig(frames=a.frames, objects=a.objects)
    scenes = [make_scene(cfg, seed) for seed in range(a.seeds)]
    print(f"{a.seeds} scenes x {a.frames} frames, {a.objects} objects each\n")
    print("| Tracker | MOTA ↑ | IDF1 ↑ | ID switches ↓ | FP ↓ | FN ↓ | ms/frame |")
    print("|---|---|---|---|---|---|---|")
    for name, factory in TRACKERS.items():
        rows, elapsed = [], 0.0
        for gt, det in scenes:
            t0 = time.perf_counter()
            hyp = run_tracker(factory(), det)
            elapsed += time.perf_counter() - t0
            rows.append(mot_metrics(gt, hyp))
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        ms = 1000 * elapsed / (a.seeds * a.frames)
        print(f"| {name} | {mean['mota']:.3f} | {mean['idf1']:.3f} | {mean['id_switches']:.1f} | "
              f"{mean['false_positives']:.0f} | {mean['false_negatives']:.0f} | {ms:.2f} |")


if __name__ == "__main__":
    main()
