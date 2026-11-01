"""Synthetic aerial tracking scenes with realistic detector failure modes.

Used to benchmark trackers reproducibly without a GPU or a dataset
download. Objects are small (aerial scale), move with smooth random
velocity changes, cross each other, and the simulated detector:

* jitters box coordinates,
* misses objects entirely with some probability,
* reports **low confidence** during occlusion episodes (the case ByteTrack's
  second association pass exists for),
* emits false positives at random locations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boxes import Detections
from .metrics import Frame


@dataclass
class SceneConfig:
    frames: int = 300
    objects: int = 25
    width: int = 1920
    height: int = 1080
    min_size: float = 12
    max_size: float = 40
    speed: float = 6.0
    jitter: float = 1.5
    miss_prob: float = 0.08
    occlusion_rate: float = 0.02  # chance per frame that an occlusion episode starts
    occlusion_len: tuple[int, int] = (5, 20)
    occluded_score: tuple[float, float] = (0.15, 0.45)
    false_positives_per_frame: float = 1.0
    entry_spread: float = 0.5  # fraction of the clip over which objects appear


def make_scene(cfg: SceneConfig, seed: int = 0) -> tuple[list[Frame], list[Detections]]:
    rng = np.random.default_rng(seed)
    n = cfg.objects
    pos = rng.uniform([0, 0], [cfg.width, cfg.height], size=(n, 2))
    vel = rng.normal(0, cfg.speed, size=(n, 2))
    width = rng.uniform(cfg.min_size, cfg.max_size, size=n)
    size = np.stack([width, width * rng.uniform(0.6, 1.4, size=n)], axis=1)
    start = rng.integers(0, max(1, int(cfg.frames * cfg.entry_spread)), size=n)
    occluded_until = np.full(n, -1)

    gt: list[Frame] = []
    det: list[Detections] = []
    for f in range(cfg.frames):
        vel = 0.95 * vel + rng.normal(0, cfg.speed * 0.15, size=(n, 2))
        pos = pos + vel
        # bounce off frame edges so objects stay in view
        for axis, limit in ((0, cfg.width), (1, cfg.height)):
            out = (pos[:, axis] < 0) | (pos[:, axis] > limit)
            vel[out, axis] *= -1
            pos[:, axis] = pos[:, axis].clip(0, limit)

        frame_gt: Frame = {}
        boxes, scores = [], []
        for i in range(n):
            if f < start[i]:
                continue
            w, h = size[i]
            box = np.array([pos[i, 0] - w / 2, pos[i, 1] - h / 2, pos[i, 0] + w / 2, pos[i, 1] + h / 2])
            frame_gt[i + 1] = box
            if occluded_until[i] < f and rng.random() < cfg.occlusion_rate:
                occluded_until[i] = f + rng.integers(*cfg.occlusion_len)
            if rng.random() < cfg.miss_prob:
                continue
            noisy = box + rng.normal(0, cfg.jitter, size=4)
            score = rng.uniform(*cfg.occluded_score) if occluded_until[i] >= f else rng.uniform(0.6, 0.95)
            boxes.append(noisy)
            scores.append(score)
        for _ in range(rng.poisson(cfg.false_positives_per_frame)):
            c = rng.uniform([0, 0], [cfg.width, cfg.height])
            s = rng.uniform(cfg.min_size, cfg.max_size)
            boxes.append(np.r_[c - s / 2, c + s / 2])
            scores.append(rng.uniform(0.3, 0.7))
        gt.append(frame_gt)
        det.append(Detections(np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32),
                              np.zeros(len(scores), np.int64)))
    return gt, det


def run_tracker(tracker, detections: list[Detections]) -> list[Frame]:
    return [{t.id: t.box for t in tracker.update(d)} for d in detections]
