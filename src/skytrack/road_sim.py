"""Synthetic nadir flights over a road with potholes of known size.

Every pixel is ray-traced against the ground, so the rendered image, the
depth map and the ground-truth pothole geometry are exactly consistent with
the camera model. This is what the measurement, geotagging and de-duplication
code is tested against. It says nothing about how well a trained detector
works on real roads; that needs RDD2022 or your own footage.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .boxes import Detections
from .geometry import Intrinsics, Pose, ground_hit, project


@dataclass
class Pothole:
    east: float
    north: float
    a: float  # semi-axes, m
    b: float
    angle: float  # rad
    depth: float  # max depth, m

    @property
    def area(self) -> float:
        return float(np.pi * self.a * self.b)

    @property
    def volume(self) -> float:
        return 0.5 * self.area * self.depth  # paraboloid bowl

    def depth_at(self, e: np.ndarray, n: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.angle), np.sin(self.angle)
        x, y = (e - self.east) * c + (n - self.north) * s, -(e - self.east) * s + (n - self.north) * c
        r2 = (x / self.a) ** 2 + (y / self.b) ** 2
        return np.where(r2 < 1, self.depth * (1 - r2), 0.0)

    def outline(self, k: int = 32) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, k, endpoint=False)
        c, s = np.cos(self.angle), np.sin(self.angle)
        x, y = self.a * np.cos(t), self.b * np.sin(t)
        return np.stack([self.east + x * c - y * s, self.north + x * s + y * c, np.zeros(k)], axis=1)


def random_potholes(n: int, length_m: float, road_half_width: float = 3.0, min_gap_m: float = 1.5,
                   seed: int = 0) -> list[Pothole]:
    """Random potholes along a road running north from the origin, at least ``min_gap_m`` apart."""
    rng = np.random.default_rng(seed)
    out: list[Pothole] = []
    while len(out) < n:
        a = rng.uniform(0.15, 0.6)
        p = Pothole(rng.uniform(-road_half_width + 0.8, road_half_width - 0.8), rng.uniform(5, length_m - 5), a,
                    a * rng.uniform(0.5, 1.0), rng.uniform(0, np.pi), rng.uniform(0.02, 0.12))
        if all(np.hypot(p.east - q.east, p.north - q.north) > q.a + p.a + min_gap_m for q in out):
            out.append(p)
    return sorted(out, key=lambda p: p.north)


def _texture(e: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Deterministic asphalt-like noise fixed to the ground, so frames agree with each other."""
    i, j = np.floor(e / 0.02).astype(np.int64), np.floor(n / 0.02).astype(np.int64)
    h = (i * 73856093) ^ (j * 19349663)
    return ((h % 1000) / 1000.0 - 0.5) * 30


def render(intr: Intrinsics, pose: Pose, potholes: list[Pothole], road_half_width: float = 3.5):
    """Grayscale BGR image and camera-frame depth (H, W) for one frame."""
    v, u = np.mgrid[0 : intr.height, 0 : intr.width]
    pts, z = ground_hit(intr, pose, u + 0.5, v + 0.5)
    e, n = pts[..., 0], pts[..., 1]
    hole = np.zeros_like(z)
    for p in potholes:
        if abs(p.north - pose.north) < 20:
            hole = np.maximum(hole, p.depth_at(e, n))
    # A point `hole` metres below the plane along the same ray: z scales with the vertical distance.
    z_true = z * (pose.alt + hole) / pose.alt
    img = 105 + _texture(e, n)
    img = np.where(np.abs(e) > road_half_width, img * 0.6 + 60, img)  # verge
    img = img - np.where(hole > 0, 50 + 400 * hole, 0)  # exposed base course is darker, more so when deeper
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), z_true.astype(np.float32)


def visible_boxes(intr: Intrinsics, pose: Pose, potholes: list[Pothole]) -> tuple[np.ndarray, list[int]]:
    boxes, ids = [], []
    for i, p in enumerate(potholes):
        uv, z = project(intr, pose, p.outline())
        if (z <= 0).any():
            continue
        x0, y0 = uv.min(0)
        x1, y1 = uv.max(0)
        if x1 < 0 or y1 < 0 or x0 >= intr.width or y0 >= intr.height:
            continue
        boxes.append([max(x0, 0), max(y0, 0), min(x1, intr.width - 1), min(y1, intr.height - 1)])
        ids.append(i)
    return np.asarray(boxes, np.float64).reshape(-1, 4), ids


class SimulatedDetector:
    """Stands in for YOLO on synthetic frames: true boxes plus jitter, misses and false positives."""

    def __init__(self, intr: Intrinsics, potholes: list[Pothole], miss_prob: float = 0.1, jitter_px: float = 3.0,
                 false_pos_rate: float = 0.05, seed: int = 0):
        self.intr, self.potholes = intr, potholes
        self.miss, self.jitter, self.fp = miss_prob, jitter_px, false_pos_rate
        self.rng = np.random.default_rng(seed)
        self.pose: Pose | None = None

    def __call__(self, frame: np.ndarray) -> Detections:
        boxes, _ = visible_boxes(self.intr, self.pose, self.potholes)
        keep = self.rng.random(len(boxes)) > self.miss
        boxes = boxes[keep] + self.rng.normal(0, self.jitter, (keep.sum(), 4))
        scores = self.rng.uniform(0.55, 0.95, len(boxes))
        if self.rng.random() < self.fp:
            c = self.rng.uniform([0, 0], [self.intr.width, self.intr.height])
            boxes = np.vstack([boxes, np.r_[c - 15, c + 15]])
            scores = np.r_[scores, self.rng.uniform(0.6, 0.8)]
        return Detections(boxes.astype(np.float32), scores.astype(np.float32), np.zeros(len(scores), np.int64))


def threshold_segmenter(crop_bgr: np.ndarray) -> np.ndarray:
    """Stands in for DeepLabv3+: dark pixels in the crop, largest connected blob."""
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    m = (g < 80).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros(g.shape, np.float32)
    return (lab == 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())).astype(np.float32)


class SimulatedDepth:
    """Stands in for a metric monocular depth model: true depth with a per-frame bias, a small
    scale error and pixel noise. Set ``kind="relative"`` to instead emulate a relative model
    (inverse depth with an arbitrary scale and shift)."""

    def __init__(self, kind: str = "metric", scale_err: float = 0.05, bias_m: float = 0.5, noise_m: float = 0.002,
                 seed: int = 0):
        self.kind, self.scale_err, self.bias, self.noise = kind, scale_err, bias_m, noise_m
        self.rng = np.random.default_rng(seed)
        self.depth: np.ndarray | None = None  # full-frame truth, set by the caller each frame
        self.offset = (0, 0)

    def __call__(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        y0, x0 = self.offset
        z = self.depth[y0 : y0 + h, x0 : x0 + w].astype(np.float64)
        if self.kind == "relative":
            inv = 1.0 / z
            return (self.rng.uniform(5, 50) * inv + self.rng.uniform(-0.1, 0.1)).astype(np.float32)
        s = 1 + self.rng.uniform(-self.scale_err, self.scale_err)
        b = self.rng.uniform(-self.bias, self.bias)
        return (s * z + b + self.rng.normal(0, self.noise, z.shape)).astype(np.float32)
