"""3D wire segments from a wire mask and a stereo disparity map.

Wires are 1-2 px wide, so per-pixel stereo depth on them is noisy and full of
holes. Instead of trusting individual pixels, this uses a geometric fact: along
the image of a straight 3D line, inverse depth (and so disparity) changes
*linearly* with position. A robust line fit to disparity along each segment
therefore recovers the wire's depth at both ends even when most pixels are bad.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Intrinsics, Pose, camera_to_world


@dataclass
class Wire:
    p0: np.ndarray  # world ENU endpoints, metres
    p1: np.ndarray
    support: float  # fraction of samples consistent with the disparity fit

    def as_segment(self) -> np.ndarray:
        return np.stack([self.p0, self.p1])


def mask_segments(mask: np.ndarray, min_length: int = 40) -> np.ndarray:
    """Straight segments (N, 4) as x0, y0, x1, y1 from a binary wire mask."""
    m = (mask > 0).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(m, 1, np.pi / 360, threshold=30, minLineLength=min_length, maxLineGap=15)
    if lines is None:
        return np.zeros((0, 4))
    segs = lines.reshape(-1, 4).astype(np.float64)
    keep: list[np.ndarray] = []
    for s in sorted(segs, key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1])):
        if not any(_same_line(s, k) for k in keep):  # Hough returns many near-duplicates per wire
            keep.append(s)
    return np.asarray(keep)


def _same_line(a: np.ndarray, b: np.ndarray, angle_tol: float = np.radians(3), dist_tol: float = 6.0) -> bool:
    da, db = a[2:] - a[:2], b[2:] - b[:2]
    cos = abs(da @ db) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-9)
    if cos < np.cos(angle_tol):
        return False
    n = np.array([-db[1], db[0]]) / (np.linalg.norm(db) + 1e-9)
    return abs((a[:2] - b[:2]) @ n) < dist_tol and abs((a[2:] - b[:2]) @ n) < dist_tol


def _robust_line(s: np.ndarray, d: np.ndarray, iters: int = 100, tol: float = 0.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    best = np.zeros(len(s), bool)
    for _ in range(iters):
        i, j = rng.choice(len(s), 2, replace=False)
        if abs(s[j] - s[i]) < 1e-6:
            continue
        k = (d[j] - d[i]) / (s[j] - s[i])
        inl = np.abs(d - (d[i] + k * (s - s[i]))) < tol
        if inl.sum() > best.sum():
            best = inl
    k, b = np.polyfit(s[best], d[best], 1)
    return k, b, best.mean()


def wires_from_stereo(mask: np.ndarray, disparity: np.ndarray, intr: Intrinsics, baseline_m: float,
                      pose: Pose, min_length: int = 40, min_disparity: float = 0.5) -> list[Wire]:
    """Recover wires in world coordinates from the forward stereo camera (left image frame)."""
    R = camera_to_world(pose, "forward")
    out: list[Wire] = []
    for x0, y0, x1, y1 in mask_segments(mask, min_length):
        n = int(np.hypot(x1 - x0, y1 - y0))
        s = np.linspace(0.0, 1.0, n)
        u = np.round(x0 + s * (x1 - x0)).astype(int)
        v = np.round(y0 + s * (y1 - y0)).astype(int)
        # look one pixel either side: the segment from Hough can sit a pixel off a thin wire
        cand = np.stack([disparity[np.clip(v + dv, 0, disparity.shape[0] - 1), u] for dv in (-1, 0, 1)])
        cand = np.where(cand > min_disparity, cand, np.nan)
        ok = np.isfinite(cand).any(axis=0)
        d = np.full(len(s), np.nan)
        d[ok] = np.nanmedian(cand[:, ok], axis=0)
        if ok.sum() < 10:
            continue
        k, b, support = _robust_line(s[ok], d[ok])
        ends = []
        for t, (u_, v_) in ((0.0, (x0, y0)), (1.0, (x1, y1))):
            disp = b + k * t
            if disp <= min_disparity:
                break
            z = intr.fx * baseline_m / disp
            pc = np.array([(u_ - intr.cx) / intr.fx * z, (v_ - intr.cy) / intr.fy * z, z])
            ends.append(pose.position + R @ pc)
        if len(ends) == 2:
            out.append(Wire(ends[0], ends[1], float(support)))
    return out
