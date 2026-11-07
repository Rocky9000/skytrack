"""Pothole size from a segmentation mask and a metric depth map.

The idea: fit the road surface on a ring of pixels just outside the pothole,
then measure everything relative to that plane. This makes the result
independent of small errors in drone altitude or camera tilt, because both the
road and the pothole floor are seen through the same (slightly wrong) geometry.

A monocular depth model is never exactly right at a new altitude. ``align_depth``
corrects it against the depth the camera pose predicts for the flat road
around the pothole, so absolute errors in the model cancel out.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .geometry import Intrinsics


@dataclass
class PotholeMeasurement:
    area_m2: float
    length_m: float
    width_m: float
    max_depth_m: float
    mean_depth_m: float
    volume_m3: float
    plane_rmse_m: float  # residual of the road-plane fit; large values mean the depth map is unreliable

    def to_dict(self) -> dict:
        return {k: round(float(v), 5) for k, v in asdict(self).items()}


class DepthAlignmentError(ValueError):
    pass


def align_depth(pred: np.ndarray, target: np.ndarray, valid: np.ndarray, kind: str = "metric") -> np.ndarray:
    """Map a depth model's output onto the depth the camera pose predicts for the road.

    kind="metric": ``pred`` is metric depth (e.g. Depth Anything V2 Metric-Outdoor). Only a shift
        is fitted, which absorbs the model's bias at this altitude. A residual *scale* error in
        the model carries straight into pothole depth, so it should be measured once on
        calibration flights (see the README).
    kind="relative": ``pred`` is relative inverse depth (Depth Anything V2, MiDaS). Scale and shift
        are fitted in inverse-depth space, as in the MiDaS evaluation protocol. The scale is
        only observable when the road depth varies across the crop (camera tilt, camber); over
        a flat road seen straight down it is not, and this raises ``DepthAlignmentError``
        rather than return a confident wrong answer.
    """
    valid = valid & np.isfinite(target) & (target > 0) & np.isfinite(pred)
    if valid.sum() < 10:
        raise DepthAlignmentError("need at least 10 valid road pixels to align depth")
    if kind == "metric":
        return pred + float(np.median(target[valid] - pred[valid]))
    if kind != "relative":
        raise ValueError("kind must be 'metric' or 'relative'")
    y = 1.0 / target[valid]
    if np.std(y) < 2e-3 * np.mean(y):
        raise DepthAlignmentError("road depth is nearly constant in this crop; scale is unobservable")
    A = np.stack([pred[valid], np.ones(valid.sum())], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, y, rcond=None)
    out = s * pred + t
    return np.where(out > 1e-6, 1.0 / np.maximum(out, 1e-6), np.nan)


def backproject(intr: Intrinsics, depth: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pixels with camera-frame depth (z) to 3D camera coordinates, shape (N, 3)."""
    z = depth[v, u]
    return np.stack([(u - intr.cx) / intr.fx * z, (v - intr.cy) / intr.fy * z, z], axis=1)


def fit_plane(points: np.ndarray, iters: int = 200, thresh: float = 0.01, seed: int = 0) -> tuple[np.ndarray, float]:
    """RANSAC plane, refined by least squares on the inliers.

    Returns ``(n, d)`` with unit normal ``n`` oriented towards the camera
    (``n . p + d = 0`` on the plane, ``d > 0``).
    """
    rng = np.random.default_rng(seed)
    best = np.zeros(len(points), bool)
    for _ in range(iters):
        p = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        inliers = np.abs(points @ n - p[0] @ n) < thresh
        if inliers.sum() > best.sum():
            best = inliers
    q = points[best] if best.sum() >= 3 else points
    c = q.mean(axis=0)
    n = np.linalg.svd(q - c)[2][-1]
    d = -n @ c
    if d < 0:
        n, d = -n, -d
    return n, float(d)


def measure_pothole(mask: np.ndarray, depth: np.ndarray, intr: Intrinsics, ring_px: int = 12) -> PotholeMeasurement:
    """Measure one pothole.

    Args:
        mask: boolean (H, W) pothole mask.
        depth: metric camera-frame depth (H, W), same size as ``mask``.
        intr: intrinsics of this image (use ``Intrinsics.crop`` for a crop).
        ring_px: width of the road ring used to fit the surface plane.
    """
    mask = mask.astype(bool)
    if mask.sum() < 5:
        raise ValueError("mask is empty")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    grown = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    gap = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)  # skip the broken rim
    ring = grown & ~gap & np.isfinite(depth)
    if ring.sum() < 20:
        raise ValueError("not enough road pixels around the mask; enlarge the crop")

    rv, ru = np.nonzero(ring)
    road = backproject(intr, depth, ru, rv)
    n, d = fit_plane(road, thresh=max(0.005, 0.002 * float(np.median(road[:, 2]))))
    rmse = float(np.sqrt(np.mean((road @ n + d) ** 2)))

    pv, pu = np.nonzero(mask & np.isfinite(depth))
    pts = backproject(intr, depth, pu, pv)
    below = np.clip(-(pts @ n + d), 0, None)  # distance behind the road plane

    # Footprint of each pixel on the road plane: solid angle * range^2 / cos(incidence).
    rays = np.stack([(pu - intr.cx) / intr.fx, (pv - intr.cy) / intr.fy, np.ones(len(pu))], axis=1)
    norm = np.linalg.norm(rays, axis=1)
    cos_axis = 1.0 / norm
    cos_inc = np.abs(rays @ n) / norm
    plane_range = d / np.maximum(cos_inc, 1e-6)
    pixel_area = (cos_axis**3 / (intr.fx * intr.fy)) * plane_range**2 / np.maximum(cos_inc, 1e-6)

    # Length and width: project the pothole onto the plane and take the minimum-area rectangle.
    on_plane = rays / np.maximum(-(rays @ n), 1e-9)[:, None] * d  # ray hits on the plane, camera frame
    e1 = np.cross(n, [0.0, 1.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(n, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    xy = np.stack([on_plane @ e1, on_plane @ e2], axis=1).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(xy)

    return PotholeMeasurement(
        area_m2=float(pixel_area.sum()),
        length_m=float(max(w, h)),
        width_m=float(min(w, h)),
        max_depth_m=float(np.percentile(below, 98)) if len(below) else 0.0,  # robust to single bad pixels
        mean_depth_m=float(below.mean()) if len(below) else 0.0,
        volume_m3=float((below * pixel_area).sum()),
        plane_rmse_m=rmse,
    )
