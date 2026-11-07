"""Camera geometry: pixel rays, ground projection, and GPS geotagging.

Frames
------
* World: local ENU (x east, y north, z up) in metres. The road is the plane z = 0,
  so the drone's z coordinate is its height above ground.
* Body: x right, y forward, z up, rotated from world by heading (compass yaw,
  clockwise from north), then pitch (nose up positive) and roll (right wing down positive).
* Camera: OpenCV convention, x right, y down, z along the optical axis.

Two mounts are supported: ``"nadir"`` (gimbal pointing straight down, image top
towards the nose) for the road camera, and ``"forward"`` for the obstacle camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# WGS84
_A = 6378137.0
_E2 = 6.69437999014e-3


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_sensor(cls, focal_mm: float, sensor_width_mm: float, width: int, height: int) -> Intrinsics:
        """Square pixels, principal point at the image centre."""
        f = focal_mm / sensor_width_mm * width
        return cls(f, f, width / 2, height / 2, width, height)

    def crop(self, x0: int, y0: int, width: int, height: int) -> Intrinsics:
        return Intrinsics(self.fx, self.fy, self.cx - x0, self.cy - y0, width, height)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


@dataclass(frozen=True)
class Pose:
    """Drone pose in the local ENU frame. Angles in radians."""

    east: float
    north: float
    alt: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.east, self.north, self.alt], dtype=np.float64)


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Columns are the camera axes expressed in the body frame.
_CAM_IN_BODY = {
    "nadir": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64).T,
    "forward": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64).T,
}


def camera_to_world(pose: Pose, mount: str = "nadir") -> np.ndarray:
    """Rotation R such that ``v_world = R @ v_camera``."""
    if mount not in _CAM_IN_BODY:
        raise ValueError(f"mount must be one of {list(_CAM_IN_BODY)}")
    body = _rz(-pose.yaw) @ _rx(pose.pitch) @ _ry(pose.roll)
    return body @ _CAM_IN_BODY[mount]


def pixel_rays(intr: Intrinsics, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Camera-frame ray directions with z = 1, shape (..., 3)."""
    u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
    return np.stack([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones_like(u)], axis=-1)


def ground_hit(intr: Intrinsics, pose: Pose, u, v, mount: str = "nadir") -> tuple[np.ndarray, np.ndarray]:
    """Intersect pixel rays with the road plane.

    Returns ``(points_enu, z_cam)``: the ground points (..., 3) and the camera-frame
    depth of each hit, which is what a depth sensor or a metric depth model reports.
    Rays that point at or above the horizon get NaN.
    """
    rays = pixel_rays(intr, u, v)
    d = rays @ camera_to_world(pose, mount).T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d[..., 2] < -1e-9, -pose.alt / d[..., 2], np.nan)
    return pose.position + t[..., None] * d, t  # rays have z_cam = 1, so t is the depth


def project(intr: Intrinsics, pose: Pose, points_enu: np.ndarray,
            mount: str = "nadir") -> tuple[np.ndarray, np.ndarray]:
    """World points (N, 3) to pixels (N, 2) and camera depth (N,)."""
    pc = (np.asarray(points_enu, np.float64) - pose.position) @ camera_to_world(pose, mount)
    z = pc[:, 2]
    return np.stack([intr.fx * pc[:, 0] / z + intr.cx, intr.fy * pc[:, 1] / z + intr.cy], axis=1), z


def gsd(intr: Intrinsics, alt: float) -> float:
    """Ground sampling distance at nadir, metres per pixel."""
    return alt / intr.fx


def _radii(lat_rad: float) -> tuple[float, float]:
    s2 = np.sin(lat_rad) ** 2
    meridian = _A * (1 - _E2) / (1 - _E2 * s2) ** 1.5
    normal = _A / np.sqrt(1 - _E2 * s2)
    return meridian, normal


def geodetic_to_enu(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Small-area tangent-plane approximation; error is centimetres over a few km."""
    m, n = _radii(np.radians(lat0))
    return (np.radians(lon - lon0) * n * np.cos(np.radians(lat0)), np.radians(lat - lat0) * m)


def enu_to_geodetic(east, north, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    m, n = _radii(np.radians(lat0))
    lat = lat0 + np.degrees(np.asarray(north) / m)
    lon = lon0 + np.degrees(np.asarray(east) / (n * np.cos(np.radians(lat0))))
    return lat, lon
