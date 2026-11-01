"""Constant-velocity Kalman filter over (cx, cy, aspect, height), as in SORT/ByteTrack.

State: [cx, cy, a, h, vx, vy, va, vh]. Process and measurement noise scale
with box height, so a 15 px car and a 300 px truck get comparable relative
uncertainty, which matters in aerial footage where scale varies wildly.
"""

from __future__ import annotations

import numpy as np

_F = np.eye(8)
_F[:4, 4:] = np.eye(4)
_H = np.eye(4, 8)

W_POS = 1.0 / 20
W_VEL = 1.0 / 160


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def xyah_to_xyxy(m: np.ndarray) -> np.ndarray:
    cx, cy, a, h = m[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)


def initiate(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = z[3]
    mean = np.r_[z, np.zeros(4)]
    std = [2 * W_POS * h, 2 * W_POS * h, 1e-2, 2 * W_POS * h,
           10 * W_VEL * h, 10 * W_VEL * h, 1e-5, 10 * W_VEL * h]
    return mean, np.diag(np.square(std))


def predict(mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = mean[3]
    std = [W_POS * h, W_POS * h, 1e-2, W_POS * h, W_VEL * h, W_VEL * h, 1e-5, W_VEL * h]
    return _F @ mean, _F @ cov @ _F.T + np.diag(np.square(std))


def update(mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = mean[3]
    r = np.diag(np.square([W_POS * h, W_POS * h, 1e-1, W_POS * h]))
    s = _H @ cov @ _H.T + r
    k = np.linalg.solve(s, (cov @ _H.T).T).T  # cov H^T S^-1 without an explicit inverse
    mean = mean + k @ (z - _H @ mean)
    cov = (np.eye(8) - k @ _H) @ cov
    return mean, cov
