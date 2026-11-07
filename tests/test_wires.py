import cv2
import numpy as np

from skytrack.geometry import Intrinsics, Pose, project
from skytrack.wires import mask_segments, wires_from_stereo

INTR = Intrinsics(800, 800, 640, 360, 1280, 720)
BASELINE = 0.12


def render_wire(pose: Pose, p0, p1, rng, dropout=0.6, noise=0.3):
    pts = np.linspace(p0, p1, 400)
    uv, z = project(INTR, pose, pts, mount="forward")
    mask = np.zeros((INTR.height, INTR.width), np.uint8)
    disp = np.zeros((INTR.height, INTR.width), np.float32)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < INTR.width) & (uv[:, 1] >= 0) & (uv[:, 1] < INTR.height) & (z > 0)
    uv, z = uv[inside], z[inside]
    cv2.polylines(mask, [np.round(uv).astype(np.int32)], False, 255, 2)
    for (u, v), zz in zip(np.round(uv).astype(int), z, strict=True):
        if rng.random() > dropout:  # most wire pixels have no valid stereo match
            disp[v, u] = INTR.fx * BASELINE / zz + rng.normal(0, noise)
    disp[rng.random(disp.shape) < 0.002] = rng.uniform(1, 40)  # speckle elsewhere
    return mask, disp


def test_hough_merges_duplicates():
    mask = np.zeros((200, 400), np.uint8)
    cv2.line(mask, (10, 100), (390, 120), 255, 3)
    assert len(mask_segments(mask)) == 1


def _offsets(wire, p0, p1):
    d = p1 - p0
    return [np.linalg.norm(np.cross(q - p0, d)) / np.linalg.norm(d) for q in (wire.p0, wire.p1)]


POSE = Pose(0, 0, 10, yaw=0.2)
P0, P1 = np.array([-6.0, 14.0, 10.8]), np.array([6.0, 18.0, 10.2])  # 12-19 m from the camera


def test_recovers_wire_exactly_from_clean_stereo():
    mask, disp = render_wire(POSE, P0, P1, np.random.default_rng(0), dropout=0.0, noise=0.0)
    (w,) = wires_from_stereo(mask, disp, INTR, BASELINE, POSE)
    assert max(_offsets(w, P0, P1)) < 0.05


def test_recovers_wire_from_sparse_noisy_stereo():
    # 60 % of wire pixels unmatched, 0.3 px disparity noise, random speckle elsewhere
    for seed in range(3):
        mask, disp = render_wire(POSE, P0, P1, np.random.default_rng(seed))
        wires = wires_from_stereo(mask, disp, INTR, BASELINE, POSE)
        assert len(wires) == 1
        rng_m = np.linalg.norm(wires[0].p1 - POSE.position)
        # stereo depth error grows with range^2; at ~19 m with a 12 cm baseline, 3 % of range is realistic
        assert max(_offsets(wires[0], P0, P1)) < 0.03 * rng_m
