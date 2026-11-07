import numpy as np
import pytest

from skytrack.geometry import (
    Intrinsics,
    Pose,
    camera_to_world,
    enu_to_geodetic,
    geodetic_to_enu,
    ground_hit,
    gsd,
    project,
)

INTR = Intrinsics(1000, 1000, 640, 360, 1280, 720)


def test_rotations_are_orthonormal():
    for mount in ("nadir", "forward"):
        R = camera_to_world(Pose(0, 0, 10, yaw=0.7, pitch=0.1, roll=-0.2), mount)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)


@pytest.mark.parametrize("yaw", [0.0, 0.5, np.pi / 2, 3.0])
def test_nadir_centre_ray_hits_ground_below_drone(yaw):
    pose = Pose(12.0, -4.0, 25.0, yaw=yaw)
    pt, z = ground_hit(INTR, pose, INTR.cx, INTR.cy)
    np.testing.assert_allclose(pt, [12.0, -4.0, 0.0], atol=1e-9)
    assert z == pytest.approx(25.0)


def test_image_top_points_along_heading():
    north = ground_hit(INTR, Pose(0, 0, 10, yaw=0.0), INTR.cx, 0)[0]
    east = ground_hit(INTR, Pose(0, 0, 10, yaw=np.pi / 2), INTR.cx, 0)[0]
    assert north[1] > 3 and abs(north[0]) < 1e-9
    assert east[0] > 3 and abs(east[1]) < 1e-9


def test_pitch_moves_footprint_forward():
    pitch = np.radians(10)
    pt, _ = ground_hit(INTR, Pose(0, 0, 10, pitch=pitch), INTR.cx, INTR.cy)
    assert pt[1] == pytest.approx(10 * np.tan(pitch))


def test_project_inverts_ground_hit():
    pose = Pose(3, 7, 15, yaw=0.3, pitch=0.05, roll=-0.04)
    u, v = np.array([10.0, 640, 1200]), np.array([20.0, 360, 700])
    pts, _ = ground_hit(INTR, pose, u, v)
    uv, _ = project(INTR, pose, pts)
    np.testing.assert_allclose(uv, np.stack([u, v], 1), atol=1e-6)


def test_gsd_and_sensor_intrinsics():
    intr = Intrinsics.from_sensor(focal_mm=8.8, sensor_width_mm=13.2, width=5472, height=3648)  # 1" sensor
    assert gsd(intr, 50) == pytest.approx(50 * 13.2 / (8.8 * 5472))


def test_geodetic_roundtrip_and_scale():
    lat0, lon0 = 17.385, 78.4867
    lat, lon = enu_to_geodetic(100.0, 250.0, lat0, lon0)
    e, n = geodetic_to_enu(lat, lon, lat0, lon0)
    assert (e, n) == pytest.approx((100.0, 250.0), abs=1e-6)
    assert (lat - lat0) * 110_700 == pytest.approx(250, rel=0.01)  # ~110.7 km per degree of latitude here
